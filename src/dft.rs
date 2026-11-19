//! Closed-shell Kohn–Sham DFT: numerical XC energy and potential on the molecular
//! grid (theory.typ section 8: eq:rho, eq:exc, eq:vxc) and the RKS driver. UNCHECKED(Johnson1993) VERIFIED(Kohn1965, Eq. (2.3), p. A1133; continuous LDA integral only, quadrature not in paper)
//!
//! Grid points are processed per [`GridBlock`] (one radial shell of one atom) in
//! parallel. For every block only the shells that can be non-negligible on it are
//! evaluated: a shell centred at distance R from the block's atom is at least
//! |r_i − R| away from every point of the block, and it is dropped when that is
//! beyond its [`shell_cutoff_radius`] for |χ|, |∇χ|, |∇∇χ| < 10⁻¹⁵ — a rigorous bound,
//! so the screening has no measurable effect. AO values of as many blocks as fit
//! the memory budget are computed once and reused in every SCF iteration.

use crate::ao_eval::{AoValues, D_X, eval_ao_shells, shell_cutoff_radius};
use crate::basis::Basis;
use crate::error::Result;
use crate::functional::Functional;
use crate::grid::{GridSettings, MolecularGrid};
use crate::integrals::ExchangeKernel;
use crate::linalg::Matrix;
use crate::molecule::Molecule;
use crate::scf::{ExchangeCorrelation, ScfOptions, ScfResult, XcTerm, run_scf};
use rayon::prelude::*;
use std::time::Instant;

/// Points with ρ below this contribute nothing to E_xc and V^xc. theory.typ section 8.3
pub const RHO_CUTOFF: f64 = 1e-10;
/// σ is raised to at least this value (derivatives of √σ). theory.typ section 8.3
pub const SIGMA_FLOOR: f64 = 1e-40;
/// Screening threshold on |χ|, |∇χ| and |∇∇χ| (see the module documentation).
pub const AO_SCREENING_EPS: f64 = 1e-15;

/// Tuning knobs of the XC integration that do not change the result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XcOptions {
    /// Memory budget (bytes) for AO values cached across SCF iterations; blocks
    /// beyond it are recomputed on every evaluation.
    pub ao_cache_bytes: usize,
}

impl Default for XcOptions {
    fn default() -> Self {
        XcOptions { ao_cache_bytes: 1 << 30 }
    }
}

/// E_xc, V^xc and the integrated electron count for one density matrix.
#[derive(Debug, Clone)]
pub struct XcEvaluation {
    /// E_xc = Σ_g w_g f(ρ_g, σ_g). theory.typ eq:exc
    pub energy: f64,
    /// V^xc_μν. theory.typ eq:vxc
    pub potential: Matrix,
    /// Σ_g w_g ρ_g (≈ N_elec; grid-accuracy diagnostic).
    pub electron_count: f64,
}

/// Time spent in the phases of the XC integration of one [`XcIntegrator`], summed
/// over its [`ExchangeCorrelation::evaluate`] calls (the SCF iterations). Timing
/// only reads the clock around each phase; it does not change the order of any
/// floating-point operation.
///
/// The per-block phases run in parallel, so their times are thread-seconds (the sum
/// over blocks of the time each block spent in that phase), not wall time; compare
/// them with `parallel_wall_seconds` × the number of threads.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub struct XcPhaseTimes {
    /// Number of evaluations included.
    pub evaluations: usize,
    /// Wall time of computing the cached AO values when the integrator was built.
    pub ao_cache_seconds: f64,
    /// Blocks whose AO values were not cached and were recomputed, summed over
    /// evaluations.
    pub recomputed_blocks: usize,
    /// AO values χ, ∇χ of the blocks outside the cache (thread-seconds).
    pub ao_seconds: f64,
    /// Gathering the block's sub-matrix of P (thread-seconds).
    pub gather_seconds: f64,
    /// [`density_on_points`]: t = Pχ, ρ and ∇ρ (thread-seconds).
    pub density_seconds: f64,
    /// [`xc_on_points`] with Σ w ρ and Σ w f (thread-seconds).
    pub functional_seconds: f64,
    /// Building φ_ν(g) (thread-seconds).
    pub phi_seconds: f64,
    /// M = χ φᵀ and V = M + Mᵀ on the block (thread-seconds).
    pub products_seconds: f64,
    /// Wall time of the parallel loops over blocks.
    pub parallel_wall_seconds: f64,
    /// Wall time of the sums over blocks in block order (E_xc, Σ w ρ and the
    /// scatter of the block V into V^xc, parallel over the rows of V^xc).
    pub reduction_seconds: f64,
}

impl XcPhaseTimes {
    fn add(&mut self, o: &XcPhaseTimes) {
        self.evaluations += o.evaluations;
        self.ao_cache_seconds += o.ao_cache_seconds;
        self.recomputed_blocks += o.recomputed_blocks;
        self.ao_seconds += o.ao_seconds;
        self.gather_seconds += o.gather_seconds;
        self.density_seconds += o.density_seconds;
        self.functional_seconds += o.functional_seconds;
        self.phi_seconds += o.phi_seconds;
        self.products_seconds += o.products_seconds;
        self.parallel_wall_seconds += o.parallel_wall_seconds;
        self.reduction_seconds += o.reduction_seconds;
    }
}

/// ρ, ∇ρ, f, v_ρ and v_σ at the points of one block.
#[derive(Debug, Clone, Default)]
pub struct PointData {
    /// ρ_g. theory.typ eq:rho
    pub rho: Vec<f64>,
    /// ∇ρ_g (empty unless requested). theory.typ eq:rho
    pub grad_rho: Vec<[f64; 3]>,
    /// f(ρ_g, σ_g); 0 where ρ_g < [`RHO_CUTOFF`].
    pub f: Vec<f64>,
    /// v_ρ; 0 where ρ_g < [`RHO_CUTOFF`].
    pub v_rho: Vec<f64>,
    /// v_σ; 0 where ρ_g < [`RHO_CUTOFF`] (and for LDA functionals).
    pub v_sigma: Vec<f64>,
}

/// ρ (and ∇ρ if `ao.deriv ≥ 1` and `gradient`) at the points of `ao` for the density
/// matrix `p` restricted to the AOs of `ao` (row-major `n_ao × n_ao`). Also returns
/// t_μ(g) = Σ_ν P_μν χ_ν(g) as an `n_ao × n_points` array. theory.typ eq:rho
pub fn density_on_points(ao: &AoValues, p: &[f64], gradient: bool) -> (Vec<f64>, Vec<[f64; 3]>, Vec<f64>) {
    let (na, np) = (ao.n_ao, ao.n_points);
    assert_eq!(p.len(), na * na);
    let chi = ao.component(0);
    let t = density_times_ao(p, chi, na, np);
    let mut rho = vec![0.0; np];
    for a in 0..na {
        for ((r, c), tv) in rho.iter_mut().zip(&chi[a * np..(a + 1) * np]).zip(&t[a * np..(a + 1) * np]) {
            *r += c * tv;
        }
    }
    let mut grad = Vec::new();
    if gradient {
        assert!(ao.deriv >= 1, "∇ρ needs AO gradients");
        grad = vec![[0.0; 3]; np];
        for i in 0..3 {
            let d = ao.component(D_X + i);
            for a in 0..na {
                for ((gr, c), tv) in grad.iter_mut().zip(&d[a * np..(a + 1) * np]).zip(&t[a * np..(a + 1) * np]) {
                    gr[i] += 2.0 * c * tv;
                }
            }
        }
    }
    (rho, grad, t)
}

/// f, v_ρ, v_σ at points with density `rho` and gradient `grad` (empty for LDA),
/// applying the cut-offs of theory.typ section 8.3.
pub fn xc_on_points(functional: &Functional, rho: Vec<f64>, grad: Vec<[f64; 3]>) -> PointData {
    let np = rho.len();
    let (mut f, mut vr, mut vs) = (vec![0.0; np], vec![0.0; np], vec![0.0; np]);
    for g in 0..np {
        if rho[g] < RHO_CUTOFF {
            continue;
        }
        let sigma = match grad.get(g) {
            Some(d) => (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).max(SIGMA_FLOOR),
            None => SIGMA_FLOOR,
        };
        let x = functional.eval(rho[g], sigma);
        f[g] = x.f;
        vr[g] = x.v_rho;
        vs[g] = x.v_sigma;
    }
    PointData { rho, grad_rho: grad, f, v_rho: vr, v_sigma: vs }
}

/// Rows (TR) and columns (TC) of the register tiles of [`density_times_ao`] and
/// [`ao_products`]: TR·TC = 32 accumulators, 8 AVX registers.
const TR: usize = 8;
const TC: usize = 4;
/// Independent partial sums per point index in [`ao_products`] (point g goes to
/// lane g mod LANES; the last np mod LANES points are added afterwards in order).
const LANES: usize = 8;

/// C_ij = Σ_l a_l[i] b_l[j], accumulated from 0.0 in the order of l. Each output
/// element sees exactly the scalar sequence `c += a·b`, so the result does not
/// depend on the tiling; the accumulators stay in registers.
#[inline(always)]
fn tile_product(a: &[[f64; TR]], b: &[[f64; TC]]) -> [[f64; TC]; TR] {
    debug_assert_eq!(a.len(), b.len());
    let mut c = [[0.0f64; TC]; TR];
    for (x, y) in a.iter().zip(b) {
        for i in 0..TR {
            for j in 0..TC {
                c[i][j] += x[i] * y[j];
            }
        }
    }
    c
}

/// Packs `m` (row-major, `rows × cols`) into column strips of `W` rows:
/// out[s·cols + l][i] = m[(W·s + i)·cols + l], zero for rows ≥ `rows`.
#[inline(always)]
fn pack_row_tiles<const W: usize>(m: &[f64], rows: usize, cols: usize) -> Vec<[f64; W]> {
    let strips = rows.div_ceil(W);
    let mut out = vec![[0.0; W]; strips * cols];
    for r in 0..rows {
        let (s, i) = (r / W, r % W);
        for (l, &v) in m[r * cols..(r + 1) * cols].iter().enumerate() {
            out[s * cols + l][i] = v;
        }
    }
    out
}

/// Runs `$tiled` compiled for AVX when the CPU has it, else for the baseline target.
/// Only the vector width changes: AVX `vmulpd`/`vaddpd` round exactly like the SSE2
/// and scalar instructions, FMA is not enabled, and the order of operations is fixed
/// by the source, so both paths return the same bits.
macro_rules! dispatch_avx {
    ($avx:ident, $tiled:ident, ($($arg:ident: $ty:ty),*)) => {{
        #[cfg(target_arch = "x86_64")]
        {
            #[target_feature(enable = "avx")]
            fn $avx($($arg: $ty),*) -> Vec<f64> {
                $tiled($($arg),*)
            }
            if std::arch::is_x86_feature_detected!("avx") {
                // SAFETY: the CPU supports AVX (checked just above).
                return unsafe { $avx($($arg),*) };
            }
        }
        $tiled($($arg),*)
    }};
}

/// t_a(g) = Σ_b P_ab χ_b(g) for row-major `p` (`na × na`) and `chi` (`na × np`),
/// with the sum over b taken in order from 0.0 for every (a, g). theory.typ eq:rho
fn density_times_ao(p: &[f64], chi: &[f64], na: usize, np: usize) -> Vec<f64> {
    dispatch_avx!(density_times_ao_avx, density_times_ao_tiled, (p: &[f64], chi: &[f64], na: usize, np: usize))
}

/// M_ab = Σ_g x_a(g) y_b(g) for row-major `x`, `y` (`na × np`); see [`ao_products_tiled`].
fn ao_products(x: &[f64], y: &[f64], na: usize, np: usize) -> Vec<f64> {
    dispatch_avx!(ao_products_avx, ao_products_tiled, (x: &[f64], y: &[f64], na: usize, np: usize))
}

/// [`density_times_ao`] with [`TR`]×[`TC`] register tiles.
///
/// Terms with P_ab = 0 are not skipped: they add ±0 to a partial sum that is
/// never −0 (it starts at +0 and a rounded sum is −0 only if both operands are),
/// which leaves it bit for bit unchanged.
#[inline(always)]
fn density_times_ao_tiled(p: &[f64], chi: &[f64], na: usize, np: usize) -> Vec<f64> {
    let mut t = vec![0.0; na * np];
    if na == 0 || np == 0 {
        return t;
    }
    // a-strip s: [b][i] = P[TR·s + i, b]; g-strip s: [b][j] = χ_b(TC·s + j).
    let pa = pack_row_tiles::<TR>(p, na, na);
    let mut cb = vec![[0.0; TC]; np.div_ceil(TC) * na];
    for b in 0..na {
        for (g, &v) in chi[b * np..(b + 1) * np].iter().enumerate() {
            cb[(g / TC) * na + b][g % TC] = v;
        }
    }
    for (sa, a_strip) in pa.chunks_exact(na).enumerate() {
        for (sg, g_strip) in cb.chunks_exact(na).enumerate() {
            let c = tile_product(a_strip, g_strip);
            for (i, row) in c.iter().enumerate() {
                let a = sa * TR + i;
                if a >= na {
                    break;
                }
                let g0 = sg * TC;
                let n = TC.min(np - g0);
                t[a * np + g0..a * np + g0 + n].copy_from_slice(&row[..n]);
            }
        }
    }
    t
}

/// Packs the first LANES·⌊np/LANES⌋ columns of `m` (row-major, `na × np`) per lane:
/// out[(s·LANES + k)·nj + j][i] = m[W·s + i, LANES·j + k], zero for rows ≥ `na`.
#[inline(always)]
fn pack_lanes<const W: usize>(m: &[f64], na: usize, np: usize) -> Vec<[f64; W]> {
    let nj = np / LANES;
    let mut out = vec![[0.0; W]; na.div_ceil(W) * LANES * nj];
    for r in 0..na {
        let (s, i) = (r / W, r % W);
        for (j, chunk) in m[r * np..r * np + nj * LANES].as_chunks::<LANES>().0.iter().enumerate() {
            for (k, &v) in chunk.iter().enumerate() {
                out[(s * LANES + k) * nj + j][i] = v;
            }
        }
    }
    out
}

/// M_ab = Σ_g x_a(g) y_b(g) for row-major `x`, `y` (`na × np`). The sum keeps
/// [`LANES`] independent partial sums over g ≡ k (mod LANES) for the first
/// LANES·⌊np/LANES⌋ points, adds them in lane order to 0.0 and then adds the
/// remaining points in order. Every lane is a [`tile_product`] over the packed
/// points of that lane.
#[inline(always)]
fn ao_products_tiled(x: &[f64], y: &[f64], na: usize, np: usize) -> Vec<f64> {
    let mut m = vec![0.0; na * na];
    if na == 0 {
        return m;
    }
    let nj = np / LANES;
    let main = nj * LANES;
    let (px, py) = (pack_lanes::<TR>(x, na, np), pack_lanes::<TC>(y, na, np));
    let lane = |s: usize, k: usize| (s * LANES + k) * nj..(s * LANES + k + 1) * nj;
    for sa in 0..na.div_ceil(TR) {
        for sb in 0..na.div_ceil(TC) {
            let mut acc = [[[0.0f64; TC]; TR]; LANES];
            for (k, c) in acc.iter_mut().enumerate() {
                *c = tile_product(&px[lane(sa, k)], &py[lane(sb, k)]);
            }
            // Lane partial sums of every tile element, in lane order.
            let lanes: [[[f64; LANES]; TC]; TR] =
                std::array::from_fn(|i| std::array::from_fn(|j| std::array::from_fn(|k| acc[k][i][j])));
            for (i, lanes_i) in lanes.iter().enumerate() {
                let a = sa * TR + i;
                if a >= na {
                    break;
                }
                for (j, lanes_ij) in lanes_i.iter().enumerate() {
                    let b = sb * TC + j;
                    if b >= na {
                        break;
                    }
                    let mut s: f64 = lanes_ij.iter().sum();
                    for (xv, yv) in x[a * np + main..(a + 1) * np].iter().zip(&y[b * np + main..(b + 1) * np]) {
                        s += xv * yv;
                    }
                    m[a * na + b] = s;
                }
            }
        }
    }
    m
}

/// Which shells/AOs are active on one grid block.
#[derive(Debug, Clone)]
struct BlockSetup {
    shells: Vec<usize>,
    /// Active AO indices, strictly increasing.
    aos: Vec<usize>,
    /// Maximal runs of consecutive AO indices in `aos`: `(position in aos, first AO,
    /// length)`, so that `aos[pos..pos + len] == first..first + len`.
    runs: Vec<(usize, usize, usize)>,
}

impl BlockSetup {
    fn new(shells: Vec<usize>, aos: Vec<usize>) -> Self {
        // Shells are in AO order (Basis numbers them consecutively); the reduction
        // relies on this to find a row with a binary search.
        assert!(aos.windows(2).all(|w| w[0] < w[1]), "active AOs must be increasing");
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        for (pos, &a) in aos.iter().enumerate() {
            match runs.last_mut() {
                Some((_, first, len)) if *first + *len == a => *len += 1,
                _ => runs.push((pos, a, 1)),
            }
        }
        BlockSetup { shells, aos, runs }
    }

    /// The block's sub-matrix P[aos, aos] (row-major `na × na`) of the `n × n`
    /// row-major `p`, copied run by run.
    fn gather(&self, p: &[f64], n: usize) -> Vec<f64> {
        let na = self.aos.len();
        let mut sub = vec![0.0; na * na];
        for (dst, &a) in sub.chunks_exact_mut(na.max(1)).zip(&self.aos) {
            let row = &p[a * n..(a + 1) * n];
            for &(pos, first, len) in &self.runs {
                dst[pos..pos + len].copy_from_slice(&row[first..first + len]);
            }
        }
        sub
    }
}

/// Per-block result of the parallel phase: E_xc and Σ w ρ of the block, its
/// `na × na` potential V[aos, aos] and its phase times.
type BlockPart = (f64, f64, Vec<f64>, XcPhaseTimes);

/// Upper bound (bytes) on the block potentials held at once: blocks are processed in
/// consecutive chunks of at most this size (a chunk has at least one block).
const REDUCTION_CHUNK_BYTES: usize = 256 << 20;

/// Numerical integrator of the XC energy and potential for a fixed basis, functional
/// and grid. Implements [`ExchangeCorrelation`] for [`run_scf`].
#[derive(Debug, Clone)]
pub struct XcIntegrator {
    functional: Functional,
    basis: Basis,
    grid: MolecularGrid,
    deriv: usize,
    blocks: Vec<BlockSetup>,
    cache: Vec<Option<AoValues>>,
    last: Option<(f64, f64)>,
    times: XcPhaseTimes,
}

impl XcIntegrator {
    /// Builds the grid of `mol` and prepares the integration.
    pub fn new(mol: &Molecule, basis: &Basis, functional: &Functional, grid: GridSettings) -> Result<Self> {
        Self::with_grid(basis, functional, MolecularGrid::new(mol, grid)?, XcOptions::default())
    }

    /// Uses an existing grid and explicit options.
    pub fn with_grid(basis: &Basis, functional: &Functional, grid: MolecularGrid, options: XcOptions) -> Result<Self> {
        let deriv = usize::from(functional.is_gga());
        let shells = basis.shells();
        let cutoffs: Vec<f64> = shells.iter().map(|s| shell_cutoff_radius(s, AO_SCREENING_EPS)).collect();
        let blocks: Vec<BlockSetup> = grid
            .blocks
            .iter()
            .map(|b| {
                let mut act = Vec::new();
                let mut aos = Vec::new();
                for (i, s) in shells.iter().enumerate() {
                    let r = ((s.center[0] - b.center[0]).powi(2)
                        + (s.center[1] - b.center[1]).powi(2)
                        + (s.center[2] - b.center[2]).powi(2))
                    .sqrt();
                    if (b.radius - r).abs() < cutoffs[i] {
                        act.push(i);
                        aos.extend(s.ao_offset..s.ao_offset + s.n_functions());
                    }
                }
                BlockSetup::new(act, aos)
            })
            .collect();
        // Cache AO values while the budget lasts.
        let ncomp = crate::ao_eval::N_DERIV_COMPONENTS[deriv];
        let mut used = 0usize;
        let want: Vec<bool> = blocks
            .iter()
            .zip(&grid.blocks)
            .map(|(s, b)| {
                let bytes = ncomp * s.aos.len() * b.range.len() * 8;
                if used + bytes <= options.ao_cache_bytes {
                    used += bytes;
                    true
                } else {
                    false
                }
            })
            .collect();
        let cache_start = Instant::now();
        let cache = if functional.has_density_part() {
            blocks
                .par_iter()
                .zip(grid.blocks.par_iter())
                .zip(want.par_iter())
                .map(|((s, b), &w)| {
                    (w && !s.aos.is_empty()).then(|| eval_ao_shells(basis, &s.shells, &grid.points[b.range.clone()], deriv))
                })
                .collect()
        } else {
            vec![None; blocks.len()]
        };
        let times = XcPhaseTimes { ao_cache_seconds: cache_start.elapsed().as_secs_f64(), ..XcPhaseTimes::default() };
        Ok(XcIntegrator { functional: functional.clone(), basis: basis.clone(), grid, deriv, blocks, cache, last: None, times })
    }

    /// The grid.
    pub fn grid(&self) -> &MolecularGrid {
        &self.grid
    }

    /// The functional.
    pub fn functional(&self) -> &Functional {
        &self.functional
    }

    /// AO indices that are not screened out on grid block `block`.
    pub fn block_aos(&self, block: usize) -> &[usize] {
        &self.blocks[block].aos
    }

    /// Shell indices that are not screened out on grid block `block`.
    pub fn block_shells(&self, block: usize) -> &[usize] {
        &self.blocks[block].shells
    }

    /// Electron count Σ w ρ of the most recent [`ExchangeCorrelation::evaluate`] call.
    pub fn last_electron_count(&self) -> Option<f64> {
        self.last.map(|l| l.1)
    }

    /// Time spent in each phase of the XC integration, summed over the
    /// [`ExchangeCorrelation::evaluate`] calls so far (calls of
    /// [`XcIntegrator::evaluate_density`] alone are not included).
    pub fn phase_times(&self) -> XcPhaseTimes {
        self.times
    }

    /// E_xc, V^xc and Σ w ρ for density matrix `p`. theory.typ eq:rho, eq:exc, eq:vxc
    pub fn evaluate_density(&self, p: &Matrix) -> XcEvaluation {
        self.evaluate_density_timed(p).0
    }

    /// [`XcIntegrator::evaluate_density`] and the time spent in each of its phases.
    fn evaluate_density_timed(&self, p: &Matrix) -> (XcEvaluation, XcPhaseTimes) {
        self.evaluate_density_chunked(p, REDUCTION_CHUNK_BYTES)
    }

    /// [`XcIntegrator::evaluate_density_timed`] holding at most `chunk_bytes` of block
    /// potentials at once (at least one block).
    ///
    /// Every sum is taken in block order, as a serial loop over all blocks would:
    /// E_xc and Σ w ρ block by block, and each element V^xc_ab by adding the block
    /// contributions in block order. The rows of V^xc are independent, so they are
    /// summed in parallel; the result does not depend on `chunk_bytes` or the number
    /// of threads.
    fn evaluate_density_chunked(&self, p: &Matrix, chunk_bytes: usize) -> (XcEvaluation, XcPhaseTimes) {
        let n = self.basis.nao();
        assert_eq!((p.rows(), p.cols()), (n, n), "density matrix shape");
        let mut times = XcPhaseTimes { evaluations: 1, ..XcPhaseTimes::default() };
        let mut energy = 0.0;
        let mut electron_count = 0.0;
        let mut potential = Matrix::zeros(n, n);
        let nb = self.blocks.len();
        let mut first = 0;
        while first < nb {
            let mut end = first;
            let mut bytes = 0usize;
            while end < nb {
                let na = self.blocks[end].aos.len();
                let b = na * na * std::mem::size_of::<f64>();
                if end > first && bytes + b > chunk_bytes {
                    break;
                }
                bytes += b;
                end += 1;
            }
            let parallel_start = Instant::now();
            let parts: Vec<Option<BlockPart>> = (first..end).into_par_iter().map(|ib| self.block_part(ib, p)).collect();
            times.parallel_wall_seconds += parallel_start.elapsed().as_secs_f64();
            let reduction_start = Instant::now();
            for (e, nel, _, bt) in parts.iter().flatten() {
                times.add(bt);
                energy += e;
                electron_count += nel;
            }
            let blocks = &self.blocks[first..end];
            potential.as_mut_slice().par_chunks_mut(n.max(1)).enumerate().for_each(|(a, row)| {
                for (setup, part) in blocks.iter().zip(&parts) {
                    let Some((_, _, vs, _)) = part else { continue };
                    let Ok(ia) = setup.aos.binary_search(&a) else { continue };
                    let na = setup.aos.len();
                    let vrow = &vs[ia * na..(ia + 1) * na];
                    for &(pos, b0, len) in &setup.runs {
                        for (o, v) in row[b0..b0 + len].iter_mut().zip(&vrow[pos..pos + len]) {
                            *o += v;
                        }
                    }
                }
            });
            times.reduction_seconds += reduction_start.elapsed().as_secs_f64();
            first = end;
        }
        (XcEvaluation { energy, potential, electron_count }, times)
    }

    /// E_xc, Σ w ρ and the potential V[aos, aos] of block `ib` (`None` if no AO is
    /// active on it).
    fn block_part(&self, ib: usize, p: &Matrix) -> Option<BlockPart> {
        let n = self.basis.nao();
        let gga = self.deriv >= 1;
        let setup = &self.blocks[ib];
        let na = setup.aos.len();
        if na == 0 {
            return None;
        }
        let mut bt = XcPhaseTimes::default();
        let mut clock = Instant::now();
        let mut lap = || {
            let now = Instant::now();
            let dt = (now - clock).as_secs_f64();
            clock = now;
            dt
        };
        let range = self.grid.blocks[ib].range.clone();
        let computed;
        let ao = match &self.cache[ib] {
            Some(a) => a,
            None => {
                computed = eval_ao_shells(&self.basis, &setup.shells, &self.grid.points[range.clone()], self.deriv);
                bt.recomputed_blocks = 1;
                &computed
            }
        };
        bt.ao_seconds = lap();
        let w = &self.grid.weights[range];
        let np = w.len();
        let psub = setup.gather(p.as_slice(), n);
        bt.gather_seconds = lap();
        let (rho, grad, _) = density_on_points(ao, &psub, gga);
        bt.density_seconds = lap();
        let nel: f64 = rho.iter().zip(w).map(|(r, wg)| r * wg).sum();
        let pd = xc_on_points(&self.functional, rho, grad);
        let exc: f64 = pd.f.iter().zip(w).map(|(f, wg)| f * wg).sum();
        bt.functional_seconds = lap();
        // φ_ν(g) = w [½ v_ρ χ_ν + 2 v_σ ∇ρ·∇χ_ν]; V = M + Mᵀ with M_μν = Σ_g χ_μ φ_ν
        // reproduces theory.typ eq:vxc.
        let chi = ao.component(0);
        let grads = gga.then(|| [ao.component(D_X), ao.component(D_X + 1), ao.component(D_X + 2)]);
        let phi = phi_on_points(chi, grads, w, &pd);
        bt.phi_seconds = lap();
        let v = ao_products(chi, &phi, na, np);
        let mut vs = vec![0.0; na * na];
        for a in 0..na {
            for b in 0..na {
                vs[a * na + b] = v[a * na + b] + v[b * na + a];
            }
        }
        bt.products_seconds = lap();
        Some((exc, nel, vs, bt))
    }
}

/// φ_ν(g) = w_g [½ v_ρ χ_ν + 2 v_σ ∇ρ·∇χ_ν] for every AO row of `chi`
/// (row-major, `chi.len() / w.len()` rows); `grads` holds ∂χ/∂x, ∂χ/∂y,
/// ∂χ/∂z in the same layout for a GGA and is `None` for an LDA
/// (theory.typ eq:vxc).
///
/// The per-point factors c0 = (½w)v_ρ and c_i = ((2w)v_σ)∂_iρ are formed once
/// per block instead of once per AO. They are the same left-to-right
/// products that `0.5 * w * v_rho * χ` and `2.0 * w * v_sigma * grad_rho[i] * d`
/// evaluate, and each φ element is summed left to right as
/// ((c0·χ + c1·∂_xχ) + c2·∂_yχ) + c3·∂_zχ without FMA, the order of the
/// per-AO `ph = …; ph += …` loop, so φ is bit-identical to it.
fn phi_on_points(chi: &[f64], grads: Option<[&[f64]; 3]>, w: &[f64], pd: &PointData) -> Vec<f64> {
    let np = w.len();
    let mut phi = vec![0.0; chi.len()];
    if np == 0 {
        return phi;
    }
    let c0: Vec<f64> = w.iter().zip(&pd.v_rho).map(|(wg, vr)| 0.5 * wg * vr).collect();
    let cg: Option<[Vec<f64>; 3]> = grads.map(|_| {
        std::array::from_fn(|i| (0..np).map(|g| 2.0 * w[g] * pd.v_sigma[g] * pd.grad_rho[g][i]).collect())
    });
    for (a, ph) in phi.chunks_exact_mut(np).enumerate() {
        let row = a * np..(a + 1) * np;
        let x0 = &chi[row.clone()];
        match (&grads, &cg) {
            (Some(d), Some([c1, c2, c3])) => {
                let (x1, x2, x3) = (&d[0][row.clone()], &d[1][row.clone()], &d[2][row]);
                for g in 0..np {
                    ph[g] = c0[g] * x0[g] + c1[g] * x1[g] + c2[g] * x2[g] + c3[g] * x3[g];
                }
            }
            _ => {
                for ((p, c), x) in ph.iter_mut().zip(&c0).zip(x0) {
                    *p = c * x;
                }
            }
        }
    }
    phi
}

impl ExchangeCorrelation for XcIntegrator {
    fn exact_exchange_fraction(&self) -> f64 {
        self.functional.exact_exchange_fraction()
    }

    fn exchange_kernel(&self) -> ExchangeKernel {
        self.functional.exchange_kernel()
    }

    fn evaluate(&mut self, density: &Matrix) -> Result<Option<XcTerm>> {
        if !self.functional.has_density_part() {
            return Ok(None);
        }
        let (ev, times) = self.evaluate_density_timed(density);
        self.times.add(&times);
        self.last = Some((ev.energy, ev.electron_count));
        Ok(Some(XcTerm { energy: ev.energy, potential: ev.potential }))
    }
}

/// Result of an RKS calculation.
#[derive(Debug, Clone)]
pub struct RksResult {
    /// The converged SCF (energy, components, orbitals, density).
    pub scf: ScfResult,
    /// Σ_g w_g ρ_g for the final density; `None` if the functional has no
    /// density-functional part (HF).
    pub electron_count: Option<f64>,
    /// Number of grid points used.
    pub n_grid_points: usize,
}

/// Closed-shell Kohn–Sham single-point energy (theory.typ section 8).
pub fn run_rks(
    mol: &Molecule,
    basis: &Basis,
    functional: &Functional,
    grid: GridSettings,
    options: &ScfOptions,
) -> Result<RksResult> {
    let mut xc = if functional.has_density_part() {
        XcIntegrator::new(mol, basis, functional, grid)?
    } else {
        // No grid needed.
        let empty = MolecularGrid { points: vec![], weights: vec![], blocks: vec![], settings: grid };
        XcIntegrator::with_grid(basis, functional, empty, XcOptions::default())?
    };
    let scf = run_scf(mol, basis, &mut xc, options)?;
    Ok(RksResult { scf, electron_count: xc.last_electron_count(), n_grid_points: xc.grid.len() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Unit;
    use crate::scf::run_rhf;

    fn water() -> Molecule {
        Molecule::from_symbols(
            &["O", "H", "H"],
            &[[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
            Unit::Angstrom,
            0,
        )
        .unwrap()
    }

    #[test]
    fn hf_functional_reproduces_rhf() {
        let mol = water();
        let basis = Basis::from_name(&mol, "sto-3g").unwrap();
        let a = run_rhf(&mol, &basis, &ScfOptions::default()).unwrap();
        let b = run_rks(&mol, &basis, &Functional::hf(), GridSettings::default(), &ScfOptions::default()).unwrap();
        assert!((a.energy - b.scf.energy).abs() < 1e-10);
        assert!(b.electron_count.is_none());
    }

    #[test]
    fn potential_is_derivative_of_energy() {
        // dE_xc/dP_μν (symmetric perturbation) = V_μν: check eq:vxc against eq:exc.
        let mol = water();
        let basis = Basis::from_name(&mol, "6-31g*").unwrap();
        let rhf = run_rhf(&mol, &basis, &ScfOptions::default()).unwrap();
        let p = rhf.density;
        for fun in [Functional::svwn5(), Functional::blyp()] {
            let xc = XcIntegrator::new(&mol, &basis, &fun, GridSettings::default()).unwrap();
            let ev = xc.evaluate_density(&p);
            assert!((ev.electron_count - 10.0).abs() < 1e-4, "N = {}", ev.electron_count);
            let h = 1e-5;
            for &(m, n) in &[(0, 0), (1, 3), (2, 9), (5, 14), (13, 17)] {
                let mut pp = p.clone();
                let mut pm = p.clone();
                for (q, s) in [(&mut pp, 1.0), (&mut pm, -1.0)] {
                    q[(m, n)] += s * h;
                    if m != n {
                        q[(n, m)] += s * h;
                    }
                }
                let fd = (xc.evaluate_density(&pp).energy - xc.evaluate_density(&pm).energy) / (2.0 * h);
                let an = if m == n { ev.potential[(m, n)] } else { 2.0 * ev.potential[(m, n)] };
                assert!((fd - an).abs() < 1e-6 * an.abs().max(1.0), "{} ({m},{n}): {an} vs {fd}", fun.name());
            }
        }
    }

    #[test]
    fn phi_matches_per_ao_expression_bitwise() {
        // phi_on_points must give exactly the bits of the per-AO expression
        // φ = 0.5·w·v_ρ·χ (+ 2·w·v_σ·∂_iρ·∂_iχ for i = x, y, z).
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut rnd = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let (na, np) = (7, 37);
        let w: Vec<f64> = (0..np).map(|_| rnd().abs() * 3.7).collect();
        let chi: Vec<f64> = (0..na * np).map(|_| rnd()).collect();
        let d: Vec<Vec<f64>> = (0..3).map(|_| (0..na * np).map(|_| rnd() * 1.3).collect()).collect();
        let pd = PointData {
            rho: vec![],
            grad_rho: (0..np).map(|_| [rnd(), rnd(), rnd()]).collect(),
            f: vec![],
            v_rho: (0..np).map(|_| rnd() * 0.9).collect(),
            v_sigma: (0..np).map(|_| rnd() * 0.01).collect(),
        };
        for gga in [false, true] {
            let mut want = vec![0.0; na * np];
            for a in 0..na {
                let ph = &mut want[a * np..(a + 1) * np];
                let ca = &chi[a * np..(a + 1) * np];
                for g in 0..np {
                    ph[g] = 0.5 * w[g] * pd.v_rho[g] * ca[g];
                }
                if gga {
                    for (i, di) in d.iter().enumerate() {
                        let di = &di[a * np..(a + 1) * np];
                        for g in 0..np {
                            ph[g] += 2.0 * w[g] * pd.v_sigma[g] * pd.grad_rho[g][i] * di[g];
                        }
                    }
                }
            }
            let grads = gga.then(|| [d[0].as_slice(), d[1].as_slice(), d[2].as_slice()]);
            let got = phi_on_points(&chi, grads, &w, &pd);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "gga = {gga}");
        }
    }

    #[test]
    fn cache_and_recompute_agree() {
        let mol = water();
        let basis = Basis::from_name(&mol, "6-31g").unwrap();
        let p = run_rhf(&mol, &basis, &ScfOptions::default()).unwrap().density;
        let fun = Functional::b3lyp();
        let grid = MolecularGrid::new(&mol, GridSettings::default()).unwrap();
        let a = XcIntegrator::with_grid(&basis, &fun, grid.clone(), XcOptions::default()).unwrap();
        let b = XcIntegrator::with_grid(&basis, &fun, grid, XcOptions { ao_cache_bytes: 0 }).unwrap();
        let (ea, eb) = (a.evaluate_density(&p), b.evaluate_density(&p));
        assert_eq!(ea.energy, eb.energy);
        assert_eq!(ea.potential, eb.potential);
    }

    #[test]
    fn phase_times_accumulate_over_evaluations() {
        let mol = water();
        let basis = Basis::from_name(&mol, "6-31g").unwrap();
        let p = run_rhf(&mol, &basis, &ScfOptions::default()).unwrap().density;
        let fun = Functional::b3lyp();
        let grid = MolecularGrid::new(&mol, GridSettings::default()).unwrap();
        let mut xc = XcIntegrator::with_grid(&basis, &fun, grid, XcOptions { ao_cache_bytes: 0 }).unwrap();
        assert_eq!(xc.phase_times().evaluations, 0);
        let direct = xc.evaluate_density(&p);
        assert_eq!(xc.phase_times().evaluations, 0, "evaluate_density alone is not counted");
        let term = xc.evaluate(&p).unwrap().unwrap();
        xc.evaluate(&p).unwrap();
        // The timed path gives bit-identical results.
        assert_eq!(term.energy, direct.energy);
        assert_eq!(term.potential, direct.potential);
        let t = xc.phase_times();
        assert_eq!(t.evaluations, 2);
        let active = (0..xc.grid().blocks.len()).filter(|&b| !xc.block_aos(b).is_empty()).count();
        assert_eq!(t.recomputed_blocks, 2 * active, "nothing is cached with a zero budget");
        let phases = [t.ao_seconds, t.gather_seconds, t.density_seconds, t.functional_seconds, t.phi_seconds];
        assert!(phases.iter().chain([&t.products_seconds, &t.parallel_wall_seconds, &t.reduction_seconds]).all(|&s| s >= 0.0));
        assert!(t.ao_seconds > 0.0 && t.parallel_wall_seconds > 0.0);
    }

    #[test]
    fn chunked_row_parallel_sum_matches_serial_block_order_bitwise() {
        let mol = water();
        let basis = Basis::from_name(&mol, "6-31g*").unwrap();
        let p = run_rhf(&mol, &basis, &ScfOptions::default()).unwrap().density;
        let n = basis.nao();
        let xc = XcIntegrator::new(&mol, &basis, &Functional::b3lyp(), GridSettings::default()).unwrap();
        let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        // Run-wise gather = element-wise gather.
        let pr = &p;
        for setup in &xc.blocks {
            let elementwise: Vec<f64> = setup.aos.iter().flat_map(|&a| setup.aos.iter().map(move |&b| pr[(a, b)])).collect();
            assert_eq!(bits(&setup.gather(p.as_slice(), n)), bits(&elementwise));
        }
        assert!(xc.blocks.iter().any(|s| s.runs.len() > 1), "no block with a gap in its AOs");
        // Previous form: all block parts, then a serial sum in block order.
        let (mut e_ref, mut n_ref, mut v_ref) = (0.0, 0.0, Matrix::zeros(n, n));
        for ib in 0..xc.blocks.len() {
            let Some((e, nel, vs, _)) = xc.block_part(ib, &p) else { continue };
            e_ref += e;
            n_ref += nel;
            let aos = &xc.blocks[ib].aos;
            for (ia, &a) in aos.iter().enumerate() {
                for (jb, &b) in aos.iter().enumerate() {
                    v_ref[(a, b)] += vs[ia * aos.len() + jb];
                }
            }
        }
        let check = |ev: &XcEvaluation, label: &str| {
            assert_eq!(ev.energy.to_bits(), e_ref.to_bits(), "E_xc, {label}");
            assert_eq!(ev.electron_count.to_bits(), n_ref.to_bits(), "N, {label}");
            assert_eq!(bits(ev.potential.as_slice()), bits(v_ref.as_slice()), "V, {label}");
        };
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            for chunk in [0, 50_000, REDUCTION_CHUNK_BYTES] {
                let (ev, _) = pool.install(|| xc.evaluate_density_chunked(&p, chunk));
                check(&ev, &format!("{threads} threads, chunk {chunk} B"));
            }
        }
        check(&xc.evaluate_density(&p), "default pool");
    }

    /// Previous row-axpy form of t (skips P_ab = 0).
    fn reference_t(p: &[f64], chi: &[f64], na: usize, np: usize) -> Vec<f64> {
        let mut t = vec![0.0; na * np];
        for a in 0..na {
            let ta = &mut t[a * np..(a + 1) * np];
            for b in 0..na {
                let pab = p[a * na + b];
                if pab == 0.0 {
                    continue;
                }
                for (o, c) in ta.iter_mut().zip(&chi[b * np..(b + 1) * np]) {
                    *o += pab * c;
                }
            }
        }
        t
    }

    /// Previous per-(a, b) 8-lane dot form of M.
    fn reference_m(x: &[f64], y: &[f64], na: usize, np: usize) -> Vec<f64> {
        let dot = |a: &[f64], b: &[f64]| {
            let mut acc = [0.0f64; 8];
            let (ca, ra) = a.as_chunks::<8>();
            let (cb, rb) = b.as_chunks::<8>();
            for (u, v) in ca.iter().zip(cb) {
                for k in 0..8 {
                    acc[k] += u[k] * v[k];
                }
            }
            let mut s: f64 = acc.iter().sum();
            for (u, v) in ra.iter().zip(rb) {
                s += u * v;
            }
            s
        };
        let mut m = vec![0.0; na * na];
        for a in 0..na {
            for b in 0..na {
                m[a * na + b] = dot(&x[a * np..(a + 1) * np], &y[b * np..(b + 1) * np]);
            }
        }
        m
    }

    #[test]
    fn tiled_kernels_match_scalar_loops_bitwise() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Mixed signs and magnitudes, with some exact zeros.
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            if u < 0.1 { 0.0 } else { (2.0 * u - 1.0) * 10f64.powi((state % 9) as i32 - 4) }
        };
        for &na in &[1, 3, 4, 5, 8, 13, 22] {
            for &np in &[0, 1, 3, 7, 8, 9, 16, 37, 101] {
                let p: Vec<f64> = (0..na * na).map(|_| next()).collect();
                let x: Vec<f64> = (0..na * np).map(|_| next()).collect();
                let y: Vec<f64> = (0..na * np).map(|_| next()).collect();
                let (t, t_ref) = (density_times_ao(&p, &x, na, np), reference_t(&p, &x, na, np));
                let (m, m_ref) = (ao_products(&x, &y, na, np), reference_m(&x, &y, na, np));
                let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&t), bits(&t_ref), "t: na = {na}, np = {np}");
                assert_eq!(bits(&m), bits(&m_ref), "M: na = {na}, np = {np}");
                // The baseline-target build of the same kernels (no AVX dispatch).
                assert_eq!(bits(&density_times_ao_tiled(&p, &x, na, np)), bits(&t_ref), "t: na = {na}, np = {np}");
                assert_eq!(bits(&ao_products_tiled(&x, &y, na, np)), bits(&m_ref), "M: na = {na}, np = {np}");
            }
        }
    }

    #[test]
    fn screening_has_no_measurable_effect() {
        // Compare against an unscreened evaluation (every AO on every point).
        let mol = water();
        let basis = Basis::from_name(&mol, "6-31g").unwrap();
        let p = run_rhf(&mol, &basis, &ScfOptions::default()).unwrap().density;
        let fun = Functional::blyp();
        let xc = XcIntegrator::new(&mol, &basis, &fun, GridSettings::default()).unwrap();
        let ev = xc.evaluate_density(&p);
        let grid = xc.grid();
        let all: Vec<usize> = (0..basis.shells().len()).collect();
        let ao = eval_ao_shells(&basis, &all, &grid.points, 1);
        let (rho, grad, _) = density_on_points(&ao, p.as_slice(), true);
        let pd = xc_on_points(&fun, rho, grad);
        let e_full: f64 = pd.f.iter().zip(&grid.weights).map(|(f, w)| f * w).sum();
        assert!((e_full - ev.energy).abs() < 1e-12, "{e_full} vs {}", ev.energy);
        let screened: usize = (0..grid.blocks.len()).map(|b| xc.block_aos(b).len() * grid.blocks[b].range.len()).sum();
        assert!(screened < basis.nao() * grid.len(), "screening removed nothing");
    }
}

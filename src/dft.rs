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

/// Σ_g a_g · b_g with independent partial sums (lets the compiler vectorize).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f64; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s: f64 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Which shells/AOs are active on one grid block.
#[derive(Debug, Clone)]
struct BlockSetup {
    shells: Vec<usize>,
    aos: Vec<usize>,
}

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
                BlockSetup { shells: act, aos }
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
        Ok(XcIntegrator { functional: functional.clone(), basis: basis.clone(), grid, deriv, blocks, cache, last: None })
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

    /// E_xc, V^xc and Σ w ρ for density matrix `p`. theory.typ eq:rho, eq:exc, eq:vxc
    pub fn evaluate_density(&self, p: &Matrix) -> XcEvaluation {
        let n = self.basis.nao();
        assert_eq!((p.rows(), p.cols()), (n, n), "density matrix shape");
        let gga = self.deriv >= 1;
        let parts: Vec<Option<(f64, f64, Vec<f64>)>> = (0..self.blocks.len())
            .into_par_iter()
            .map(|ib| {
                let setup = &self.blocks[ib];
                let na = setup.aos.len();
                if na == 0 {
                    return None;
                }
                let range = self.grid.blocks[ib].range.clone();
                let computed;
                let ao = match &self.cache[ib] {
                    Some(a) => a,
                    None => {
                        computed = eval_ao_shells(&self.basis, &setup.shells, &self.grid.points[range.clone()], self.deriv);
                        &computed
                    }
                };
                let w = &self.grid.weights[range];
                let np = w.len();
                let psub: Vec<f64> =
                    setup.aos.iter().flat_map(|&a| setup.aos.iter().map(move |&b| p[(a, b)])).collect();
                let (rho, grad, _) = density_on_points(ao, &psub, gga);
                let nel: f64 = rho.iter().zip(w).map(|(r, wg)| r * wg).sum();
                let pd = xc_on_points(&self.functional, rho, grad);
                let exc: f64 = pd.f.iter().zip(w).map(|(f, wg)| f * wg).sum();
                // φ_ν(g) = w [½ v_ρ χ_ν + 2 v_σ ∇ρ·∇χ_ν]; V = M + Mᵀ with M_μν = Σ_g χ_μ φ_ν
                // reproduces theory.typ eq:vxc.
                let chi = ao.component(0);
                let mut phi = vec![0.0; na * np];
                for a in 0..na {
                    let ph = &mut phi[a * np..(a + 1) * np];
                    let ca = &chi[a * np..(a + 1) * np];
                    for g in 0..np {
                        ph[g] = 0.5 * w[g] * pd.v_rho[g] * ca[g];
                    }
                    if gga {
                        for i in 0..3 {
                            let d = &ao.component(D_X + i)[a * np..(a + 1) * np];
                            for g in 0..np {
                                ph[g] += 2.0 * w[g] * pd.v_sigma[g] * pd.grad_rho[g][i] * d[g];
                            }
                        }
                    }
                }
                let mut v = vec![0.0; na * na];
                for a in 0..na {
                    for b in 0..na {
                        v[a * na + b] = dot(&chi[a * np..(a + 1) * np], &phi[b * np..(b + 1) * np]);
                    }
                }
                let mut vs = vec![0.0; na * na];
                for a in 0..na {
                    for b in 0..na {
                        vs[a * na + b] = v[a * na + b] + v[b * na + a];
                    }
                }
                Some((exc, nel, vs))
            })
            .collect();
        // Deterministic reduction in block order.
        let mut energy = 0.0;
        let mut electron_count = 0.0;
        let mut potential = Matrix::zeros(n, n);
        for (ib, part) in parts.into_iter().enumerate() {
            let Some((e, nel, vs)) = part else { continue };
            energy += e;
            electron_count += nel;
            let aos = &self.blocks[ib].aos;
            let na = aos.len();
            for (ia, &a) in aos.iter().enumerate() {
                for (jb, &b) in aos.iter().enumerate() {
                    potential[(a, b)] += vs[ia * na + jb];
                }
            }
        }
        XcEvaluation { energy, potential, electron_count }
    }
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
        let ev = self.evaluate_density(density);
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

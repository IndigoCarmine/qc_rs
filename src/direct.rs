//! Integral-direct Coulomb and exchange matrices (theory.typ sec:direct,
//! eq:jk-quartet, eq:schwarz-dens, eq:fock-incr).
//!
//! For a given density matrix the unique shell quartets (8-fold symmetry,
//! eq:eri-sym) are computed on the fly with the McMurchie–Davidson form of eq:eri
//! (and eq:eri-erf for a range-separated exchange kernel) and contracted at once into
//! J and the exchange matrix (eq:jk-quartet); no integral is stored.
//!
//! * Everything is done over the normalized *Cartesian* components: the density is
//!   transformed once to P_cart = Tᵀ P T and the results back with J = T J_cart Tᵀ
//!   (T the eq:sph-d / eq:sph-f matrix, the identity for Cartesian shells). This equals the UNCHECKED(Schlegel1995)
//!   contraction of the spherical integrals and avoids transforming every quartet.
//! * The Hermite expansion coefficients E of every shell pair (eq:hermite-exp) are
//!   computed once per basis and kept in sparse form; a quartet only evaluates the
//!   Hermite Coulomb integrals R (eq:R-rec). For a range-separated kernel the Coulomb
//!   and the attenuated R of the same primitive quartet share these E. R uses the
//!   tabulated Boys function (eq:boys-taylor). For each bra primitive pair the ket
//!   primitive pairs are summed in Hermite space before the bra coefficients are
//!   applied.
//! * A quartet is skipped when its density-weighted Schwarz bound (eq:schwarz-dens)
//!   is below the threshold. The Coulomb factors Q bound the attenuated integrals as
//!   well ((μν|μν)_ω ≤ (μν|μν), erf(ωr)/r and erfc(ωr)/r being positive-definite
//!   kernels), so one test covers both operators. Inside a quartet, primitive
//!   quartets are screened the same way with primitive Schwarz factors at τ/10.
//! * Work is split into fixed chunks of bra shell pairs; each chunk accumulates into
//!   its own matrices and the chunks are added in a fixed order, so the result does
//!   not depend on the number of threads or on scheduling.

use crate::basis::Basis;
use crate::boys::{TABULATED_NMAX, boys_tabulated};
use crate::hermite::erf_attenuation;
use crate::integrals::{ExchangeKernel, ShellPairData, cartesian_schwarz_factors, pair_index};
use crate::linalg::Matrix;
use rayon::prelude::*;
use std::ops::Range;
use std::time::Instant;

/// Number of work chunks a J/K build is split into (fixed, so that the order of the
/// floating-point additions, and with it the result, does not depend on the number
/// of threads).
const N_CHUNKS: usize = 384;

/// 2 π^{5/2}, the prefactor of theory.typ eq:eri.
const TWO_PI_POW_2_5: f64 = 34.986_836_655_249_725;

/// Primitive quartets are skipped below this fraction of the shell-quartet threshold
/// (their bound Q_a Q_c max|P| uses the primitive Schwarz factors).
const PRIM_FACTOR: f64 = 0.1;

/// Statistics of one direct J/K build.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DirectBuildStatistics {
    /// Whether the build used the density difference ΔP (theory.typ eq:fock-incr).
    pub incremental: bool,
    /// Number of unique shell quartets of the basis.
    pub quartets_total: u64,
    /// Number of shell quartets that passed the screening and were computed.
    pub quartets_computed: u64,
    /// Wall time of the build (seconds).
    pub seconds: f64,
}

impl DirectBuildStatistics {
    /// Fraction of the unique shell quartets skipped by the screening.
    pub fn skipped_fraction(&self) -> f64 {
        if self.quartets_total == 0 { 0.0 } else { 1.0 - self.quartets_computed as f64 / self.quartets_total as f64 }
    }
}

/// One primitive pair of a shell pair: exponent, centre, and the nonzero Hermite
/// coefficients in the order of the pair's sparsity pattern.
#[derive(Clone)]
struct DirectPrim {
    p: f64,
    center: [f64; 3],
    val: Vec<f64>,
    /// Schwarz factor of the primitive pair, sqrt(max |(ab|ab)|) over its component
    /// pairs (theory.typ eq:schwarz applied to the primitive product).
    q: f64,
}

/// A shell pair (a >= b) prepared for direct J/K builds. The Hermite coefficients
/// E^{ab}_{tuv} (times d̃_a d̃_b c_a c_b) are stored sparsely, by Cartesian component
/// pair: entries `off[c]..off[c + 1]` of `idx` (index into `tuv`) and of each
/// primitive's `val` belong to component pair `c`.
struct DirectPair {
    sa: usize,
    sb: usize,
    lab: usize,
    ncomp: usize,
    tuv: Vec<[usize; 3]>,
    /// (-1)^{t+u+v} per entry of `tuv` (for the ket side of eq:eri).
    sign: Vec<f64>,
    off: Vec<usize>,
    idx: Vec<usize>,
    prims: Vec<DirectPrim>,
}

impl DirectPair {
    fn new(basis: &Basis, a: usize, b: usize) -> Self {
        let sp = ShellPairData::new(basis.shells(), a, b);
        let (ncomp, ntuv) = (sp.ncomp, sp.tuv.len());
        // Union of the nonzero entries over the primitives.
        let mut off = Vec::with_capacity(ncomp + 1);
        let mut idx = Vec::new();
        off.push(0);
        for c in 0..ncomp {
            for it in 0..ntuv {
                if sp.prims.iter().any(|pp| pp.e[it * ncomp + c] != 0.0) {
                    idx.push(it);
                }
            }
            off.push(idx.len());
        }
        let mut prims = Vec::with_capacity(sp.prims.len());
        for pp in &sp.prims {
            let mut val = Vec::with_capacity(idx.len());
            for c in 0..ncomp {
                for &it in &idx[off[c]..off[c + 1]] {
                    val.push(pp.e[it * ncomp + c]);
                }
            }
            // A primitive pair whose coefficients all underflowed contributes nothing.
            if val.iter().any(|&v| v != 0.0) {
                prims.push(DirectPrim { p: pp.p, center: pp.center, val, q: 0.0 });
            }
        }
        let sign = sp.tuv.iter().map(|h| if (h[0] + h[1] + h[2]) % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut pair = DirectPair { sa: sp.sa, sb: sp.sb, lab: sp.lab, ncomp, tuv: sp.tuv, sign, off, idx, prims };
        // Primitive Schwarz factors; primitives in decreasing order of them, so that the
        // primitive loops of `quartet` can stop at the first negligible one.
        let mut s = Scratch::default();
        let all = std::mem::take(&mut pair.prims);
        let mut prims = Vec::with_capacity(all.len());
        for pp in all {
            pair.prims = vec![pp];
            quartet(&pair, &pair, None, 0.0, &mut s);
            let m = (0..ncomp).map(|c| s.v[c * ncomp + c].abs()).fold(0.0, f64::max);
            let mut pp = pair.prims.pop().expect("one primitive");
            pp.q = m.sqrt();
            prims.push(pp);
        }
        prims.sort_by(|x, y| y.q.total_cmp(&x.q));
        pair.prims = prims;
        pair
    }

    /// Rough operation count of a quartet with `cd` (for balancing the chunks).
    fn cost(&self, cd: &DirectPair) -> f64 {
        let (nti, ntj) = (self.tuv.len() as f64, cd.tuv.len() as f64);
        let per_prim = nti * ntj + cd.idx.len() as f64 * nti + self.idx.len() as f64 * cd.ncomp as f64 + 30.0;
        (self.prims.len() * cd.prims.len()) as f64 * per_prim + (self.ncomp * cd.ncomp) as f64 * 6.0 + 50.0
    }
}

/// Hermite Coulomb integrals R_tuv(α, R) of theory.typ eq:R-rec with the tabulated
/// Boys function (eq:boys-taylor), in the layout of [`crate::hermite::HermiteR`]:
/// R_tuv at `(t * dim + u) * dim + v`, dim = lmax + 1.
#[derive(Default)]
struct HermiteRFast {
    data: Vec<f64>,
    scratch: Vec<f64>,
    f: [f64; TABULATED_NMAX + 1],
}

impl HermiteRFast {
    fn compute(&mut self, lmax: usize, alpha: f64, r: [f64; 3]) {
        let dim = lmax + 1;
        let size = dim * dim * dim;
        if self.data.len() != size {
            self.data.clear();
            self.data.resize(size, 0.0);
            self.scratch.clear();
            self.scratch.resize(size, 0.0);
        }
        let x2 = alpha * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        boys_tabulated(lmax, x2, &mut self.f);
        if lmax == 0 {
            self.data[0] = self.f[0];
            return;
        }
        // (-2α)^n F_n
        let m2a = -2.0 * alpha;
        let mut pw = 1.0;
        for n in 0..=lmax {
            self.f[n] *= pw;
            pw *= m2a;
        }
        let idx = |t: usize, u: usize, v: usize| (t * dim + u) * dim + v;
        self.data[0] = self.f[lmax];
        // Layer n from layer n + 1 (eq:R-rec); only degrees <= lmax - n are written,
        // and only degrees below that are read.
        for n in (0..lmax).rev() {
            let m = lmax - n;
            let cur = &self.data;
            let nx = &mut self.scratch;
            nx[0] = self.f[n];
            for t in 1..=m {
                for u in 0..=m - t {
                    for v in 0..=m - t - u {
                        let mut val = r[0] * cur[idx(t - 1, u, v)];
                        if t > 1 {
                            val += (t - 1) as f64 * cur[idx(t - 2, u, v)];
                        }
                        nx[idx(t, u, v)] = val;
                    }
                }
            }
            for u in 1..=m {
                for v in 0..=m - u {
                    let mut val = r[1] * cur[idx(0, u - 1, v)];
                    if u > 1 {
                        val += (u - 1) as f64 * cur[idx(0, u - 2, v)];
                    }
                    nx[idx(0, u, v)] = val;
                }
            }
            for v in 1..=m {
                let mut val = r[2] * cur[idx(0, 0, v - 1)];
                if v > 1 {
                    val += (v - 1) as f64 * cur[idx(0, 0, v - 2)];
                }
                nx[idx(0, 0, v)] = val;
            }
            std::mem::swap(&mut self.data, &mut self.scratch);
        }
    }
}

/// Largest l_a + l_b of a shell pair served by the unrolled kernel [`quartet_small`]
/// (s and p shells), so that its R tables have ltot <= 4.
const SMALL_LAB: usize = 2;
/// Size of an R table of [`hermite_r_small`]: (2 SMALL_LAB + 1)³.
const SMALL_CUBE: usize = (2 * SMALL_LAB + 1) * (2 * SMALL_LAB + 1) * (2 * SMALL_LAB + 1);

/// R tables of [`hermite_r_small`] (two layers, alternately read and written).
struct SmallR {
    a: [f64; SMALL_CUBE],
    b: [f64; SMALL_CUBE],
    f: [f64; TABULATED_NMAX + 1],
}

impl Default for SmallR {
    fn default() -> Self {
        SmallR { a: [0.0; SMALL_CUBE], b: [0.0; SMALL_CUBE], f: [0.0; TABULATED_NMAX + 1] }
    }
}

/// [`HermiteRFast::compute`] for `lmax <= 2 SMALL_LAB` into fixed-size buffers: the
/// same recurrence (eq:R-rec) with the same floating-point operations, so the values
/// are identical. Inlined into [`quartet_small`], where `lmax` is a constant and the
/// loops are unrolled. Returns the table in the layout of [`HermiteRFast`].
#[inline(always)]
fn hermite_r_small(lmax: usize, alpha: f64, r: [f64; 3], s: &mut SmallR) -> &[f64; SMALL_CUBE] {
    debug_assert!(lmax <= 2 * SMALL_LAB);
    let dim = lmax + 1;
    let x2 = alpha * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    let f = &mut s.f;
    boys_tabulated(lmax, x2, f);
    let (mut cur, mut nx) = (&mut s.a, &mut s.b);
    if lmax == 0 {
        cur[0] = f[0];
        return cur;
    }
    // (-2α)^n F_n
    let m2a = -2.0 * alpha;
    let mut pw = 1.0;
    for fv in f.iter_mut().take(lmax + 1) {
        *fv *= pw;
        pw *= m2a;
    }
    let idx = |t: usize, u: usize, v: usize| (t * dim + u) * dim + v;
    cur[0] = f[lmax];
    for n in (0..lmax).rev() {
        let m = lmax - n;
        nx[0] = f[n];
        for t in 1..=m {
            for u in 0..=m - t {
                for v in 0..=m - t - u {
                    let mut val = r[0] * cur[idx(t - 1, u, v)];
                    if t > 1 {
                        val += (t - 1) as f64 * cur[idx(t - 2, u, v)];
                    }
                    nx[idx(t, u, v)] = val;
                }
            }
        }
        for u in 1..=m {
            for v in 0..=m - u {
                let mut val = r[1] * cur[idx(0, u - 1, v)];
                if u > 1 {
                    val += (u - 1) as f64 * cur[idx(0, u - 2, v)];
                }
                nx[idx(0, u, v)] = val;
            }
        }
        for v in 1..=m {
            let mut val = r[2] * cur[idx(0, 0, v - 1)];
            if v > 1 {
                val += (v - 1) as f64 * cur[idx(0, 0, v - 2)];
            }
            nx[idx(0, 0, v)] = val;
        }
        std::mem::swap(&mut cur, &mut nx);
    }
    cur
}

/// Number of Hermite indices t + u + v <= l.
const fn n_tuv(l: usize) -> usize {
    (l + 1) * (l + 2) * (l + 3) / 6
}

/// Hermite indices with t + u + v <= l in the order of `ShellPairData::tuv`.
const fn tuv_list<const N: usize>(l: usize) -> [[usize; 3]; N] {
    assert!(N == n_tuv(l));
    let mut out = [[0; 3]; N];
    let mut k = 0;
    let mut t = 0;
    while t <= l {
        let mut u = 0;
        while u <= l - t {
            let mut v = 0;
            while v <= l - t - u {
                out[k] = [t, u, v];
                k += 1;
                v += 1;
            }
            u += 1;
        }
        t += 1;
    }
    out
}

/// Index into the R table (dim = lab + lcd + 1) of every (ket tuv, bra tuv),
/// ket-major; the table `Scratch::ridx` of the general kernel.
const fn ridx_table<const NTI: usize, const NTJ: usize>(lab: usize, lcd: usize) -> [[usize; NTI]; NTJ] {
    let (hs, gs) = (tuv_list::<NTI>(lab), tuv_list::<NTJ>(lcd));
    let dim = lab + lcd + 1;
    let mut out = [[0; NTI]; NTJ];
    let mut jt = 0;
    while jt < NTJ {
        let mut it = 0;
        while it < NTI {
            let (h, g) = (hs[it], gs[jt]);
            out[jt][it] = ((h[0] + g[0]) * dim + h[1] + g[1]) * dim + h[2] + g[2];
            it += 1;
        }
        jt += 1;
    }
    out
}

/// (-1)^{τ+ν+φ} per ket Hermite index (`DirectPair::sign`).
const fn sign_table<const NTJ: usize>(lcd: usize) -> [f64; NTJ] {
    let gs = tuv_list::<NTJ>(lcd);
    let mut out = [0.0; NTJ];
    let mut jt = 0;
    while jt < NTJ {
        out[jt] = if (gs[jt][0] + gs[jt][1] + gs[jt][2]).is_multiple_of(2) { 1.0 } else { -1.0 };
        jt += 1;
    }
    out
}

/// [`accumulate_ket`] with rows of length `NTI`.
#[inline(always)]
fn accumulate_ket_small<const NTI: usize, const NTJ: usize>(
    cd: &DirectPair,
    ecd: &[f64],
    rm: &[[f64; NTI]; NTJ],
    x: &mut [f64],
) {
    for (c, xr) in x.as_chunks_mut::<NTI>().0.iter_mut().enumerate() {
        for k in cd.off[c]..cd.off[c + 1] {
            let e = ecd[k];
            for (xv, &r) in xr.iter_mut().zip(&rm[cd.idx[k]]) {
                *xv += e * r;
            }
        }
    }
}

/// [`quartet`] for l_ab = `LAB`, l_cd = `LCD` <= SMALL_LAB (`NTI`, `NTJ` their numbers
/// of Hermite indices), without (`ERF` false) or with (`ERF` true, `erf` Some) a
/// range-separated kernel: the same sums in the same order as [`quartet_general`],
/// with the R recurrences unrolled for ltot = LAB + LCD, the R index and sign tables
/// built at compile time, and fixed-length Hermite rows.
fn quartet_small<const LAB: usize, const LCD: usize, const NTI: usize, const NTJ: usize, const ERF: bool>(
    ab: &DirectPair,
    cd: &DirectPair,
    erf: Option<ErfKernel>,
    prim_cut: f64,
    s: &mut Scratch,
) {
    let ridx: [[usize; NTI]; NTJ] = const { ridx_table::<NTI, NTJ>(LAB, LCD) };
    let sign: [f64; NTJ] = const { sign_table::<NTJ>(LCD) };
    debug_assert!(ab.lab == LAB && cd.lab == LCD);
    debug_assert_eq!(ERF, erf.is_some());
    // A compile-time None without the kernel, so that its code is dropped.
    let erf = if ERF { erf } else { None };
    let n = ab.ncomp * cd.ncomp;
    s.x.resize(NTI * cd.ncomp, 0.0);
    s.v.clear();
    s.v.resize(n, 0.0);
    if erf.is_some() {
        s.x_x.resize(NTI * cd.ncomp, 0.0);
        s.v_x.clear();
        s.v_x.resize(n, 0.0);
    }
    let mut rm = [[0.0; NTI]; NTJ];
    let mut rm_x = [[0.0; NTI]; NTJ];
    let qc_max = cd.prims.first().map_or(0.0, |x| x.q);
    for pa in &ab.prims {
        if pa.q * qc_max < prim_cut {
            break;
        }
        s.x.fill(0.0);
        if erf.is_some() {
            s.x_x.fill(0.0);
        }
        for pc in &cd.prims {
            if pa.q * pc.q < prim_cut {
                break;
            }
            let (p, q) = (pa.p, pc.p);
            let alpha = p * q / (p + q);
            let pq = [0, 1, 2].map(|k| pa.center[k] - pc.center[k]);
            let pref = TWO_PI_POW_2_5 / (p * q * (p + q).sqrt());
            let rc = hermite_r_small(LAB + LCD, alpha, pq, &mut s.small_r);
            for ((row, ix), &sg) in rm.iter_mut().zip(&ridx).zip(&sign) {
                let f = sg * pref;
                for (d, &i) in row.iter_mut().zip(ix) {
                    *d = f * rc[i];
                }
            }
            accumulate_ket_small(cd, &pc.val, &rm, &mut s.x);
            if let Some((omega, ka, kb)) = erf {
                // theory.typ eq:eri-erf: exponent α' and factor √(ω²/(α+ω²)).
                let (a_eff, scale) = erf_attenuation(alpha, omega);
                let re = hermite_r_small(LAB + LCD, a_eff, pq, &mut s.small_r_erf);
                let (fc, fe) = (ka * pref, kb * pref * scale);
                for ((row, ix), &sg) in rm_x.iter_mut().zip(&ridx).zip(&sign) {
                    for (d, &i) in row.iter_mut().zip(ix) {
                        *d = sg * (fc * rc[i] + fe * re[i]);
                    }
                }
                accumulate_ket_small(cd, &pc.val, &rm_x, &mut s.x_x);
            }
        }
        contract_bra(ab, cd.ncomp, &pa.val, &s.x, &mut s.v);
        if erf.is_some() {
            contract_bra(ab, cd.ncomp, &pa.val, &s.x_x, &mut s.v_x);
        }
    }
}

/// Per-thread scratch buffers.
#[derive(Default)]
struct Scratch {
    r: HermiteRFast,
    r_erf: HermiteRFast,
    small_r: SmallR,
    /// R table of the erf kernel in [`quartet_small`].
    small_r_erf: SmallR,
    /// Index into the R table of every (ket tuv, bra tuv), ket-major.
    ridx: Vec<usize>,
    /// pref (-1)^{τ+ν+φ} R_{t+τ,u+ν,v+φ}, ket-major `[jt][it]`; Coulomb kernel.
    rm: Vec<f64>,
    /// The same for the exchange kernel α/r + β erf(ωr)/r.
    rm_x: Vec<f64>,
    /// X[cd][it] = Σ_{ket prims} Σ_jt rm[jt][it] E^{cd}_jt, component-major (Coulomb).
    x: Vec<f64>,
    /// The same for the exchange kernel.
    x_x: Vec<f64>,
    /// Coulomb quartet (ab|cd), row-major `[comp_ab][comp_cd]`.
    v: Vec<f64>,
    /// Exchange-kernel quartet α (ab|cd) + β (ab|cd)_ω.
    v_x: Vec<f64>,
}

/// x[cd][it] += Σ_jt rm[jt][it] E^{cd}_jt for one ket primitive pair.
#[inline]
fn accumulate_ket(nti: usize, cd: &DirectPair, ecd: &[f64], rm: &[f64], x: &mut [f64]) {
    for c in 0..cd.ncomp {
        let xr = &mut x[c * nti..(c + 1) * nti];
        for k in cd.off[c]..cd.off[c + 1] {
            let e = ecd[k];
            let rr = &rm[cd.idx[k] * nti..(cd.idx[k] + 1) * nti];
            for (xv, &r) in xr.iter_mut().zip(rr) {
                *xv += e * r;
            }
        }
    }
}

/// out[ab][cd] += Σ_it E^{ab}_it x[cd][it] for one bra primitive pair.
#[inline]
fn contract_bra(ab: &DirectPair, ncd: usize, eab: &[f64], x: &[f64], out: &mut [f64]) {
    let nti = ab.tuv.len();
    for a in 0..ab.ncomp {
        let (lo, hi) = (ab.off[a], ab.off[a + 1]);
        let (ia, ea) = (&ab.idx[lo..hi], &eab[lo..hi]);
        let orow = &mut out[a * ncd..(a + 1) * ncd];
        for (c, o) in orow.iter_mut().enumerate() {
            let xr = &x[c * nti..(c + 1) * nti];
            let mut s = 0.0;
            for (&it, &e) in ia.iter().zip(ea) {
                s += e * xr[it];
            }
            *o += s;
        }
    }
}

/// Exchange-kernel parameters of a range-separated kernel: (ω, α, β).
type ErfKernel = (f64, f64, f64);

/// Contracted Cartesian quartet (ab|cd) into `s.v`, and for a range-separated kernel
/// α (ab|cd) + β (ab|cd)_ω into `s.v_x` (theory.typ eq:eri, eq:eri-erf, eq:cam-k).
///
/// For every bra primitive pair the ket primitive pairs are summed in Hermite space
/// first (x), so that the bra coefficients are applied once per bra primitive pair.
/// Primitive quartets with Q_a Q_c < `prim_cut` (primitive Schwarz factors) are skipped.
fn quartet(ab: &DirectPair, cd: &DirectPair, erf: Option<ErfKernel>, prim_cut: f64, s: &mut Scratch) {
    match (ab.lab, cd.lab, erf.is_some()) {
        (0, 0, false) => quartet_small::<0, 0, 1, 1, false>(ab, cd, erf, prim_cut, s),
        (0, 1, false) => quartet_small::<0, 1, 1, 4, false>(ab, cd, erf, prim_cut, s),
        (0, 2, false) => quartet_small::<0, 2, 1, 10, false>(ab, cd, erf, prim_cut, s),
        (1, 0, false) => quartet_small::<1, 0, 4, 1, false>(ab, cd, erf, prim_cut, s),
        (1, 1, false) => quartet_small::<1, 1, 4, 4, false>(ab, cd, erf, prim_cut, s),
        (1, 2, false) => quartet_small::<1, 2, 4, 10, false>(ab, cd, erf, prim_cut, s),
        (2, 0, false) => quartet_small::<2, 0, 10, 1, false>(ab, cd, erf, prim_cut, s),
        (2, 1, false) => quartet_small::<2, 1, 10, 4, false>(ab, cd, erf, prim_cut, s),
        (2, 2, false) => quartet_small::<2, 2, 10, 10, false>(ab, cd, erf, prim_cut, s),
        (0, 0, true) => quartet_small::<0, 0, 1, 1, true>(ab, cd, erf, prim_cut, s),
        (0, 1, true) => quartet_small::<0, 1, 1, 4, true>(ab, cd, erf, prim_cut, s),
        (0, 2, true) => quartet_small::<0, 2, 1, 10, true>(ab, cd, erf, prim_cut, s),
        (1, 0, true) => quartet_small::<1, 0, 4, 1, true>(ab, cd, erf, prim_cut, s),
        (1, 1, true) => quartet_small::<1, 1, 4, 4, true>(ab, cd, erf, prim_cut, s),
        (1, 2, true) => quartet_small::<1, 2, 4, 10, true>(ab, cd, erf, prim_cut, s),
        (2, 0, true) => quartet_small::<2, 0, 10, 1, true>(ab, cd, erf, prim_cut, s),
        (2, 1, true) => quartet_small::<2, 1, 10, 4, true>(ab, cd, erf, prim_cut, s),
        (2, 2, true) => quartet_small::<2, 2, 10, 10, true>(ab, cd, erf, prim_cut, s),
        _ => quartet_general(ab, cd, erf, prim_cut, s),
    }
}

/// [`quartet`] for any angular momenta and kernel.
fn quartet_general(ab: &DirectPair, cd: &DirectPair, erf: Option<ErfKernel>, prim_cut: f64, s: &mut Scratch) {
    let (nti, ntj) = (ab.tuv.len(), cd.tuv.len());
    let n = ab.ncomp * cd.ncomp;
    let ltot = ab.lab + cd.lab;
    let dim = ltot + 1;
    s.ridx.clear();
    for g in &cd.tuv {
        for h in &ab.tuv {
            s.ridx.push(((h[0] + g[0]) * dim + h[1] + g[1]) * dim + h[2] + g[2]);
        }
    }
    s.rm.resize(nti * ntj, 0.0);
    s.x.resize(nti * cd.ncomp, 0.0);
    s.v.clear();
    s.v.resize(n, 0.0);
    if erf.is_some() {
        s.rm_x.resize(nti * ntj, 0.0);
        s.x_x.resize(nti * cd.ncomp, 0.0);
        s.v_x.clear();
        s.v_x.resize(n, 0.0);
    }
    let qc_max = cd.prims.first().map_or(0.0, |x| x.q);
    for pa in &ab.prims {
        if pa.q * qc_max < prim_cut {
            break;
        }
        s.x.fill(0.0);
        if erf.is_some() {
            s.x_x.fill(0.0);
        }
        for pc in &cd.prims {
            if pa.q * pc.q < prim_cut {
                break;
            }
            let (p, q) = (pa.p, pc.p);
            let alpha = p * q / (p + q);
            let pq = [0, 1, 2].map(|k| pa.center[k] - pc.center[k]);
            let pref = TWO_PI_POW_2_5 / (p * q * (p + q).sqrt());
            s.r.compute(ltot, alpha, pq);
            let rc = &s.r.data;
            for (jt, &sg) in cd.sign.iter().enumerate() {
                let f = sg * pref;
                let dst = &mut s.rm[jt * nti..(jt + 1) * nti];
                for (d, &ix) in dst.iter_mut().zip(&s.ridx[jt * nti..(jt + 1) * nti]) {
                    *d = f * rc[ix];
                }
            }
            accumulate_ket(nti, cd, &pc.val, &s.rm, &mut s.x);
            if let Some((omega, ka, kb)) = erf {
                // theory.typ eq:eri-erf: exponent α' and factor √(ω²/(α+ω²)).
                let (a_eff, scale) = erf_attenuation(alpha, omega);
                s.r_erf.compute(ltot, a_eff, pq);
                let (rc, re) = (&s.r.data, &s.r_erf.data);
                let (fc, fe) = (ka * pref, kb * pref * scale);
                for (jt, &sg) in cd.sign.iter().enumerate() {
                    let dst = &mut s.rm_x[jt * nti..(jt + 1) * nti];
                    for (d, &ix) in dst.iter_mut().zip(&s.ridx[jt * nti..(jt + 1) * nti]) {
                        *d = sg * (fc * rc[ix] + fe * re[ix]);
                    }
                }
                accumulate_ket(nti, cd, &pc.val, &s.rm_x, &mut s.x_x);
            }
        }
        contract_bra(ab, cd.ncomp, &pa.val, &s.x, &mut s.v);
        if erf.is_some() {
            contract_bra(ab, cd.ncomp, &pa.val, &s.x_x, &mut s.v_x);
        }
    }
}

/// Integral-direct builder of the Coulomb matrix J and the exchange matrix of an
/// [`ExchangeKernel`] (theory.typ eq:jk, eq:cam-k, eq:jk-quartet).
///
/// Prepares the shell-pair data (Hermite coefficients, Schwarz factors) of a basis
/// once; [`DirectJk::coulomb_exchange`] then builds J and K for any density without
/// storing integrals. Memory is O(N²) plus the shell-pair data.
pub struct DirectJk {
    nao: usize,
    ncart: usize,
    /// First normalized-Cartesian index of every shell.
    coff: Vec<usize>,
    /// Number of Cartesian components of every shell.
    ncomp: Vec<usize>,
    pairs: Vec<DirectPair>,
    /// Cartesian Schwarz factors Q_ab (theory.typ eq:schwarz), by `pair_index`.
    q: Vec<f64>,
    /// Basis AO μ = Σ w χ_c over normalized Cartesian components c (eq:sph-d, eq:sph-f).
    to_cart: Vec<Vec<(usize, f64)>>,
    erf: Option<ErfKernel>,
    threshold: f64,
    chunks: Vec<Range<usize>>,
}

impl std::fmt::Debug for DirectJk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectJk")
            .field("nao", &self.nao)
            .field("n_shell_pairs", &self.pairs.len())
            .field("range_separated", &self.erf.is_some())
            .field("threshold", &self.threshold)
            .finish()
    }
}

impl DirectJk {
    /// Prepares direct J/K builds for `basis` with exchange kernel `kernel`. Shell
    /// quartets whose density-weighted Schwarz bound (theory.typ eq:schwarz-dens) is
    /// below `threshold` are skipped.
    pub fn new(basis: &Basis, kernel: ExchangeKernel, threshold: f64) -> Self {
        let shells = basis.shells();
        let mut coff = Vec::with_capacity(shells.len());
        let mut ncomp = Vec::with_capacity(shells.len());
        let mut ncart = 0;
        let mut to_cart = vec![Vec::new(); basis.nao()];
        for sh in shells {
            coff.push(ncart);
            ncomp.push(sh.n_cartesian());
            match sh.spherical_transform() {
                None => {
                    for k in 0..sh.n_cartesian() {
                        to_cart[sh.ao_offset + k].push((ncart + k, 1.0));
                    }
                }
                Some(t) => {
                    for (m, row) in t.iter().enumerate() {
                        for (c, &w) in row.iter().enumerate() {
                            if w != 0.0 {
                                to_cart[sh.ao_offset + m].push((ncart + c, w));
                            }
                        }
                    }
                }
            }
            ncart += sh.n_cartesian();
        }
        let mut idx = Vec::new();
        for a in 0..shells.len() {
            for b in 0..=a {
                idx.push((a, b));
            }
        }
        let pairs: Vec<DirectPair> = idx.par_iter().map(|&(a, b)| DirectPair::new(basis, a, b)).collect();
        let q = cartesian_schwarz_factors(basis);
        // Balance the chunks by the estimated cost of the quartets that survive the
        // Schwarz test for a density of order one.
        let cost: Vec<f64> = (0..pairs.len())
            .into_par_iter()
            .map(|ij| {
                let qa = q[ij];
                pairs[..=ij]
                    .iter()
                    .zip(&q)
                    .filter(|&(_, &qc)| qa * qc >= threshold)
                    .map(|(cd, _)| pairs[ij].cost(cd))
                    .sum::<f64>()
                    + 1.0
            })
            .collect();
        let total: f64 = cost.iter().sum();
        let nchunks = N_CHUNKS.min(pairs.len()).max(1);
        let mut chunks = Vec::with_capacity(nchunks);
        let (mut start, mut acc) = (0, 0.0);
        for (ij, c) in cost.iter().enumerate() {
            acc += c;
            if acc >= total * (chunks.len() + 1) as f64 / nchunks as f64 && chunks.len() + 1 < nchunks {
                chunks.push(start..ij + 1);
                start = ij + 1;
            }
        }
        if start < pairs.len() || chunks.is_empty() {
            chunks.push(start..pairs.len());
        }
        let erf = kernel.is_range_separated().then_some((kernel.omega, kernel.alpha, kernel.beta));
        DirectJk { nao: basis.nao(), ncart, coff, ncomp, pairs, q, to_cart, erf, threshold, chunks }
    }

    /// Number of AOs.
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// Number of unique shell quartets (8-fold symmetry, eq:eri-sym).
    pub fn n_quartets(&self) -> u64 {
        let n = self.pairs.len() as u64;
        n * (n + 1) / 2
    }

    /// Whether the exchange kernel has an erf(ωr)/r part.
    pub fn is_range_separated(&self) -> bool {
        self.erf.is_some()
    }

    /// Coulomb matrix J and, if `want_k`, the exchange matrix for a symmetric matrix
    /// `p` (a density or a density difference), theory.typ eq:jk-quartet.
    ///
    /// The exchange matrix follows the convention of [`crate::integrals::EriTensor`]:
    /// for β = 0 it is the plain K of eq:jk (the caller scales it by α = a_HF), for a
    /// range-separated kernel it is K^x of eq:cam-k (already containing α and β).
    pub fn coulomb_exchange(&self, p: &Matrix, want_k: bool) -> (Matrix, Option<Matrix>, DirectBuildStatistics) {
        self.coulomb_exchange_screened(p, want_k, self.threshold)
    }

    /// [`DirectJk::coulomb_exchange`] with screening threshold `threshold` instead of
    /// the one given to [`DirectJk::new`].
    pub fn coulomb_exchange_screened(
        &self,
        p: &Matrix,
        want_k: bool,
        threshold: f64,
    ) -> (Matrix, Option<Matrix>, DirectBuildStatistics) {
        assert!(p.rows() == self.nao && p.cols() == self.nao, "density does not match the basis");
        let start = Instant::now();
        let nc = self.ncart;
        let pc = self.ao_to_cart(p);
        // max |P| per shell-pair block, by pair_index (theory.typ eq:schwarz-dens).
        let nsh = self.coff.len();
        let mut pmax = vec![0.0f64; self.pairs.len()];
        for a in 0..nsh {
            for b in 0..=a {
                let mut m: f64 = 0.0;
                for x in 0..self.ncomp[a] {
                    let row = &pc[(self.coff[a] + x) * nc..(self.coff[a] + x + 1) * nc];
                    for &v in &row[self.coff[b]..self.coff[b] + self.ncomp[b]] {
                        m = m.max(v.abs());
                    }
                }
                pmax[pair_index(a, b)] = m;
            }
        }
        let mut jt = vec![0.0; nc * nc];
        let mut kt = if want_k { vec![0.0; nc * nc] } else { Vec::new() };
        let mut computed = 0u64;
        // Waves of chunks: each chunk has its own accumulators; after a wave the
        // results are added in chunk order (deterministic for any thread count).
        let wave = (2 * rayon::current_num_threads()).max(1);
        for group in self.chunks.chunks(wave) {
            let results: Vec<(Vec<f64>, Vec<f64>, u64)> = group
                .par_iter()
                .with_max_len(1)
                .map_init(Scratch::default, |s, range| self.chunk(range.clone(), &pc, &pmax, want_k, threshold, s))
                .collect();
            jt.par_chunks_mut(nc).enumerate().for_each(|(row, dst)| {
                for (j, _, _) in &results {
                    for (d, v) in dst.iter_mut().zip(&j[row * nc..(row + 1) * nc]) {
                        *d += v;
                    }
                }
            });
            if want_k {
                kt.par_chunks_mut(nc).enumerate().for_each(|(row, dst)| {
                    for (_, k, _) in &results {
                        for (d, v) in dst.iter_mut().zip(&k[row * nc..(row + 1) * nc]) {
                            *d += v;
                        }
                    }
                });
            }
            computed += results.iter().map(|r| r.2).sum::<u64>();
        }
        // eq:jk-quartet: the accumulators hold one of each transposed pair of terms.
        let j = self.cart_to_ao(&symmetrize(&jt, nc));
        let k = want_k.then(|| self.cart_to_ao(&symmetrize(&kt, nc)));
        let stats = DirectBuildStatistics {
            incremental: false,
            quartets_total: self.n_quartets(),
            quartets_computed: computed,
            seconds: start.elapsed().as_secs_f64(),
        };
        (j, k, stats)
    }

    /// Contributions of the bra shell pairs `range` (each with all kets kl <= ij).
    fn chunk(
        &self,
        range: Range<usize>,
        pc: &[f64],
        pmax: &[f64],
        want_k: bool,
        threshold: f64,
        s: &mut Scratch,
    ) -> (Vec<f64>, Vec<f64>, u64) {
        let nc = self.ncart;
        let mut ja = vec![0.0; nc * nc];
        let mut ka = if want_k { vec![0.0; nc * nc] } else { Vec::new() };
        let mut computed = 0u64;
        let pglob = pmax.iter().fold(0.0f64, |m, &v| m.max(v));
        let qmax = self.q.iter().fold(0.0f64, |m, &v| m.max(v));
        for ij in range {
            let ab = &self.pairs[ij];
            let qab = self.q[ij];
            if qab * qmax * pglob < threshold {
                continue;
            }
            let (a, b) = (ab.sa, ab.sb);
            for (kl, cd) in self.pairs[..=ij].iter().enumerate() {
                let qq = qab * self.q[kl];
                if qq * pglob < threshold {
                    continue;
                }
                let (c, d) = (cd.sa, cd.sb);
                // theory.typ eq:schwarz-dens: the densities that multiply this quartet.
                let mut dm = pmax[ij].max(pmax[kl]);
                if want_k {
                    dm = dm
                        .max(pmax[pair_index(a, c)])
                        .max(pmax[pair_index(a, d)])
                        .max(pmax[pair_index(b, c)])
                        .max(pmax[pair_index(b, d)]);
                }
                if qq * dm < threshold {
                    continue;
                }
                computed += 1;
                quartet(ab, cd, if want_k { self.erf } else { None }, PRIM_FACTOR * threshold / dm, s);
                // Permutational weight (eq:jk-quartet): ½ for each coincidence.
                let mut f = 1.0;
                if a == b {
                    f *= 0.5;
                }
                if c == d {
                    f *= 0.5;
                }
                if ij == kl {
                    f *= 0.5;
                }
                let (oa, ob, oc, od) = (self.coff[a], self.coff[b], self.coff[c], self.coff[d]);
                let (na, nb, ncc, nd) = (self.ncomp[a], self.ncomp[b], self.ncomp[c], self.ncomp[d]);
                let ncd = ncc * nd;
                let v = &s.v;
                // J: J_ab += 2f Σ_cd (ab|cd) P_cd, J_cd += 2f Σ_ab (ab|cd) P_ab.
                for x in 0..na {
                    for y in 0..nb {
                        let row = &v[(x * nb + y) * ncd..(x * nb + y + 1) * ncd];
                        let pab = 2.0 * f * pc[(oa + x) * nc + ob + y];
                        let mut sum = 0.0;
                        for z in 0..ncc {
                            let prow = &pc[(oc + z) * nc + od..(oc + z) * nc + od + nd];
                            let jrow = &mut ja[(oc + z) * nc + od..(oc + z) * nc + od + nd];
                            let vrow = &row[z * nd..(z + 1) * nd];
                            for w in 0..nd {
                                sum += vrow[w] * prow[w];
                                jrow[w] += pab * vrow[w];
                            }
                        }
                        ja[(oa + x) * nc + ob + y] += 2.0 * f * sum;
                    }
                }
                if want_k {
                    // K_ac += f (ab|cd) P_bd, K_bc += f (ab|cd) P_ad,
                    // K_ad += f (ab|cd) P_bc, K_bd += f (ab|cd) P_ac.
                    let vk = if self.erf.is_some() { &s.v_x } else { &s.v };
                    for x in 0..na {
                        let ia = oa + x;
                        for y in 0..nb {
                            let ib = ob + y;
                            let row = &vk[(x * nb + y) * ncd..(x * nb + y + 1) * ncd];
                            for z in 0..ncc {
                                let ic = oc + z;
                                let (p_bc, p_ac) = (pc[ib * nc + ic], pc[ia * nc + ic]);
                                let mut k_ac = 0.0;
                                let mut k_bc = 0.0;
                                for w in 0..nd {
                                    let id = od + w;
                                    let val = f * row[z * nd + w];
                                    k_ac += val * pc[ib * nc + id];
                                    k_bc += val * pc[ia * nc + id];
                                    ka[ia * nc + id] += val * p_bc;
                                    ka[ib * nc + id] += val * p_ac;
                                }
                                ka[ia * nc + ic] += k_ac;
                                ka[ib * nc + ic] += k_bc;
                            }
                        }
                    }
                }
            }
        }
        (ja, ka, computed)
    }

    /// P_cart = Tᵀ P T (row-major `ncart × ncart`).
    fn ao_to_cart(&self, p: &Matrix) -> Vec<f64> {
        let (n, nc) = (self.nao, self.ncart);
        // Y = P T (n × nc), then Tᵀ Y.
        let mut y = vec![0.0; n * nc];
        for mu in 0..n {
            for nu in 0..n {
                let v = p[(mu, nu)];
                for &(d, w) in &self.to_cart[nu] {
                    y[mu * nc + d] += v * w;
                }
            }
        }
        let mut out = vec![0.0; nc * nc];
        for mu in 0..n {
            for &(c, w) in &self.to_cart[mu] {
                for (o, &yv) in out[c * nc..(c + 1) * nc].iter_mut().zip(&y[mu * nc..(mu + 1) * nc]) {
                    *o += w * yv;
                }
            }
        }
        out
    }

    /// M = T M_cart Tᵀ.
    fn cart_to_ao(&self, m: &[f64]) -> Matrix {
        let (n, nc) = (self.nao, self.ncart);
        // Z = T M_cart (n × nc), then Z Tᵀ.
        let mut z = vec![0.0; n * nc];
        for mu in 0..n {
            for &(c, w) in &self.to_cart[mu] {
                for (o, &mv) in z[mu * nc..(mu + 1) * nc].iter_mut().zip(&m[c * nc..(c + 1) * nc]) {
                    *o += w * mv;
                }
            }
        }
        let mut out = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                out[(mu, nu)] = self.to_cart[nu].iter().map(|&(d, w)| w * z[mu * nc + d]).sum();
            }
        }
        out
    }
}

/// A + Aᵀ of a row-major `n × n` matrix.
fn symmetrize(a: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = a[i * n + j] + a[j * n + i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{BasisOptions, BasisSetDefinition};
    use crate::integrals::EriTensor;
    use crate::molecule::{Molecule, Unit};

    fn molecule() -> Molecule {
        Molecule::from_symbols(
            &["C", "O", "H", "H"],
            &[[0.0, 0.05, -0.53], [0.02, 0.0, 0.68], [0.0, 0.94, -1.11], [0.1, -0.94, -1.11]],
            Unit::Angstrom,
            0,
        )
        .unwrap()
    }

    fn test_density(n: usize) -> Matrix {
        Matrix::from_fn(n, n, |i, j| {
            let (a, b) = (i.max(j), i.min(j));
            ((a * 7 + b * 3 + a * b) % 13) as f64 * 0.02 - 0.1 + if i == j { 0.6 } else { 0.0 }
        })
    }

    #[test]
    fn matches_stored_contraction() {
        // J, K and K^x of the direct build against the stored tensors (eq:jk, eq:cam-k),
        // spherical and Cartesian d, without screening.
        for cart in [false, true] {
            let b = Basis::new(&molecule(), &BasisSetDefinition::builtin("6-31g*").unwrap(), BasisOptions {
                cartesian_d: cart,
            })
            .unwrap();
            let p = test_density(b.nao());
            let eri = EriTensor::compute(&b, 0.0);
            let (j0, k0) = eri.coulomb_exchange(&p, true);
            let k0 = k0.unwrap();
            let direct = DirectJk::new(&b, ExchangeKernel::global(1.0), 0.0);
            let (j, k, st) = direct.coulomb_exchange(&p, true);
            assert_eq!(st.quartets_computed, st.quartets_total);
            let k = k.unwrap();
            let dj = (&j - &j0).max_abs();
            let dk = (&k - &k0).max_abs();
            assert!(dj < 1e-12 && dk < 1e-12, "cart={cart}: |dJ| {dj:.2e} |dK| {dk:.2e}");
            // J alone.
            let (j2, none, _) = direct.coulomb_exchange(&p, false);
            assert!(none.is_none());
            assert!((&j2 - &j0).max_abs() < 1e-12);
            // Range-separated kernel.
            let kernel = ExchangeKernel { alpha: 0.19, beta: 0.46, omega: 0.33 };
            let x = eri.exchange_tensor(&b, kernel, 0.0).unwrap();
            let kx0 = x.exchange(&p);
            let direct = DirectJk::new(&b, kernel, 0.0);
            let (j3, kx, _) = direct.coulomb_exchange(&p, true);
            assert!((&j3 - &j0).max_abs() < 1e-12);
            let dkx = (&kx.unwrap() - &kx0).max_abs();
            assert!(dkx < 1e-12, "cart={cart}: |dK^x| {dkx:.2e}");
        }
    }

    #[test]
    fn small_kernel_matches_general() {
        // The unrolled s/p kernel performs the same operations as the general one, so
        // every shell quartet agrees exactly, with and without primitive screening, for
        // the Coulomb kernel alone and with a range-separated exchange kernel.
        let b = Basis::from_name(&molecule(), "6-31g").unwrap();
        let pairs: Vec<DirectPair> =
            (0..b.shells().len()).flat_map(|a| (0..=a).map(move |c| (a, c))).map(|(a, c)| DirectPair::new(&b, a, c)).collect();
        let (mut s1, mut s2) = (Scratch::default(), Scratch::default());
        let mut classes = std::collections::BTreeSet::new();
        let erf = Some((0.33, 0.19, 0.46));
        for ab in &pairs {
            for cd in &pairs {
                assert!(ab.lab <= SMALL_LAB && cd.lab <= SMALL_LAB);
                classes.insert((ab.lab, cd.lab));
                for cut in [0.0, 1e-3] {
                    quartet(ab, cd, None, cut, &mut s1);
                    quartet_general(ab, cd, None, cut, &mut s2);
                    assert_eq!(s1.v, s2.v);
                    quartet(ab, cd, erf, cut, &mut s1);
                    quartet_general(ab, cd, erf, cut, &mut s2);
                    assert_eq!(s1.v, s2.v);
                    assert_eq!(s1.v_x, s2.v_x);
                }
            }
        }
        assert_eq!(classes.len(), (SMALL_LAB + 1) * (SMALL_LAB + 1));
    }

    #[test]
    fn deterministic_and_screened() {
        let b = Basis::from_name(&molecule(), "6-31g*").unwrap();
        let p = test_density(b.nao());
        let direct = DirectJk::new(&b, ExchangeKernel::global(1.0), 1e-10);
        let (j1, k1, s1) = direct.coulomb_exchange(&p, true);
        let (j2, k2, _) = direct.coulomb_exchange(&p, true);
        assert_eq!(j1, j2);
        assert_eq!(k1, k2);
        // A smaller density skips more quartets.
        let (_, _, s3) = direct.coulomb_exchange(&p.scaled(1e-4), true);
        assert!(s3.quartets_computed < s1.quartets_computed);
    }
}

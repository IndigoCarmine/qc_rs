//! Primitive-level McMurchie–Davidson machinery (theory.typ sections 3–5).
//!
//! Everything here works on *unnormalized* primitive Cartesian Gaussians
//! g_lmn(r; a, A) = x_A^l y_A^m z_A^n exp(-a r_A^2) (theory.typ eq:prim) with UNCHECKED(Helgaker2000)
//! arbitrary angular momentum, so that derivative integrals (which need L+1,
//! theory.typ eq:center-deriv) can be built from the same routines. UNCHECKED(Pulay1969)
//! Contraction and normalization live in `integrals.rs`.
//!
//! Reference:
//! - L. E. McMurchie, E. R. Davidson, J. Comput. Phys. 26, 218 (1978): Hermite
//!   expansion of Gaussian products and the Hermite Coulomb integrals R_tuv. The
//!   collected file `docs/papers/mcmurchie1978.pdf` is the typewritten preprint
//!   (UCRL-13791, pages numbered -1- to -20-), not the journal article; all equation
//!   and page numbers in the tags here are the preprint's, and the journal numbering
//!   has not been confirmed.
//!
//! Checked against the paper (preprint), all by derivation: the spec uses the modern
//! form E_t^{ij} = exp(−μ X_AB²) d_t^{ij}, its Λ_t including the Gaussian, whereas the
//! paper's Λ_t (Eq. (2.15)) is the polynomial alone and its coefficients d exclude
//! exp(−μ X_AB²); the recurrences are the same. Checked numerically with mpmath up to
//! f functions: the 1-D identities to 1e-22, nuclear attraction and ERIs against
//! independent quadrature to 1e-15.
//! Suspected misprints in the preprint (they do not affect the code): in Eq. (2.41),
//! p. 9, the term −2n̄α_A d^{n,n̄−1} e^{ℓ+1,ℓ̄} should have the sign +; Eq. (4.3),
//! p. 16, lacks the factor (−2α)^(N+L+M−n−ℓ−m).

use crate::boys::boys;
use std::f64::consts::PI;

/// Hermite expansion coefficients E_t^{ij} for one Cartesian direction. theory.typ eq:E-rec VERIFIED(McMurchie1978, derived from Eqs. (2.20)-(2.22), p. 7 (preprint); E = exp(-mu X_AB^2) d, same recurrence)
/// The expansion they define: theory.typ eq:hermite-exp VERIFIED(McMurchie1978, derived from Eqs. (2.15)-(2.17), (2.25), pp. 6-7 (preprint))
#[derive(Debug, Clone)]
pub struct HermiteE {
    jmax: usize,
    tdim: usize,
    data: Vec<f64>,
}

impl HermiteE {
    /// Builds E_t^{ij} for i <= imax, j <= jmax, with exponents a, b and X_AB = A_x - B_x.
    pub fn new(imax: usize, jmax: usize, a: f64, b: f64, xab: f64) -> Self {
        let p = a + b;
        let mu = a * b / p;
        // theory.typ eq:gpt VERIFIED(McMurchie1978, derived from Eqs. (2.8)-(2.10), (2.14), pp. 5-6 (preprint))
        let xpa = -b / p * xab;
        let xpb = a / p * xab;
        let half_inv_p = 0.5 / p;
        let tdim = imax + jmax + 1;
        let mut e = HermiteE { jmax, tdim, data: vec![0.0; (imax + 1) * (jmax + 1) * tdim] };
        e.data[0] = (-mu * xab * xab).exp();
        for i in 0..=imax {
            for j in 0..=jmax {
                if i == 0 && j == 0 {
                    continue;
                }
                // theory.typ eq:E-rec: raise i from (i-1, 0), otherwise raise j from (i, j-1).
                let (pi, pj, x) = if j == 0 { (i - 1, 0, xpa) } else { (i, j - 1, xpb) };
                for t in 0..=i + j {
                    let mut v = x * e.get(pi, pj, t) + (t + 1) as f64 * e.get(pi, pj, t + 1);
                    if t > 0 {
                        v += half_inv_p * e.get(pi, pj, t - 1);
                    }
                    let idx = e.index(i, j, t);
                    e.data[idx] = v;
                }
            }
        }
        e
    }

    #[inline]
    fn index(&self, i: usize, j: usize, t: usize) -> usize {
        (i * (self.jmax + 1) + j) * self.tdim + t
    }

    /// E_t^{ij}; zero for t > i + j.
    #[inline]
    pub fn get(&self, i: usize, j: usize, t: usize) -> f64 {
        if t > i + j { 0.0 } else { self.data[self.index(i, j, t)] }
    }
}

/// Hermite Coulomb integrals R_tuv(p, R) for t+u+v <= lmax. theory.typ eq:R-rec VERIFIED(McMurchie1978, derived from Eqs. (4.4)-(4.8), pp. 16-17 (preprint))
#[derive(Debug, Clone, Default)]
pub struct HermiteR {
    lmax: usize,
    dim: usize,
    data: Vec<f64>,
    scratch: Vec<f64>,
    fboys: Vec<f64>,
}

impl HermiteR {
    /// Empty table; call [`HermiteR::compute`] before use. Buffers are reused between calls.
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes R_tuv = R^0_tuv for exponent `p` and displacement `r` (e.g. P - C).
    pub fn compute(&mut self, lmax: usize, p: f64, r: [f64; 3]) {
        let dim = lmax + 1;
        let size = dim * dim * dim;
        self.lmax = lmax;
        self.dim = dim;
        self.data.clear();
        self.data.resize(size, 0.0);
        self.scratch.clear();
        self.scratch.resize(size, 0.0);
        self.fboys.resize(dim, 0.0);
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        boys(lmax, p * r2, &mut self.fboys);
        let idx = |t: usize, u: usize, v: usize| (t * dim + u) * dim + v;
        // Layer n is built from layer n+1 (held in `data`) into `scratch`, then swapped.
        let m2p = -2.0 * p;
        for n in (0..=lmax).rev() {
            let m = lmax - n;
            self.scratch[0] = m2p.powi(n as i32) * self.fboys[n];
            for t in 0..=m {
                for u in 0..=m - t {
                    for v in 0..=m - t - u {
                        if t + u + v == 0 {
                            continue;
                        }
                        let nx = &self.data;
                        let val = if t > 0 {
                            let mut s = r[0] * nx[idx(t - 1, u, v)];
                            if t > 1 {
                                s += (t - 1) as f64 * nx[idx(t - 2, u, v)];
                            }
                            s
                        } else if u > 0 {
                            let mut s = r[1] * nx[idx(t, u - 1, v)];
                            if u > 1 {
                                s += (u - 1) as f64 * nx[idx(t, u - 2, v)];
                            }
                            s
                        } else {
                            let mut s = r[2] * nx[idx(t, u, v - 1)];
                            if v > 1 {
                                s += (v - 1) as f64 * nx[idx(t, u, v - 2)];
                            }
                            s
                        };
                        self.scratch[idx(t, u, v)] = val;
                    }
                }
            }
            std::mem::swap(&mut self.data, &mut self.scratch);
        }
    }

    /// R_tuv (requires t+u+v <= lmax of the last `compute`).
    #[inline]
    pub fn get(&self, t: usize, u: usize, v: usize) -> f64 {
        debug_assert!(t + u + v <= self.lmax);
        self.data[(t * self.dim + u) * self.dim + v]
    }

    /// Flat storage of the last `compute`: R_tuv is at `(t * dim + u) * dim + v`
    /// with `dim = lmax + 1` (entries with t+u+v > lmax are unspecified).
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

/// Gaussian product of two primitives with their Hermite E tables. theory.typ eq:gpt
#[derive(Debug, Clone)]
pub struct PrimitivePair {
    /// Exponent of the bra primitive.
    pub a: f64,
    /// Exponent of the ket primitive.
    pub b: f64,
    /// p = a + b.
    pub p: f64,
    /// Product centre P.
    pub center: [f64; 3],
    la_max: usize,
    lb_max: usize,
    e: [HermiteE; 3],
}

impl PrimitivePair {
    /// Pair of primitives with exponent `a` at `ra` and `b` at `rb`, supporting
    /// per-direction powers up to `la_max` / `lb_max`.
    pub fn new(a: f64, ra: [f64; 3], la_max: usize, b: f64, rb: [f64; 3], lb_max: usize) -> Self {
        let p = a + b;
        let center = [0, 1, 2].map(|k| (a * ra[k] + b * rb[k]) / p);
        // lb_max + 2 so that the kinetic-energy integral (eq:kin-1d) is available. VERIFIED(McMurchie1978, derived from Eqs. (2.30), (2.37), p. 8 (preprint); second-derivative form by integration by parts)
        let e = [0, 1, 2].map(|k| HermiteE::new(la_max, lb_max + 2, a, b, ra[k] - rb[k]));
        PrimitivePair { a, b, p, center, la_max, lb_max, e }
    }

    /// E_t^{ij} in direction `k` (0 = x, 1 = y, 2 = z).
    #[inline]
    pub fn e(&self, k: usize, i: usize, j: usize, t: usize) -> f64 {
        self.e[k].get(i, j, t)
    }

    /// E_{tuv}^{ab} = E_t^{l1 l2} E_u^{m1 m2} E_v^{n1 n2}.
    #[inline]
    pub fn hermite_coefficient(&self, la: [usize; 3], lb: [usize; 3], tuv: [usize; 3]) -> f64 {
        self.e(0, la[0], lb[0], tuv[0]) * self.e(1, la[1], lb[1], tuv[1]) * self.e(2, la[2], lb[2], tuv[2])
    }

    fn check(&self, la: [usize; 3], lb: [usize; 3]) {
        debug_assert!(la.iter().all(|&l| l <= self.la_max) && lb.iter().all(|&l| l <= self.lb_max));
    }

    /// 1D overlap S_ij. theory.typ eq:ovlp-1d VERIFIED(McMurchie1978, derived from Eq. (3.3) with (2.17), p. 10 (preprint))
    #[inline]
    fn s1(&self, k: usize, i: usize, j: usize) -> f64 {
        self.e(k, i, j, 0) * (PI / self.p).sqrt()
    }

    /// 1D second-derivative matrix element D^2_ij. theory.typ eq:kin-1d
    fn d2(&self, k: usize, i: usize, j: usize) -> f64 {
        let b = self.b;
        let mut v = -2.0 * b * (2 * j + 1) as f64 * self.s1(k, i, j) + 4.0 * b * b * self.s1(k, i, j + 2);
        if j >= 2 {
            v += (j * (j - 1)) as f64 * self.s1(k, i, j - 2);
        }
        v
    }

    /// Overlap ⟨g_a|g_b⟩. theory.typ eq:ovlp VERIFIED(McMurchie1978, derived from Eq. (3.4) with (2.27), p. 10 (preprint))
    pub fn overlap(&self, la: [usize; 3], lb: [usize; 3]) -> f64 {
        self.check(la, lb);
        self.s1(0, la[0], lb[0]) * self.s1(1, la[1], lb[1]) * self.s1(2, la[2], lb[2])
    }

    /// Kinetic energy ⟨g_a|-∇²/2|g_b⟩. theory.typ eq:kin VERIFIED(McMurchie1978, derived from Eqs. (2.35)-(2.37), p. 8 and p. 11 text (preprint); factor 1/2 not printed)
    pub fn kinetic(&self, la: [usize; 3], lb: [usize; 3]) -> f64 {
        self.check(la, lb);
        let s = [0, 1, 2].map(|k| self.s1(k, la[k], lb[k]));
        let d = [0, 1, 2].map(|k| self.d2(k, la[k], lb[k]));
        -0.5 * (d[0] * s[1] * s[2] + s[0] * d[1] * s[2] + s[0] * s[1] * d[2])
    }

    /// First moments ⟨g_a|r - C|g_b⟩ for the three directions. theory.typ eq:dip-1d VERIFIED(McMurchie1978, derived from Eqs. (3.5)-(3.6), pp. 10-11 (preprint))
    pub fn dipole(&self, la: [usize; 3], lb: [usize; 3], origin: [f64; 3]) -> [f64; 3] {
        self.check(la, lb);
        let s = [0, 1, 2].map(|k| self.s1(k, la[k], lb[k]));
        let m = [0, 1, 2].map(|k| {
            let xpc = self.center[k] - origin[k];
            (self.e(k, la[k], lb[k], 1) + xpc * self.e(k, la[k], lb[k], 0)) * (PI / self.p).sqrt()
        });
        [m[0] * s[1] * s[2], s[0] * m[1] * s[2], s[0] * s[1] * m[2]]
    }

    /// Nuclear-attraction-type integral ⟨g_a|1/|r - C||g_b⟩ given R_tuv(p, P - C)
    /// computed up to at least la+lb. theory.typ eq:nuc-prim VERIFIED(McMurchie1978, derived from Eqs. (3.13)-(3.14), (2.27), pp. 7, 12 (preprint))
    pub fn nuclear(&self, la: [usize; 3], lb: [usize; 3], r: &HermiteR) -> f64 {
        self.check(la, lb);
        let mut sum = 0.0;
        for t in 0..=la[0] + lb[0] {
            let et = self.e(0, la[0], lb[0], t);
            for u in 0..=la[1] + lb[1] {
                let eu = et * self.e(1, la[1], lb[1], u);
                for v in 0..=la[2] + lb[2] {
                    sum += eu * self.e(2, la[2], lb[2], v) * r.get(t, u, v);
                }
            }
        }
        2.0 * PI / self.p * sum
    }
}

fn ltot(l: [usize; 3]) -> usize {
    l[0] + l[1] + l[2]
}

fn lmax(l: [usize; 3]) -> usize {
    l[0].max(l[1]).max(l[2])
}

/// Overlap of two unnormalized primitives. theory.typ eq:ovlp
pub fn overlap_primitive(a: f64, ra: [f64; 3], la: [usize; 3], b: f64, rb: [f64; 3], lb: [usize; 3]) -> f64 {
    PrimitivePair::new(a, ra, lmax(la), b, rb, lmax(lb)).overlap(la, lb)
}

/// Kinetic-energy integral of two unnormalized primitives. theory.typ eq:kin
pub fn kinetic_primitive(a: f64, ra: [f64; 3], la: [usize; 3], b: f64, rb: [f64; 3], lb: [usize; 3]) -> f64 {
    PrimitivePair::new(a, ra, lmax(la), b, rb, lmax(lb)).kinetic(la, lb)
}

/// ⟨g_a|r - origin|g_b⟩ of two unnormalized primitives. theory.typ eq:dip-1d
pub fn dipole_primitive(
    a: f64,
    ra: [f64; 3],
    la: [usize; 3],
    b: f64,
    rb: [f64; 3],
    lb: [usize; 3],
    origin: [f64; 3],
) -> [f64; 3] {
    PrimitivePair::new(a, ra, lmax(la), b, rb, lmax(lb)).dipole(la, lb, origin)
}

/// ⟨g_a|1/|r - C||g_b⟩ of two unnormalized primitives (no charge factor). theory.typ eq:nuc-prim
pub fn nuclear_primitive(
    a: f64,
    ra: [f64; 3],
    la: [usize; 3],
    b: f64,
    rb: [f64; 3],
    lb: [usize; 3],
    rc: [f64; 3],
) -> f64 {
    let pair = PrimitivePair::new(a, ra, lmax(la), b, rb, lmax(lb));
    let mut r = HermiteR::new();
    let pc = [0, 1, 2].map(|k| pair.center[k] - rc[k]);
    r.compute(ltot(la) + ltot(lb), pair.p, pc);
    pair.nuclear(la, lb, &r)
}

/// A primitive Cartesian Gaussian: exponent, centre and powers (l, m, n).
#[derive(Debug, Clone, Copy)]
pub struct Primitive {
    /// Exponent.
    pub exponent: f64,
    /// Centre (Bohr).
    pub center: [f64; 3],
    /// Cartesian powers (l, m, n).
    pub lmn: [usize; 3],
}

/// Exponent α' and prefactor factor √(ω²/(α + ω²)) that turn the Coulomb integral
/// with Gaussian-product exponent α = pq/(p+q) into the erf(ωr)/r integral.
/// theory.typ eq:eri-erf. ω = ∞ gives (α, 1), i.e. the Coulomb integral. VERIFIED(Adamson1999, derived from Eqs. (1), (11)-(13), p. 922; the paper prints the erfc integral)
#[inline]
pub fn erf_attenuation(alpha: f64, omega: f64) -> (f64, f64) {
    if omega.is_infinite() {
        return (alpha, 1.0);
    }
    let w2 = omega * omega;
    let d = alpha + w2;
    (alpha * w2 / d, (w2 / d).sqrt())
}

/// Two-electron integral (g_a g_b | g_c g_d) of unnormalized primitives. theory.typ eq:eri VERIFIED(McMurchie1978, derived from Eqs. (3.30)-(3.34), p. 14 (preprint), with (2.27))
pub fn eri_primitive(ga: &Primitive, gb: &Primitive, gc: &Primitive, gd: &Primitive) -> f64 {
    eri_primitive_kernel(ga, gb, gc, gd, None)
}

/// Attenuated two-electron integral (g_a g_b | erf(ω r₁₂)/r₁₂ | g_c g_d) of unnormalized
/// primitives. theory.typ eq:eri-erf
pub fn eri_primitive_erf(ga: &Primitive, gb: &Primitive, gc: &Primitive, gd: &Primitive, omega: f64) -> f64 {
    eri_primitive_kernel(ga, gb, gc, gd, Some(omega))
}

/// 1/r (`omega = None`, eq:eri) or erf(ωr)/r (eq:eri-erf) integral of four primitives.
fn eri_primitive_kernel(ga: &Primitive, gb: &Primitive, gc: &Primitive, gd: &Primitive, omega: Option<f64>) -> f64 {
    let ab = PrimitivePair::new(ga.exponent, ga.center, lmax(ga.lmn), gb.exponent, gb.center, lmax(gb.lmn));
    let cd = PrimitivePair::new(gc.exponent, gc.center, lmax(gc.lmn), gd.exponent, gd.center, lmax(gd.lmn));
    let (la, lb, lc, ld) = (ga.lmn, gb.lmn, gc.lmn, gd.lmn);
    let (p, q) = (ab.p, cd.p);
    let mut alpha = p * q / (p + q);
    let mut pref = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt());
    if let Some(w) = omega {
        // theory.typ eq:eri-erf: same Hermite sums with α' and an extra prefactor.
        let (a_eff, scale) = erf_attenuation(alpha, w);
        alpha = a_eff;
        pref *= scale;
    }
    let pq = [0, 1, 2].map(|k| ab.center[k] - cd.center[k]);
    let mut r = HermiteR::new();
    r.compute(ltot(la) + ltot(lb) + ltot(lc) + ltot(ld), alpha, pq);
    let mut sum = 0.0;
    for t in 0..=la[0] + lb[0] {
        for u in 0..=la[1] + lb[1] {
            for v in 0..=la[2] + lb[2] {
                let eab = ab.hermite_coefficient(la, lb, [t, u, v]);
                if eab == 0.0 {
                    continue;
                }
                for tau in 0..=lc[0] + ld[0] {
                    for nu in 0..=lc[1] + ld[1] {
                        for phi in 0..=lc[2] + ld[2] {
                            let sign = if (tau + nu + phi) % 2 == 0 { 1.0 } else { -1.0 };
                            let ecd = cd.hermite_coefficient(lc, ld, [tau, nu, phi]);
                            sum += eab * sign * ecd * r.get(t + tau, u + nu, v + phi);
                        }
                    }
                }
            }
        }
    }
    pref * sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D Gauss-Hermite-free brute-force integral on a fine grid.
    fn integrate_1d(f: impl Fn(f64) -> f64) -> f64 {
        let (lo, hi, n) = (-12.0, 12.0, 24_000);
        let h = (hi - lo) / n as f64;
        let mut s = 0.5 * (f(lo) + f(hi));
        for i in 1..n {
            s += f(lo + i as f64 * h);
        }
        s * h
    }

    #[test]
    fn overlap_kinetic_dipole_vs_quadrature_1d() {
        // Integrals factorize; check the 1D pieces by placing everything along x.
        let (a, b) = (0.8, 1.3);
        let (ax, bx) = (0.3, -0.45);
        for i in 0..4 {
            for j in 0..4 {
                let ga = |x: f64| (x - ax).powi(i) * (-a * (x - ax).powi(2)).exp();
                let gb = |x: f64| (x - bx).powi(j) * (-b * (x - bx).powi(2)).exp();
                let s_ref = integrate_1d(|x| ga(x) * gb(x));
                let pair = PrimitivePair::new(a, [ax, 0.0, 0.0], i as usize, b, [bx, 0.0, 0.0], j as usize);
                // y,z overlaps of s functions at the same point: sqrt(pi/p) each
                let syz = PI / (a + b);
                let s = pair.overlap([i as usize, 0, 0], [j as usize, 0, 0]) / syz;
                assert!((s - s_ref).abs() < 1e-12, "S {i}{j}: {s} {s_ref}");
                // second derivative of gb via finite difference inside the integrand
                let h = 1e-4;
                let d2_ref = integrate_1d(|x| ga(x) * (gb(x + h) - 2.0 * gb(x) + gb(x - h)) / (h * h));
                let d2 = pair.d2(0, i as usize, j as usize);
                assert!((d2 - d2_ref).abs() < 1e-6, "D2 {i}{j}: {d2} {d2_ref}");
                let c = 0.7;
                let m_ref = integrate_1d(|x| ga(x) * (x - c) * gb(x));
                let m = pair.dipole([i as usize, 0, 0], [j as usize, 0, 0], [c, 0.0, 0.0])[0] / syz;
                assert!((m - m_ref).abs() < 1e-12, "M {i}{j}: {m} {m_ref}");
            }
        }
    }

    #[test]
    fn nuclear_ss_closed_form() {
        // <s_a|1/|r-C||s_b> = 2π/p exp(-μ AB²) F0(p PC²)
        let (a, b): (f64, f64) = (0.6, 1.1);
        let ra: [f64; 3] = [0.1, -0.2, 0.3];
        let rb: [f64; 3] = [-0.4, 0.5, 0.2];
        let rc: [f64; 3] = [0.9, 0.1, -0.7];
        let p = a + b;
        let pp = [0, 1, 2].map(|k| (a * ra[k] + b * rb[k]) / p);
        let ab2: f64 = (0..3).map(|k| (ra[k] - rb[k]).powi(2)).sum();
        let pc2: f64 = (0..3).map(|k| (pp[k] - rc[k]).powi(2)).sum();
        let mut f = [0.0];
        boys(0, p * pc2, &mut f);
        let exact = 2.0 * PI / p * (-a * b / p * ab2).exp() * f[0];
        let v = nuclear_primitive(a, ra, [0, 0, 0], b, rb, [0, 0, 0], rc);
        assert!((v - exact).abs() < 1e-15);
    }

    #[test]
    fn nuclear_translation_invariance() {
        // d/dC of <a|1/|r-C||b> equals -(d/dA + d/dB); check with finite differences
        // and the centre-derivative rule eq:center-deriv on higher-L primitives.
        let (a, b) = (0.7, 0.9);
        let ra = [0.1, -0.2, 0.3];
        let rb = [-0.4, 0.5, 0.2];
        let rc = [0.6, 0.1, -0.5];
        let la = [1, 0, 1];
        let lb = [0, 2, 0];
        let h = 1e-5;
        let shift = |r: [f64; 3], d: f64| [r[0] + d, r[1], r[2]];
        let dc = (nuclear_primitive(a, ra, la, b, rb, lb, shift(rc, h))
            - nuclear_primitive(a, ra, la, b, rb, lb, shift(rc, -h)))
            / (2.0 * h);
        let da = 2.0 * a * nuclear_primitive(a, ra, [2, 0, 1], b, rb, lb, rc)
            - nuclear_primitive(a, ra, [0, 0, 1], b, rb, lb, rc);
        let db = 2.0 * b * nuclear_primitive(a, ra, la, b, rb, [1, 2, 0], rc);
        assert!((dc + da + db).abs() < 1e-8, "{dc} {da} {db}");
    }

    #[test]
    fn eri_ssss_closed_form() {
        let g = |e: f64, c: [f64; 3]| Primitive { exponent: e, center: c, lmn: [0, 0, 0] };
        let (ga, gb, gc, gd) = (g(0.5, [0.0; 3]), g(1.2, [0.0, 0.0, 1.0]), g(0.8, [0.5, 0.2, 0.0]), g(2.0, [-0.3, 0.0, 0.4]));
        let pair = |x: &Primitive, y: &Primitive| {
            let p = x.exponent + y.exponent;
            let c = [0, 1, 2].map(|k| (x.exponent * x.center[k] + y.exponent * y.center[k]) / p);
            let d2: f64 = (0..3).map(|k| (x.center[k] - y.center[k]).powi(2)).sum();
            (p, c, (-x.exponent * y.exponent / p * d2).exp())
        };
        let (p, pc, kab) = pair(&ga, &gb);
        let (q, qc, kcd) = pair(&gc, &gd);
        let alpha = p * q / (p + q);
        let r2: f64 = (0..3).map(|k| (pc[k] - qc[k]).powi(2)).sum();
        let mut f = [0.0];
        boys(0, alpha * r2, &mut f);
        let exact = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt()) * kab * kcd * f[0];
        assert!((eri_primitive(&ga, &gb, &gc, &gd) - exact).abs() < 1e-14);
    }

    /// erf(x) by composite Simpson quadrature of (2/√π) ∫₀ˣ e^{-t²} dt (independent of
    /// the Boys function).
    fn erf_quad(x: f64) -> f64 {
        let m = 20_000;
        let h = x / m as f64;
        let mut s = 0.0;
        for i in 0..=m {
            let w = if i == 0 || i == m { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
            s += w * (-(i as f64 * h).powi(2)).exp();
        }
        2.0 / PI.sqrt() * s * h / 3.0
    }

    fn s_prims() -> [Primitive; 4] {
        let g = |e: f64, c: [f64; 3]| Primitive { exponent: e, center: c, lmn: [0, 0, 0] };
        [g(0.5, [0.0; 3]), g(1.2, [0.0, 0.0, 1.0]), g(0.8, [0.5, 0.2, 0.0]), g(2.0, [-0.3, 0.0, 0.4])]
    }

    #[test]
    fn eri_erf_ssss_closed_form() {
        // Two s-type charge distributions K_ab e^{-p r_P²}, K_cd e^{-q r_Q²}: convolving the
        // Gaussians gives (π²/pq)^{3/2} K_ab K_cd erf(√α' R)/R with 1/α' = 1/p + 1/q + 1/ω²
        // (the erf(ωr)/r kernel is the Coulomb potential of a Gaussian of exponent ω²).
        let [ga, gb, gc, gd] = s_prims();
        let pair = |x: &Primitive, y: &Primitive| {
            let p = x.exponent + y.exponent;
            let c = [0, 1, 2].map(|k| (x.exponent * x.center[k] + y.exponent * y.center[k]) / p);
            let d2: f64 = (0..3).map(|k| (x.center[k] - y.center[k]).powi(2)).sum();
            (p, c, (-x.exponent * y.exponent / p * d2).exp())
        };
        let (p, pc, kab) = pair(&ga, &gb);
        let (q, qc, kcd) = pair(&gc, &gd);
        let r: f64 = (0..3).map(|k| (pc[k] - qc[k]).powi(2)).sum::<f64>().sqrt();
        for omega in [0.05, 0.33, 1.0, 4.0, 30.0] {
            let a_eff = 1.0 / (1.0 / p + 1.0 / q + 1.0 / (omega * omega));
            let exact = (PI * PI / (p * q)).powf(1.5) * kab * kcd * erf_quad(a_eff.sqrt() * r) / r;
            let v = eri_primitive_erf(&ga, &gb, &gc, &gd, omega);
            assert!((v - exact).abs() < 1e-13 * exact.abs(), "ω={omega}: {v} vs {exact}");
        }
        // Same centres (R = 0): (π²/pq)^{3/2} K K 2√(α'/π).
        let g0 = |e: f64| Primitive { exponent: e, center: [0.1, 0.2, 0.3], lmn: [0, 0, 0] };
        let (p, q, omega) = (0.9 + 1.4, 0.6 + 2.2, 0.7);
        let a_eff = 1.0 / (1.0 / p + 1.0 / q + 1.0 / (omega * omega));
        let exact = (PI * PI / (p * q)).powf(1.5) * 2.0 * (a_eff / PI).sqrt();
        let v = eri_primitive_erf(&g0(0.9), &g0(1.4), &g0(0.6), &g0(2.2), omega);
        assert!((v - exact).abs() < 1e-14 * exact, "{v} vs {exact}");
    }

    #[test]
    fn eri_erf_limits() {
        // ω → ∞ gives the Coulomb integral, ω → 0 gives 0 (linearly in ω), for p and d
        // functions as well as s.
        let g = |e: f64, c: [f64; 3], lmn: [usize; 3]| Primitive { exponent: e, center: c, lmn };
        let sets = [
            s_prims(),
            [
                g(0.7, [0.1, -0.2, 0.3], [1, 0, 1]),
                g(1.1, [-0.4, 0.5, 0.2], [0, 2, 0]),
                g(0.5, [0.6, 0.1, -0.5], [0, 0, 1]),
                g(1.6, [0.0, 0.3, 0.4], [1, 1, 0]),
            ],
        ];
        for [ga, gb, gc, gd] in sets {
            let coul = eri_primitive(&ga, &gb, &gc, &gd);
            assert_eq!(eri_primitive_erf(&ga, &gb, &gc, &gd, f64::INFINITY), coul);
            let big = eri_primitive_erf(&ga, &gb, &gc, &gd, 1e7);
            assert!((big - coul).abs() < 1e-12 * coul.abs(), "{big} vs {coul}");
            let small = eri_primitive_erf(&ga, &gb, &gc, &gd, 1e-8);
            assert!(small.abs() < 1e-7 * coul.abs(), "{small}");
            // Leading behaviour is linear in ω.
            let s2 = eri_primitive_erf(&ga, &gb, &gc, &gd, 2e-8);
            assert!((s2 - 2.0 * small).abs() < 1e-6 * small.abs(), "{small} {s2}");
        }
    }

    #[test]
    fn eri_erf_center_derivative_rule() {
        // eq:center-deriv holds for the attenuated integral: analytic ∂/∂A_x from L ± 1
        // primitives vs central finite differences.
        let omega = 0.33;
        let g = |e: f64, c: [f64; 3], lmn: [usize; 3]| Primitive { exponent: e, center: c, lmn };
        let (a, la) = (0.7, [1, 0, 1]);
        let ra = [0.1, -0.2, 0.3];
        let (gb, gc, gd) =
            (g(1.1, [-0.4, 0.5, 0.2], [0, 2, 0]), g(0.5, [0.6, 0.1, -0.5], [0, 0, 1]), g(1.6, [0.0, 0.3, 0.4], [1, 0, 0]));
        let e = |x: f64, l: [usize; 3]| eri_primitive_erf(&g(a, [ra[0] + x, ra[1], ra[2]], l), &gb, &gc, &gd, omega);
        let h = 1e-5;
        let fd = (e(h, la) - e(-h, la)) / (2.0 * h);
        let an = 2.0 * a * e(0.0, [2, 0, 1]) - e(0.0, [0, 0, 1]);
        assert!((fd - an).abs() < 1e-9, "{fd} vs {an}");
    }
}

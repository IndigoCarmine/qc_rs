//! Values and first/second Cartesian derivatives of the AO basis functions at
//! arbitrary points (theory.typ eq:contracted, eq:ao-deriv, eq:sph-d). UNCHECKED(Helgaker2000) UNCHECKED(Schlegel1995)
//!
//! A Cartesian component is χ = c_lmn P(r_A) g(r_A²) with P = x^l y^m z^n and
//! g = Σ_k d̃_k e^{-a_k r_A²}. With G0 = g, G_i = ∂_i g = x_i R1 and
//! G_ij = ∂_i ∂_j g = δ_ij R1 + x_i x_j R2, where R1 = Σ_k (-2a_k) d̃_k e^{-a_k r²} and
//! R2 = Σ_k 4a_k² d̃_k e^{-a_k r²}, the product rule gives
//! ∂_i χ = c (P_i G0 + P G_i) and ∂_i∂_j χ = c (P_ij G0 + P_i G_j + P_j G_i + P G_ij),
//! which is eq:ao-deriv applied once and twice. Spherical d functions are the fixed
//! linear combinations eq:sph-d of the Cartesian ones (derivatives commute with it).

use crate::basis::{Basis, Shell, cartesian_components, cartesian_norm};

/// Number of derivative components stored for derivative order 0, 1, 2:
/// value; + (x, y, z); + (xx, xy, xz, yy, yz, zz).
pub const N_DERIV_COMPONENTS: [usize; 3] = [1, 4, 10];

/// Index of ∂_x in the component axis (then y = 2, z = 3).
pub const D_X: usize = 1;
/// Index of ∂_x∂_x in the component axis; the second derivatives follow in the
/// order xx, xy, xz, yy, yz, zz.
pub const D_XX: usize = 4;

/// Component index of the second derivative ∂_i ∂_j (i, j ∈ 0..3).
pub fn hessian_component(i: usize, j: usize) -> usize {
    const IDX: [[usize; 3]; 3] = [[4, 5, 6], [5, 7, 8], [6, 8, 9]];
    IDX[i][j]
}

/// AO values (and derivatives) on a batch of points, stored as
/// `data[(component * n_ao + ao) * n_points + point]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AoValues {
    /// Highest derivative order present (0, 1 or 2).
    pub deriv: usize,
    /// Number of AOs (columns of the selected shells).
    pub n_ao: usize,
    /// Number of points.
    pub n_points: usize,
    /// Row-major data, see the type documentation.
    pub data: Vec<f64>,
}

impl AoValues {
    /// Values of component `comp` (0 = value, [`D_X`].. = gradient, [`D_XX`].. = Hessian) of AO `ao` at all points.
    pub fn get(&self, comp: usize, ao: usize) -> &[f64] {
        let s = (comp * self.n_ao + ao) * self.n_points;
        &self.data[s..s + self.n_points]
    }

    /// All AOs of component `comp`, as an `n_ao × n_points` row-major slice.
    pub fn component(&self, comp: usize) -> &[f64] {
        let s = comp * self.n_ao * self.n_points;
        &self.data[s..s + self.n_ao * self.n_points]
    }
}

/// Evaluates every AO of `basis` at `points` up to derivative order `deriv` (≤ 2).
pub fn eval_ao(basis: &Basis, points: &[[f64; 3]], deriv: usize) -> AoValues {
    let all: Vec<usize> = (0..basis.shells().len()).collect();
    eval_ao_shells(basis, &all, points, deriv)
}

/// Evaluates the AOs of the listed shells (in the given order; AO columns follow
/// the shells' own component order) at `points` up to derivative order `deriv` (≤ 2).
pub fn eval_ao_shells(basis: &Basis, shells: &[usize], points: &[[f64; 3]], deriv: usize) -> AoValues {
    assert!(deriv <= 2, "AO derivatives are available up to second order");
    let shells: Vec<&Shell> = shells.iter().map(|&i| &basis.shells()[i]).collect();
    let n_ao: usize = shells.iter().map(|s| s.n_functions()).sum();
    let np = points.len();
    let ncomp = N_DERIV_COMPONENTS[deriv];
    let mut data = vec![0.0; ncomp * n_ao * np];
    let mut col = 0;
    let mut cart = Vec::new();
    for sh in shells {
        let ncart = sh.n_cartesian();
        cart.clear();
        cart.resize(ncomp * ncart * np, 0.0);
        eval_shell_cartesian(sh, points, deriv, &mut cart);
        match sh.spherical_transform() {
            None => {
                for c in 0..ncomp {
                    for k in 0..ncart {
                        let src = &cart[(c * ncart + k) * np..(c * ncart + k + 1) * np];
                        let d = (c * n_ao + col + k) * np;
                        data[d..d + np].copy_from_slice(src);
                    }
                }
            }
            Some(t) => {
                // theory.typ eq:sph-d
                for c in 0..ncomp {
                    for (m, row) in t.iter().enumerate() {
                        let d = (c * n_ao + col + m) * np;
                        let dst = &mut data[d..d + np];
                        for (k, &w) in row.iter().enumerate() {
                            if w == 0.0 {
                                continue;
                            }
                            let src = &cart[(c * ncart + k) * np..(c * ncart + k + 1) * np];
                            for (o, s) in dst.iter_mut().zip(src) {
                                *o += w * s;
                            }
                        }
                    }
                }
            }
        }
        col += sh.n_functions();
    }
    AoValues { deriv, n_ao, n_points: np, data }
}

/// Fills `out[(comp * ncart + k) * np + g]` with the normalized Cartesian components
/// of `sh` and their derivatives. theory.typ eq:contracted, eq:cart-norm, eq:ao-deriv UNCHECKED(Helgaker2000)
fn eval_shell_cartesian(sh: &Shell, points: &[[f64; 3]], deriv: usize, out: &mut [f64]) {
    let np = points.len();
    let comps = cartesian_components(sh.l);
    let ncart = comps.len();
    let norms: Vec<f64> = comps.iter().map(|&c| cartesian_norm(c)).collect();
    let c0 = sh.center;
    for (g, p) in points.iter().enumerate() {
        let x = [p[0] - c0[0], p[1] - c0[1], p[2] - c0[2]];
        let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        let (mut g0, mut r1, mut r2s) = (0.0, 0.0, 0.0);
        for (&a, &d) in sh.exponents.iter().zip(&sh.coefficients) {
            let e = d * (-a * r2).exp();
            g0 += e;
            if deriv >= 1 {
                r1 -= 2.0 * a * e;
                r2s += 4.0 * a * a * e;
            }
        }
        // Monomial factors per axis: (t^n, n t^(n-1), n(n-1) t^(n-2)).
        let fac = |t: f64, n: usize| -> [f64; 3] {
            let pw = |k: i64| if k < 0 { 0.0 } else { t.powi(k as i32) };
            let n_ = n as i64;
            [pw(n_), n as f64 * pw(n_ - 1), (n * n.saturating_sub(1)) as f64 * pw(n_ - 2)]
        };
        for (k, (lmn, &cn)) in comps.iter().zip(&norms).enumerate() {
            let f = [fac(x[0], lmn[0]), fac(x[1], lmn[1]), fac(x[2], lmn[2])];
            let pv = f[0][0] * f[1][0] * f[2][0];
            out[k * np + g] = cn * pv * g0;
            if deriv == 0 {
                continue;
            }
            // P_i: derivative order 1 along axis i only.
            let mut pd = [0.0; 3];
            for i in 0..3 {
                let mut v = 1.0;
                for (ax, fa) in f.iter().enumerate() {
                    v *= fa[usize::from(ax == i)];
                }
                pd[i] = v;
            }
            let gi = [x[0] * r1, x[1] * r1, x[2] * r1];
            for i in 0..3 {
                out[((D_X + i) * ncart + k) * np + g] = cn * (pd[i] * g0 + pv * gi[i]);
            }
            if deriv == 1 {
                continue;
            }
            for i in 0..3 {
                for j in i..3 {
                    let mut pij = 1.0;
                    for (ax, fa) in f.iter().enumerate() {
                        pij *= fa[usize::from(ax == i) + usize::from(ax == j)];
                    }
                    let gij = if i == j { r1 } else { 0.0 } + x[i] * x[j] * r2s;
                    let v = pij * g0 + pd[i] * gi[j] + pd[j] * gi[i] + pv * gij;
                    out[(hessian_component(i, j) * ncart + k) * np + g] = cn * v;
                }
            }
        }
    }
}

/// A distance beyond which every Cartesian component of `sh`, its first and its
/// second derivatives are all smaller than `eps` in magnitude.
///
/// Uses the bound |χ|, |∂χ|, |∂∂χ| ≤ c_max Σ_k |d̃_k| e^{-a_k r²} max(1, r)^L
/// [1 + 2a_k + (L + 2a_k r)²], which follows from the product-rule expressions in
/// the module documentation, and scans outward for its last crossing of `eps`.
pub fn shell_cutoff_radius(sh: &Shell, eps: f64) -> f64 {
    let l = sh.l as i32;
    let cmax = cartesian_components(sh.l).into_iter().map(cartesian_norm).fold(0.0, f64::max);
    let bound = |r: f64| -> f64 {
        let m = r.max(1.0).powi(l);
        sh.exponents
            .iter()
            .zip(&sh.coefficients)
            .map(|(&a, &d)| {
                let t = l as f64 + 2.0 * a * r;
                cmax * d.abs() * (-a * r * r).exp() * m * (1.0 + 2.0 * a + t * t)
            })
            .sum()
    };
    let step = 0.02;
    let mut last = 0.0;
    let mut r = 0.0;
    // The bound decays like a Gaussian of the smallest exponent; stop well past it.
    let a_min = sh.exponents.iter().copied().fold(f64::INFINITY, f64::min);
    let r_max = 10.0 + (60.0 / a_min).sqrt();
    while r <= r_max {
        if bound(r) >= eps {
            last = r;
        }
        r += step;
    }
    last + step
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{BasisOptions, BasisSetDefinition};
    use crate::molecule::{Molecule, Unit};

    fn test_basis(cart: bool) -> Basis {
        let mol = Molecule::from_symbols(
            &["O", "H", "C"],
            &[[0.0, 0.1, 0.2], [1.5, -0.3, 0.4], [-1.2, 1.4, -0.8]],
            Unit::Bohr,
            0,
        )
        .unwrap();
        let def = BasisSetDefinition::builtin("6-31g*").unwrap();
        Basis::new(&mol, &def, BasisOptions { cartesian_d: cart }).unwrap()
    }

    fn test_points() -> Vec<[f64; 3]> {
        vec![[0.3, -0.2, 0.5], [1.1, 0.0, 0.1], [-0.7, 0.9, -0.4], [2.0, 1.5, -1.0], [0.05, 0.12, 0.21]]
    }

    #[test]
    fn derivatives_match_finite_differences() {
        for cart in [false, true] {
            let basis = test_basis(cart);
            let pts = test_points();
            let ao = eval_ao(&basis, &pts, 2);
            // Richardson-extrapolated central differences, O(h⁴): the O 1s primitive
            // with a = 5484 needs it at these step sizes.
            let h = 2e-4;
            let central = |p: &[f64; 3], i: usize, step: f64, comp: usize, mu: usize| {
                let (mut pp, mut pm) = (*p, *p);
                pp[i] += step;
                pm[i] -= step;
                let ap = eval_ao(&basis, &[pp], 1);
                let am = eval_ao(&basis, &[pm], 1);
                (ap.get(comp, mu)[0] - am.get(comp, mu)[0]) / (2.0 * step)
            };
            let fd = |p: &[f64; 3], i, comp, mu| (4.0 * central(p, i, h / 2.0, comp, mu) - central(p, i, h, comp, mu)) / 3.0;
            let mut worst: f64 = 0.0;
            for (g, p) in pts.iter().enumerate() {
                for i in 0..3 {
                    for mu in 0..basis.nao() {
                        // First derivative from values.
                        let an = ao.get(D_X + i, mu)[g];
                        let d1 = (fd(p, i, 0, mu) - an).abs() / an.abs().max(1.0);
                        assert!(d1 < 1e-8, "d{i} ao {mu}: {an}, rel. dev {d1:e}");
                        worst = worst.max(d1);
                        // Second derivatives from gradients.
                        for j in 0..3 {
                            let an2 = ao.get(hessian_component(i, j), mu)[g];
                            let d2 = (fd(p, i, D_X + j, mu) - an2).abs() / an2.abs().max(1.0);
                            assert!(d2 < 1e-8, "d{i}d{j} ao {mu}: {an2}, rel. dev {d2:e}");
                            worst = worst.max(d2);
                        }
                    }
                }
            }
            println!("cart={cart}: worst relative FD deviation {worst:.2e}");
        }
    }

    #[test]
    fn deriv_orders_agree_and_subset_matches() {
        let basis = test_basis(false);
        let pts = test_points();
        let a0 = eval_ao(&basis, &pts, 0);
        let a2 = eval_ao(&basis, &pts, 2);
        assert_eq!(a0.component(0), a2.component(0));
        // Shells 3 and 0 only, in that order.
        let sub = eval_ao_shells(&basis, &[3, 0], &pts, 1);
        let s3 = &basis.shells()[3];
        let s0 = &basis.shells()[0];
        assert_eq!(sub.n_ao, s3.n_functions() + s0.n_functions());
        for k in 0..s3.n_functions() {
            assert_eq!(sub.get(2, k), eval_ao(&basis, &pts, 1).get(2, s3.ao_offset + k));
        }
        assert_eq!(sub.get(0, s3.n_functions()), a0.get(0, s0.ao_offset));
    }

    #[test]
    fn spherical_d_is_normalized_and_matches_definition() {
        // Value of d(0) = zz - xx/2 - yy/2 in terms of the Cartesian functions (eq:sph-d).
        let sph = test_basis(false);
        let cart = test_basis(true);
        let pts = test_points();
        let a = eval_ao(&sph, &pts, 2);
        let b = eval_ao(&cart, &pts, 2);
        let ds = sph.shells().iter().find(|s| s.l == 2).unwrap().ao_offset;
        let dc = cart.shells().iter().find(|s| s.l == 2).unwrap().ao_offset;
        for comp in 0..10 {
            for g in 0..pts.len() {
                let xx = b.get(comp, dc)[g];
                let yy = b.get(comp, dc + 3)[g];
                let zz = b.get(comp, dc + 5)[g];
                let v = a.get(comp, ds + 2)[g];
                assert!((v - (zz - 0.5 * xx - 0.5 * yy)).abs() < 1e-12);
            }
        }
        // ∫ χ² = 1 on a grid for every AO of a single-atom basis (normalization).
        let mol = Molecule::from_symbols(&["O"], &[[0.0; 3]], Unit::Bohr, 0).unwrap();
        let basis =
            Basis::new(&mol, &BasisSetDefinition::builtin("6-31g*").unwrap(), BasisOptions::default()).unwrap();
        let grid = crate::grid::MolecularGrid::new(&mol, crate::grid::GridSettings::fine()).unwrap();
        let ao = eval_ao(&basis, &grid.points, 0);
        for mu in 0..basis.nao() {
            let s: f64 = ao.get(0, mu).iter().zip(&grid.weights).map(|(c, w)| w * c * c).sum();
            assert!((s - 1.0).abs() < 1e-10, "AO {mu}: <χ|χ> = {s}");
        }
    }

    #[test]
    fn cutoff_radius_bounds_values() {
        let basis = test_basis(false);
        let eps = 1e-14;
        for (si, sh) in basis.shells().iter().enumerate() {
            let rc = shell_cutoff_radius(sh, eps);
            assert!(rc > 0.0 && rc < 100.0);
            // Points just beyond the cutoff in a few directions.
            let c = sh.center;
            let pts: Vec<[f64; 3]> = [[1.0, 0.0, 0.0], [0.6, 0.0, 0.8], [0.577, 0.577, 0.578]]
                .iter()
                .map(|d| [c[0] + rc * d[0], c[1] + rc * d[1], c[2] + rc * d[2]])
                .collect();
            let ao = eval_ao_shells(&basis, &[si], &pts, 2);
            assert!(ao.data.iter().all(|v| v.abs() < eps), "shell {si}");
        }
    }
}

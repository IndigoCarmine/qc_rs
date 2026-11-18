//! Vosko–Wilk–Nusair correlation — a Padé-type fit of the correlation energy
//! of the uniform electron gas as a function of r_s (LDA). theory.typ eq:vwn VERIFIED(Vosko1980, Eq. [4.4]; parameters Table 5 caption and p. 1207)
//!
//! Reference:
//! - S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980): gives the
//!   functional form (its Eq. [4.4], with x = r_s^(1/2)) and several parameter
//!   sets. Two are used here (paramagnetic, i.e. closed-shell, parameters only):
//!   - VWN5: the fit to the Ceperley–Alder Monte Carlo correlation energies at
//!     r_s = 10, 20, 50, 100 (Table 5 caption, p. 1207). It is the paramagnetic
//!     part of the spin interpolation the paper recommends (form V, p. 1209).
//!   - VWN-RPA: the fit to the paper's own random-phase-approximation correlation
//!     energies at the same r_s (text, p. 1207).
//!   - A is the paper's 0.0621814 Ry converted to Hartree; x0, b, c are dimensionless.
//!
//! Checked against the paper: the formula term by term and every parameter digit.
//! The code reproduces the fitted column of the paper's Table 5 to 0.005 mRy.
//! The names "VWN5", "VWN-RPA" and "VWN3" are later program conventions, not the
//! paper's (its numerals I–V label spin interpolations, which do not matter for a
//! closed shell), and that Gaussian's B3LYP uses the RPA fit is not in this paper
//! either; that identification rests on Stephens et al. 1994 and on the numerical
//! agreement with PySCF's `b3lypg`.

use crate::dual::Dual;
use std::f64::consts::PI;

/// Paramagnetic VWN parameters (A in Hartree, x0, b, c). theory.typ `<sec:xc>`, table below eq:vwn
#[derive(Debug, Clone, Copy)]
pub(super) struct VwnParams {
    a: f64,
    x0: f64,
    b: f64,
    c: f64,
}

/// Fit to Ceperley–Alder Monte Carlo data.
pub(super) const VWN5: VwnParams = VwnParams { a: 0.0310907, x0: -0.10498, b: 3.72744, c: 12.9352 };
/// Fit to RPA correlation energies.
pub(super) const VWN_RPA: VwnParams = VwnParams { a: 0.0310907, x0: -0.409286, b: 13.0720, c: 42.7198 };

/// f_c^VWN = ρ ε_c^VWN(r_s). theory.typ eq:vwn
pub(super) fn vwn(rho: Dual, p: VwnParams) -> Dual {
    let rs = (3.0 / (4.0 * PI) * rho.recip()).cbrt();
    let x = rs.sqrt();
    let big_x = |x: Dual| x * x + p.b * x + p.c;
    let xx = big_x(x);
    let x0 = p.x0;
    let xx0 = x0 * x0 + p.b * x0 + p.c;
    let q = (4.0 * p.c - p.b * p.b).sqrt();
    let at = (q / (2.0 * x + p.b)).atan();
    let eps = p.a
        * ((x * x / xx).ln() + 2.0 * p.b / q * at
            - p.b * x0 / xx0 * (((x - x0) * (x - x0) / xx).ln() + 2.0 * (p.b + 2.0 * x0) / q * at));
    rho * eps
}

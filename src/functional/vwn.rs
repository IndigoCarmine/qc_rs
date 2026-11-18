//! Vosko–Wilk–Nusair correlation — a Padé-type fit of the correlation energy
//! of the uniform electron gas as a function of r_s (LDA). theory.typ eq:vwn
//!
//! Reference:
//! - S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980): gives the
//!   functional form and several parameter sets. Two are used here (paramagnetic,
//!   i.e. closed-shell, parameters only):
//!   - VWN5: fitted to the Ceperley–Alder quantum Monte Carlo data; the
//!     parametrization recommended in the paper ("VWN5", used in B3LYP5).
//!   - VWN-RPA: fitted to the random-phase-approximation correlation energy
//!     ("VWN3" in Gaussian, "VWN_RPA" in libxc); the one in Gaussian's B3LYP.

use crate::dual::Dual;
use std::f64::consts::PI;

/// Paramagnetic VWN parameters (A, x0, b, c). theory.typ section 9 table
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

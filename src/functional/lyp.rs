//! Lee–Yang–Parr correlation — a gradient-corrected (GGA) correlation functional
//! obtained by turning the Colle–Salvetti correlation-energy formula into a
//! functional of the density. theory.typ eq:lyp
//!
//! References:
//! - C. Lee, W. Yang, R. G. Parr, Phys. Rev. B 37, 785 (1988): the functional
//!   and its parameters a, b, c, d (from Colle and Salvetti's fit to helium).
//! - B. Miehlich, A. Savin, H. Stoll, H. Preuss, Chem. Phys. Lett. 157, 200
//!   (1989): integration by parts removes the Laplacian of the density, leaving
//!   a form in ρ and |∇ρ|² only. The closed-shell reduction of that form
//!   (ρ_α = ρ_β = ρ/2) is what is implemented here.

use crate::dual::Dual;
use std::f64::consts::PI;

/// f_c^LYP for a closed shell. theory.typ eq:lyp
pub(super) fn lyp(rho: Dual, sigma: Dual) -> Dual {
    const A: f64 = 0.04918;
    const B: f64 = 0.132;
    const C: f64 = 0.2533;
    const D: f64 = 0.349;
    let cf = 0.3 * (3.0 * PI * PI).powf(2.0 / 3.0);
    let rm13 = rho.powf(-1.0 / 3.0);
    let denom = 1.0 + D * rm13;
    let omega = (-C * rm13).exp() / denom * rho.powf(-11.0 / 3.0);
    let delta = C * rm13 + D * rm13 / denom;
    -A * rho / denom - A * B * omega * rho * rho * (cf * rho.powf(8.0 / 3.0) - sigma * (3.0 + 7.0 * delta) / 72.0)
}

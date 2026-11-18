//! Lee–Yang–Parr correlation — a gradient-corrected (GGA) correlation functional
//! obtained by turning the Colle–Salvetti correlation-energy formula into a
//! functional of the density.
//! theory.typ eq:lyp VERIFIED(Lee1988, derived from Eq. (21) with t_W of Eq. (8) by integration by parts; parameters p. 785) UNCHECKED(Miehlich1989)
//!
//! References:
//! - C. Lee, W. Yang, R. G. Parr, Phys. Rev. B 37, 785 (1988): the functional,
//!   closed shell Eq. (21), p. 787, open shell Eq. (22), both written with the
//!   Weizsäcker term t_W of Eq. (8), p. 786, and the Laplacian ∇²ρ. Parameters
//!   a = 0.04918, b = 0.132, c = 0.2533, d = 0.349 (p. 785, after Eq. (3)) are
//!   Colle and Salvetti's values, fitted to the Hartree–Fock orbital of the helium
//!   atom only; LYP did not refit them. C_F = (3/10)(3π²)^(2/3) is Eq. (16), p. 786.
//! - B. Miehlich, A. Savin, H. Stoll, H. Preuss, Chem. Phys. Lett. 157, 200
//!   (1989): the Laplacian-free form in ρ and |∇ρ|², written with ω and δ. Needed
//!   here only as the source of that notation; the form itself follows from Lee1988.
//!
//! Checked against the paper: the closed-shell energy density below is not printed
//! in Lee1988. It follows from Eq. (21) with t_W = σ/(8ρ) − ∇²ρ/8 (Eq. (8); note the
//! Laplacian term) by integration by parts, which turns the σ coefficient into
//! (ab/72)·G₁·[17 + 21(1 − 5/3 + δ/3)] = (ab/72)·G₁·(3 + 7δ), G₁ = ω ρ². The two
//! energy densities differ point by point (10–40 % at r = 1 for test densities), but
//! their integrals agree to 20 digits for five test densities (radial quadrature
//! with mpmath).

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

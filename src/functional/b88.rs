//! Becke 88 exchange — Slater exchange plus a gradient correction (GGA) built
//! to give the correct -1/r asymptotic behaviour of the exchange energy density.
//! theory.typ eq:b88
//!
//! Reference:
//! - A. D. Becke, Phys. Rev. A 38, 3098 (1988): defines the correction
//!   -β ρ_s^(4/3) x² / (1 + 6β x asinh x) per spin, x = |∇ρ_s| / ρ_s^(4/3), with
//!   the single parameter β = 0.0042 fitted to the exchange energies of the
//!   noble-gas atoms.

use super::slater::slater;
use crate::dual::Dual;

/// f_x^B88 for a closed shell (ρ_s = ρ/2, |∇ρ_s| = √σ/2). theory.typ eq:b88
pub(super) fn b88(rho: Dual, sigma: Dual) -> Dual {
    const BETA: f64 = 0.0042;
    let rho_s43 = (rho * 0.5).powf(4.0 / 3.0);
    let x = sigma.sqrt() * 0.5 / rho_s43;
    slater(rho) - 2.0 * BETA * rho_s43 * x * x / (1.0 + 6.0 * BETA * x * x.asinh())
}

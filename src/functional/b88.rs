//! Becke 88 exchange — Slater exchange plus a gradient correction (GGA) whose
//! large-gradient form makes the exchange-energy density per electron (the
//! potential of the Fermi hole) decay as -1/r for an exponentially decaying density.
//! theory.typ eq:b88 VERIFIED(Becke1988, Eqs. (1), (4), (8), pp. 3098-3099; closed shell by rho_s = rho/2)
//!
//! Reference:
//! - A. D. Becke, Phys. Rev. A 38, 3098 (1988): defines the correction
//!   -β ρ_s^(4/3) x² / (1 + 6β x asinh x) per spin, x = |∇ρ_s| / ρ_s^(4/3)
//!   (its Eqs. (4), (8)). The single parameter β = 0.0042 a.u. is a least-squares
//!   fit to the Hartree–Fock exchange energies of the six noble-gas atoms He–Rn;
//!   the coefficient 6 is fixed by the -1/r condition, not fitted.
//!
//! Checked against the paper: the paper's formula is per spin; the closed-shell
//! form below follows with ρ_s = ρ/2, |∇ρ_s| = √σ/2, summed over both spins
//! (agreement with the per-spin formula to 1e-16 relative at six (ρ, σ) points).

use super::slater::slater;
use crate::dual::Dual;

/// f_x^B88 for a closed shell (ρ_s = ρ/2, |∇ρ_s| = √σ/2). theory.typ eq:b88
pub(super) fn b88(rho: Dual, sigma: Dual) -> Dual {
    const BETA: f64 = 0.0042;
    let rho_s43 = (rho * 0.5).powf(4.0 / 3.0);
    let x = sigma.sqrt() * 0.5 / rho_s43;
    slater(rho) - 2.0 * BETA * rho_s43 * x * x / (1.0 + 6.0 * BETA * x * x.asinh())
}

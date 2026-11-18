//! Slater (Dirac) exchange — the exchange energy of the uniform electron gas,
//! applied locally (LDA). theory.typ eq:slater UNCHECKED(Dirac1930) VERIFIED(Becke1988, Eqs. (1a), (1b), p. 3098; closed shell by rho_s = rho/2) VERIFIED(Kohn1965, derived from Eqs. (2.7), (2.19), pp. A1134-A1135)
//!
//! References:
//! - P. A. M. Dirac, Math. Proc. Cambridge Philos. Soc. 26, 376 (1930):
//!   exchange energy of the uniform electron gas, proportional to ρ^(4/3).
//! - J. C. Slater, Phys. Rev. 81, 385 (1951): replaces the Hartree–Fock exchange by
//!   the free-electron exchange potential averaged over the occupied states,
//!   −3(3ρ/8π)^(1/3) Hartree (Eqs. (13), (14), p. 390). That is 3/2 times ∂f_x/∂ρ of
//!   the f_x used here (the well-known factor 2/3, noted by Kohn and Sham next to their
//!   Eqs. (2.18)-(2.19)), and the paper gives no energy functional. It is not the
//!   potential used here; the name "Slater exchange" is only conventional.
//! - W. Kohn, L. J. Sham, Phys. Rev. 140, A1133 (1965): the exchange potential
//!   μ_x = −(1/π)(3π²n)^(1/3) = −(3n/π)^(1/3) (Eq. (2.19), p. A1135; Hartree, total
//!   density), which is exactly ∂f_x/∂ρ of f_x below; with Eq. (2.7), μ_x = d(n ε_x)/dn,
//!   integration from n = 0 gives f_x. This fixes the coefficient used here. The paper
//!   prints the potential, not the ρ^(4/3) energy (that form is Dirac's).
//! - A. D. Becke, Phys. Rev. A 38, 3098 (1988), Eqs. (1a), (1b): the same LDA
//!   exchange in per-spin form, E_x = -C_x Σ_s ∫ ρ_s^(4/3), C_x = (3/2)(3/4π)^(1/3);
//!   with ρ_s = ρ/2 this is -(3/4)(3/π)^(1/3) ρ^(4/3). The coefficient was checked
//!   against this paper (a secondary source).
//!
//! Checked against the papers: the coefficient against Becke1988 Eqs. (1a), (1b) and,
//! by derivation, against Kohn1965 Eqs. (2.7), (2.19). Dirac1930 is not collected;
//! Slater1951 does not give this coefficient (see above).

use crate::dual::Dual;
use std::f64::consts::PI;

/// f_x^S = -3/4 (3/π)^(1/3) ρ^(4/3). theory.typ eq:slater
pub(super) fn slater(rho: Dual) -> Dual {
    -0.75 * (3.0 / PI).cbrt() * rho.powf(4.0 / 3.0)
}

//! Slater (Dirac) exchange — the exchange energy of the uniform electron gas,
//! applied locally (LDA). theory.typ eq:slater
//!
//! References:
//! - P. A. M. Dirac, Math. Proc. Cambridge Philos. Soc. 26, 376 (1930):
//!   exchange energy of the uniform electron gas, proportional to ρ^(4/3).
//! - J. C. Slater, Phys. Rev. 81, 385 (1951): use of that ρ^(1/3) exchange
//!   potential as a local approximation in self-consistent-field calculations.

use crate::dual::Dual;
use std::f64::consts::PI;

/// f_x^S = -3/4 (3/π)^(1/3) ρ^(4/3). theory.typ eq:slater
pub(super) fn slater(rho: Dual) -> Dual {
    -0.75 * (3.0 / PI).cbrt() * rho.powf(4.0 / 3.0)
}

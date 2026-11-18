//! Short-range Becke 88 exchange of the Coulomb-attenuating method (CAM): the B88
//! exchange that belongs to the part [1 − α − β erf(ωr)]/r of the split Coulomb
//! operator, the rest being treated as exact exchange. theory.typ eq:cam-K, eq:cam-a,
//! eq:cam-att, eq:cam-b88, eq:cam-series
//!
//! References:
//! - T. Yanai, D. P. Tew, N. C. Handy, Chem. Phys. Lett. 393, 51 (2004): CAM-B3LYP;
//!   the three-parameter split (α, β, ω) of the operator and the values α = 0.19,
//!   β = 0.46, ω = 0.33 used with B3LYP-type DFT exchange and correlation.
//! - H. Iikura, T. Tsuneda, T. Yanai, K. Hirao, J. Chem. Phys. 115, 3540 (2001): the
//!   long-range correction scheme for GGA exchange — the erf-attenuated exchange
//!   energy of a GGA written through its K_s and the attenuation factor A(a_s).
//! - A. D. Becke, Phys. Rev. A 38, 3098 (1988): the B88 exchange functional whose K_s
//!   is attenuated here (β_B88 = 0.0042).

use crate::dual::Dual;
use std::f64::consts::PI;

/// β of Becke 88.
const BETA_B88: f64 = 0.0042;

/// For a ≥ this value 1 − A(a) is summed as the series in 1/a² (theory.typ
/// eq:cam-series, all terms); below it eq:cam-att is used directly.
///
/// Measured against a 60-digit evaluation: the direct formula loses digits as a grows
/// (relative error of 1 − A ~1e-14 at a = 1, ~1e-12 at a = 3, ~1e-11 at a = 6), while
/// the full series is at the rounding level (< 1e-15) for every a ≥ 0.2 and needs
/// about 17 terms at a = 0.5 (u = 1/(4a²) = 1). Switching at a = 0.5 keeps both
/// branches within ~1e-14 relative (unit test `series_and_direct_formula_agree`).
pub(super) const SERIES_SWITCH: f64 = 0.5;

/// Below a = 1/(2√700), i.e. 1/(4a²) > 700, b = −1 and erf(1/(2a)) = 1 to double
/// precision (theory.typ section 9, short-range B88).
const SMALL_A_EXPONENT: f64 = 700.0;

/// 1 − A(a) by eq:cam-att (`series = false`) or by the series eq:cam-series
/// (`series = true`), without the switch. Exposed for the unit tests.
pub(super) fn one_minus_attenuation_with(a: Dual, series: bool) -> Dual {
    // u = 1/(4a²) = y², y = 1/(2a).
    let u = 0.25 * (a * a).recip();
    if series {
        // theory.typ eq:cam-series in closed form: 1 − A = Σ_{n≥1} (−1)^{n+1} 2 uⁿ /
        // (n! (2n+1)(n+1)(n+2)); n = 1..5 are the coefficients 1/36, −1/960, 1/26880,
        // −1/829440, 1/28385280 of eq:cam-series (u = 1/(4a²)). The series converges
        // for every a; above the switch (u ≤ 1) the terms fall quickly and alternate
        // without significant cancellation.
        let mut pow = Dual::constant(1.0); // uⁿ/n!
        let mut sum = Dual::constant(0.0);
        for n in 1..=40 {
            pow = pow * u * (1.0 / n as f64);
            let nf = n as f64;
            let c = 2.0 / ((2.0 * nf + 1.0) * (nf + 1.0) * (nf + 2.0));
            let term = c * pow;
            sum = if n % 2 == 1 { sum + term } else { sum - term };
            if term.v.abs() < 1e-18 * sum.v.abs() {
                break;
            }
        }
        sum
    } else if u.v > SMALL_A_EXPONENT {
        // b = −1, erf = 1: A = 8/3 a [√π + 2a(−1 − c)] with c = −2a² + 1/2.
        let c = 0.5 - 2.0 * a * a;
        1.0 - 8.0 / 3.0 * a * (PI.sqrt() + 2.0 * a * (-1.0 - c))
    } else {
        // theory.typ eq:cam-a, eq:cam-att; b = e^{−1/(4a²)} − 1 via exp_m1 (no
        // cancellation when 1/(4a²) is small).
        let b = (-u).exp_m1();
        let c = 2.0 * a * a * b + 0.5;
        let y = 0.5 * a.recip();
        1.0 - 8.0 / 3.0 * a * (PI.sqrt() * y.erf() + 2.0 * a * (b - c))
    }
}

/// 1 − A(a): the fraction of the B88 exchange that stays at short range (theory.typ
/// eq:cam-att, eq:cam-series).
pub(super) fn one_minus_attenuation(a: Dual) -> Dual {
    one_minus_attenuation_with(a, a.v >= SERIES_SWITCH)
}

/// f_x^CAM-B88 for a closed shell (ρ_s = ρ/2, |∇ρ_s| = √σ/2). theory.typ eq:cam-b88
///
/// f = −ρ_s^{4/3} K_s [1 − α − β A(a_s)] summed over both spins, written as
/// −ρ_s^{4/3} K_s [(1 − α − β) + β (1 − A)] so that the large-a series supplies 1 − A
/// without cancellation.
pub(super) fn cam_b88(rho: Dual, sigma: Dual, alpha: f64, beta: f64, omega: f64) -> Dual {
    let rho_s = rho * 0.5;
    let rho_s43 = rho_s.powf(4.0 / 3.0);
    let x = sigma.sqrt() * 0.5 / rho_s43;
    // theory.typ eq:cam-K
    let k = 2.0 * (1.5 * (3.0 / (4.0 * PI)).cbrt() + BETA_B88 * x * x / (1.0 + 6.0 * BETA_B88 * x * x.asinh()));
    let mut factor = Dual::constant(1.0 - alpha - beta);
    if beta != 0.0 {
        // theory.typ eq:cam-a
        let a = omega * k.sqrt() / (6.0 * PI.sqrt() * rho_s.cbrt());
        factor = factor + beta * one_minus_attenuation(a);
    }
    // Two spins of −½ ρ_s^{4/3} K_s [..] each.
    -(rho_s43 * k * factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_minus_a(a: f64, series: bool) -> Dual {
        one_minus_attenuation_with(Dual::rho(a), series)
    }

    #[test]
    fn series_coefficients_are_those_of_the_spec() {
        // 1 − A ≈ 1/(36a²) − 1/(960a⁴) + 1/(26880a⁶) − 1/(829440a⁸) + 1/(28385280a¹⁰)
        // (eq:cam-series); at a = 40 the sixth term is below 1e-28 relative.
        let a: f64 = 40.0;
        let five = 1.0 / (36.0 * a.powi(2)) - 1.0 / (960.0 * a.powi(4)) + 1.0 / (26880.0 * a.powi(6))
            - 1.0 / (829440.0 * a.powi(8))
            + 1.0 / (28385280.0 * a.powi(10));
        let s = one_minus_a(a, true).v;
        assert!((s - five).abs() < 2e-16 * five, "{s} vs {five}");
    }

    #[test]
    fn series_and_direct_formula_agree() {
        // Around the switch, value and derivative of 1 − A from eq:cam-att and from the
        // series agree to ~1e-12 relative (in fact much better).
        let mut worst: f64 = 0.0;
        for i in 0..=60 {
            let a = 0.2 + 0.02 * i as f64; // 0.2 … 1.4
            let (d, s) = (one_minus_a(a, false), one_minus_a(a, true));
            let rel_v = (d.v - s.v).abs() / s.v.abs();
            let rel_d = (d.d_rho - s.d_rho).abs() / s.d_rho.abs();
            if (SERIES_SWITCH / 2.0..=2.0 * SERIES_SWITCH).contains(&a) {
                assert!(rel_v < 1e-12 && rel_d < 1e-12, "a={a}: value {rel_v:.1e}, derivative {rel_d:.1e}");
                worst = worst.max(rel_v).max(rel_d);
            }
        }
        println!("worst relative difference of direct and series 1 − A for a in [{}, {}]: {worst:.1e}", SERIES_SWITCH / 2.0, 2.0 * SERIES_SWITCH);
    }

    #[test]
    fn attenuation_is_continuous_and_has_the_right_limits() {
        // Continuity of the switched function (value and slope) at the switch point.
        // The values on both sides differ by the slope times the step.
        let da = SERIES_SWITCH * 1e-12;
        let below = one_minus_attenuation(Dual::rho(SERIES_SWITCH - da));
        let at = one_minus_attenuation(Dual::rho(SERIES_SWITCH));
        let jump = below.v - (at.v - at.d_rho * da);
        assert!(jump.abs() < 5e-14 * at.v, "{} {} jump {jump:e}", below.v, at.v);
        assert!((below.d_rho - at.d_rho).abs() < 1e-11 * at.d_rho.abs());
        // Continuity at the small-a branch: 1/(4a²) = 700.
        let a0 = 0.5 / SMALL_A_EXPONENT.sqrt();
        let da = a0 * 1e-13;
        let (l, r) = (one_minus_attenuation(Dual::rho(a0 - da)), one_minus_attenuation(Dual::rho(a0 + da)));
        let jump = r.v - (l.v + 2.0 * da * l.d_rho);
        assert!(jump.abs() < 1e-15 && (l.d_rho - r.d_rho).abs() < 1e-11 * l.d_rho.abs(), "jump {jump:e}");
        // a → 0: A → 0 (1 − A → 1); a → ∞: A → 1 with 1 − A ≈ 1/(36a²).
        assert!((one_minus_attenuation(Dual::rho(1e-6)).v - 1.0).abs() < 1e-5);
        let big = one_minus_attenuation(Dual::rho(1e6)).v;
        assert!((big * 36.0e12 - 1.0).abs() < 1e-11);
        // Monotonic decrease of 1 − A in between.
        let mut last = 1.0;
        for i in 0..400 {
            let a = 10f64.powf(-3.0 + i as f64 * 0.02);
            let v = one_minus_attenuation(Dual::rho(a)).v;
            assert!(v < last && v > 0.0, "a={a}: {v}");
            last = v;
        }
    }

    #[test]
    fn reduces_to_b88_without_attenuation() {
        // α = β = 0: f_x^CAM-B88 = f_x^B88 (theory.typ eq:cam-b88).
        for &(rho, sigma) in &[(1e-6, 1e-12), (0.05, 0.01), (2.0, 3.0), (300.0, 1e5)] {
            let a = cam_b88(Dual::rho(rho), Dual::sigma(sigma), 0.0, 0.0, 0.33);
            let b = super::super::b88::b88(Dual::rho(rho), Dual::sigma(sigma));
            for (x, y) in [(a.v, b.v), (a.d_rho, b.d_rho), (a.d_sigma, b.d_sigma)] {
                assert!((x - y).abs() <= 1e-14 * y.abs(), "ρ={rho} σ={sigma}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn potentials_match_finite_differences_over_all_densities() {
        // v_ρ and v_σ of CAM-B3LYP's exchange against central differences for ρ from
        // 1e-10 to 1e3, with σ from 0 (the floor 1e-40) to large reduced gradients; the
        // range covers both branches of 1 − A(a) and the small-a limit.
        let (alpha, beta, omega) = (0.19, 0.46, 0.33);
        let f = |r: f64, s: f64| cam_b88(Dual::rho(r), Dual::sigma(s), alpha, beta, omega);
        let mut n_series = 0;
        let mut n_direct = 0;
        for i in 0..=26 {
            let rho = 10f64.powf(-10.0 + 0.5 * i as f64);
            // σ relative to ρ^{8/3}: the reduced gradient x_s ∝ √σ / ρ^{4/3}.
            for xs in [0.0, 1e-3, 0.1, 1.0, 10.0, 100.0] {
                let sigma = (xs * rho.powf(4.0 / 3.0)).powi(2).max(1e-40);
                let k = 2.0 * 1.5 * (3.0 / (4.0 * PI)).cbrt();
                let a = omega * k.sqrt() / (6.0 * PI.sqrt() * (0.5 * rho).cbrt());
                if a >= SERIES_SWITCH { n_series += 1 } else { n_direct += 1 }
                let x = f(rho, sigma);
                let hr = 1e-5 * rho;
                let fr = (f(rho + hr, sigma).v - f(rho - hr, sigma).v) / (2.0 * hr);
                assert!(
                    (x.d_rho - fr).abs() <= 1e-7 * fr.abs(),
                    "ρ={rho:e} σ={sigma:e}: v_ρ {} vs {fr}",
                    x.d_rho
                );
                // σ steps on the scale ρ^{8/3} of the reduced gradient; one-sided
                // (second order) where σ is smaller than that, including σ → 0.
                let h = 1e-3 * rho.powf(8.0 / 3.0);
                let fs = if sigma > 2.0 * h {
                    let hs = h.min(1e-4 * sigma);
                    (f(rho, sigma + hs).v - f(rho, sigma - hs).v) / (2.0 * hs)
                } else {
                    (-3.0 * f(rho, sigma).v + 4.0 * f(rho, sigma + h).v - f(rho, sigma + 2.0 * h).v) / (2.0 * h)
                };
                assert!(
                    (x.d_sigma - fs).abs() <= 1e-5 * fs.abs(),
                    "ρ={rho:e} σ={sigma:e}: v_σ {} vs {fs}",
                    x.d_sigma
                );
            }
        }
        assert!(n_series > 0 && n_direct > 0);
    }
}

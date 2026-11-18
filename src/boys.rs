//! Boys function F_n(x). theory.typ eq:boys UNCHECKED(Helgaker2000)

/// Fills `out[0..=nmax]` with F_0(x) … F_nmax(x) (x >= 0).
///
/// Below x = 35 the highest order is summed as a series and the rest follow
/// by downward recursion; above, the asymptotic F_0 plus upward recursion.
pub fn boys(nmax: usize, x: f64, out: &mut [f64]) {
    debug_assert!(out.len() > nmax);
    debug_assert!(x >= 0.0);
    let ex = (-x).exp();
    if x < 35.0 {
        // theory.typ eq:boys-series UNCHECKED(Helgaker2000)
        let mut term = 1.0 / (2 * nmax + 1) as f64;
        let mut sum = term;
        let mut k = 1usize;
        loop {
            term *= 2.0 * x / (2 * nmax + 2 * k + 1) as f64;
            sum += term;
            if term < 1e-17 * sum {
                break;
            }
            k += 1;
        }
        out[nmax] = ex * sum;
        // theory.typ eq:boys-down UNCHECKED(Helgaker2000)
        for n in (1..=nmax).rev() {
            out[n - 1] = (2.0 * x * out[n] + ex) / (2 * n - 1) as f64;
        }
    } else {
        // theory.typ eq:boys-up UNCHECKED(Helgaker2000)
        out[0] = 0.5 * (std::f64::consts::PI / x).sqrt();
        for n in 0..nmax {
            out[n + 1] = ((2 * n + 1) as f64 * out[n] - ex) / (2.0 * x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Composite Simpson quadrature of the defining integral.
    /// Uses Kahan summation so that rounding does not limit the reference.
    fn boys_quad(n: usize, x: f64) -> f64 {
        let m = 40_000;
        let h = 1.0 / m as f64;
        let f = |t: f64| t.powi(2 * n as i32) * (-x * t * t).exp();
        let (mut s, mut comp) = (0.0f64, 0.0f64);
        for i in 0..=m {
            let w = if i == 0 || i == m { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
            let y = w * f(i as f64 * h) - comp;
            let t = s + y;
            comp = (t - s) - y;
            s = t;
        }
        s * h / 3.0
    }

    #[test]
    fn against_quadrature() {
        let mut out = [0.0; 13];
        for &x in &[0.0, 1e-8, 0.3, 1.0, 5.0, 12.5, 30.0, 34.999, 35.0, 50.0, 120.0] {
            boys(12, x, &mut out);
            for (n, &f) in out.iter().enumerate() {
                let q = boys_quad(n, x);
                assert!((f - q).abs() <= 1e-13 * q + 1e-16, "n={n} x={x}: {f} vs {q}");
            }
        }
    }

    #[test]
    fn at_zero() {
        let mut out = [0.0; 9];
        boys(8, 0.0, &mut out);
        for (n, v) in out.iter().enumerate() {
            assert_eq!(*v, 1.0 / (2 * n + 1) as f64);
        }
    }

    #[test]
    fn continuous_at_switch() {
        // Series just below 35 vs asymptotic branch at 35, corrected by the slope
        // dF_n/dx = -F_{n+1}.
        let (mut a, mut b) = ([0.0; 10], [0.0; 10]);
        let dx = 1e-9;
        boys(9, 35.0 - dx, &mut a);
        boys(9, 35.0, &mut b);
        for n in 0..9 {
            let predicted = b[n] + dx * b[n + 1];
            assert!((a[n] - predicted).abs() < 1e-14 * b[n], "n={n}: {} {predicted}", a[n]);
        }
    }
}

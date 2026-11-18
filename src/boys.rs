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

/// Grid spacing of the table of [`boys_tabulated`].
const TABLE_STEP: f64 = 0.05;
/// Number of Taylor terms of [`boys_tabulated`] (error < (Δ/2)^7/7! ≈ 1e-15 relative).
const TAYLOR_TERMS: usize = 7;
/// Highest order served by the table; larger `nmax` falls back to [`boys`].
pub const TABULATED_NMAX: usize = 16;
const TABLE_X_MAX: f64 = 35.0;

/// F_n(x_i) for x_i = i Δ < 35 and n <= TABULATED_NMAX + TAYLOR_TERMS - 1, row-major by i.
fn table() -> &'static [f64] {
    static TABLE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let width = TABULATED_NMAX + TAYLOR_TERMS;
        let npts = (TABLE_X_MAX / TABLE_STEP) as usize + 2;
        let mut t = vec![0.0; npts * width];
        for (i, row) in t.chunks_mut(width).enumerate() {
            boys(width - 1, i as f64 * TABLE_STEP, row);
        }
        t
    })
}

/// Same as [`boys`] (to ~1e-15 relative), faster: below x = 35 the highest order is
/// a 7-term Taylor expansion dF_n/dx = -F_{n+1} around the nearest point of a table
/// with spacing 0.05 (built once with [`boys`]), the rest follow by the same downward
/// recursion (theory.typ eq:boys-down). Used by the integral-direct J/K builds, which
/// evaluate the Boys function in every SCF iteration.
pub fn boys_tabulated(nmax: usize, x: f64, out: &mut [f64]) {
    if nmax > TABULATED_NMAX || x >= TABLE_X_MAX {
        boys(nmax, x, out);
        return;
    }
    debug_assert!(out.len() > nmax);
    debug_assert!(x >= 0.0);
    let width = TABULATED_NMAX + TAYLOR_TERMS;
    let i = (x / TABLE_STEP + 0.5) as usize;
    let d = x - i as f64 * TABLE_STEP;
    let row = &table()[i * width + nmax..i * width + nmax + TAYLOR_TERMS];
    // Σ_k F_{n+k}(x_i) (-d)^k / k!, Horner form.
    let mut f = row[TAYLOR_TERMS - 1];
    for k in (1..TAYLOR_TERMS).rev() {
        f = row[k - 1] - d * f / k as f64;
    }
    out[nmax] = f;
    if nmax > 0 {
        let ex = (-x).exp();
        for n in (1..=nmax).rev() {
            out[n - 1] = (2.0 * x * out[n] + ex) / (2 * n - 1) as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabulated_matches_series() {
        let (mut a, mut b) = ([0.0; 17], [0.0; 17]);
        for nmax in [0, 1, 4, 8, 16] {
            for k in 0..=7200 {
                let x = k as f64 * 0.00500037;
                boys(nmax, x, &mut a);
                boys_tabulated(nmax, x, &mut b);
                for n in 0..=nmax {
                    assert!((a[n] - b[n]).abs() <= 1e-14 * a[n], "n={n} x={x}: {} vs {}", a[n], b[n]);
                }
            }
        }
    }

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

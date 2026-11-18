//! Cartesian Hessian by central differences of an analytic gradient
//! (theory.typ sec:freq, eq:hess-fd and eq:hess-sym). DERIVED
//!
//! [`central_difference_hessian`] is generic over a function returning the gradient
//! at a coordinate vector, so it can be tested on model functions; the molecular
//! driver is `Calculation::hessian`.

use crate::error::{QcError, Result};
use crate::linalg::Matrix;

/// Default displacement h of theory.typ eq:hess-fd (Bohr). DERIVED
pub const DEFAULT_HESSIAN_STEP: f64 = 5e-3;

/// Result of [`central_difference_hessian`].
#[derive(Debug, Clone, PartialEq)]
pub struct FiniteDifferenceHessian {
    /// The symmetrised Hessian (theory.typ eq:hess-sym), n × n. DERIVED
    pub hessian: Matrix,
    /// δ_asym = max_ij |H_ij − H_ji| of the Hessian before symmetrisation.
    pub max_asymmetry: f64,
    /// Number of gradient evaluations (2 n).
    pub n_gradient_evaluations: usize,
}

/// Hessian of the function whose gradient is `grad`, at `x0`, by central differences
/// with displacement `step` (theory.typ eq:hess-fd), symmetrised by eq:hess-sym. DERIVED
///
/// `grad` is called 2 n times (n = `x0.len()`) in a fixed order: for each coordinate
/// j = 0, 1, …, first at `x0 + step e_j`, then at `x0 − step e_j`. A caller can
/// therefore collect other properties of the displaced points (e.g. dipole moments)
/// inside `grad` and match them to the displacements by this order. It must return a
/// vector of length n. `step` must be finite and positive.
pub fn central_difference_hessian(
    x0: &[f64],
    step: f64,
    mut grad: impl FnMut(&[f64]) -> Result<Vec<f64>>,
) -> Result<FiniteDifferenceHessian> {
    if !(step.is_finite() && step > 0.0) {
        return Err(QcError::InvalidInput(format!("Hessian displacement must be finite and positive, got {step}")));
    }
    let n = x0.len();
    let mut raw = Matrix::zeros(n, n);
    let mut x = x0.to_vec();
    let mut evaluate = |x: &[f64]| -> Result<Vec<f64>> {
        let g = grad(x)?;
        if g.len() != n {
            return Err(QcError::InvalidInput(format!("gradient has length {}, expected {n}", g.len())));
        }
        Ok(g)
    };
    for j in 0..n {
        x[j] = x0[j] + step;
        let g_plus = evaluate(&x)?;
        x[j] = x0[j] - step;
        let g_minus = evaluate(&x)?;
        x[j] = x0[j];
        // eq:hess-fd: column j is (g(x + h e_j) − g(x − h e_j)) / 2h.
        for i in 0..n {
            raw[(i, j)] = (g_plus[i] - g_minus[i]) / (2.0 * step);
        }
    }
    // eq:hess-sym: H <- (H + Hᵀ)/2, δ_asym = max |H_ij − H_ji|. DERIVED
    let max_asymmetry = (&raw - &raw.transpose()).max_abs();
    let hessian = Matrix::from_fn(n, n, |i, j| 0.5 * (raw[(i, j)] + raw[(j, i)]));
    Ok(FiniteDifferenceHessian { hessian, max_asymmetry, n_gradient_evaluations: 2 * n })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quadratic E = ½ xᵀ A x has gradient A x, so central differences are exact
    /// (up to rounding) for any step.
    #[test]
    fn quadratic_is_exact() {
        let a = Matrix::from_vec(3, 3, vec![2.0, 0.3, -0.1, 0.3, 1.5, 0.4, -0.1, 0.4, 0.7]).unwrap();
        let x0 = [0.2, -0.4, 1.1];
        let r = central_difference_hessian(&x0, 0.1, |x| Ok((0..3).map(|i| (0..3).map(|k| a[(i, k)] * x[k]).sum()).collect()))
            .unwrap();
        assert!((&r.hessian - &a).max_abs() < 1e-13, "{:?}", r.hessian);
        assert!(r.max_asymmetry < 1e-13);
        assert_eq!(r.n_gradient_evaluations, 6);
    }

    /// E = x⁴/4 + x² y: the error of the central difference is O(h²), so halving h
    /// divides it by 4.
    #[test]
    fn quartic_error_is_second_order() {
        let grad = |x: &[f64]| Ok(vec![x[0].powi(3) + 2.0 * x[0] * x[1], x[0] * x[0]]);
        let x0 = [0.7, -0.3];
        let exact = Matrix::from_vec(2, 2, vec![3.0 * 0.49 + 2.0 * -0.3, 1.4, 1.4, 0.0]).unwrap();
        let err = |h: f64| (&central_difference_hessian(&x0, h, grad).unwrap().hessian - &exact).max_abs();
        let (e1, e2) = (err(1e-2), err(5e-3));
        // d³g_x/dx³ = 6, so the error of H_xx is h²/6 · 6 = h² exactly.
        assert!((e1 - 1e-4).abs() < 1e-10, "{e1:e}");
        assert!((e1 / e2 - 4.0).abs() < 1e-4, "{e1:e} {e2:e}");
    }

    /// A non-symmetric "Hessian" (gradient not of a scalar function) is symmetrised
    /// and its asymmetry reported.
    #[test]
    fn asymmetry_is_measured() {
        let r = central_difference_hessian(&[0.0, 0.0], 1e-3, |x| Ok(vec![x[0] + 0.25 * x[1], 0.75 * x[0] + x[1]]))
            .unwrap();
        assert!((r.max_asymmetry - 0.5).abs() < 1e-12, "{}", r.max_asymmetry);
        assert!((r.hessian[(0, 1)] - 0.5).abs() < 1e-12 && (r.hessian[(1, 0)] - 0.5).abs() < 1e-12);
    }

    /// The gradient is evaluated at +h then −h for each coordinate in turn.
    #[test]
    fn displacement_order() {
        let mut seen = Vec::new();
        central_difference_hessian(&[1.0, 2.0], 0.5, |x| {
            seen.push(x.to_vec());
            Ok(vec![0.0, 0.0])
        })
        .unwrap();
        assert_eq!(seen, vec![vec![1.5, 2.0], vec![0.5, 2.0], vec![1.0, 2.5], vec![1.0, 1.5]]);
    }

    #[test]
    fn invalid_input() {
        for h in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            assert!(matches!(central_difference_hessian(&[0.0], h, |_| Ok(vec![0.0])), Err(QcError::InvalidInput(_))));
        }
        assert!(matches!(central_difference_hessian(&[0.0], 1e-3, |_| Ok(vec![0.0, 0.0])), Err(QcError::InvalidInput(_))));
    }
}

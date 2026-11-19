//! Geometry optimization: quasi-Newton BFGS with a trust radius in Cartesian
//! coordinates (theory.typ section 11, eq:bfgs). UNCHECKED(Fletcher1970)
//!
//! [`bfgs_minimize`] is generic over a function returning (energy, gradient) at a
//! coordinate vector, so it can be tested on model functions; the molecular driver is
//! `Calculation::optimize`.

use crate::error::{QcError, Result};
use crate::linalg::{Matrix, solve};

/// Settings of the optimizer. Defaults are those of theory.typ section 11.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerSettings {
    /// Convergence: max |g_i| (Hartree/Bohr).
    pub max_gradient: f64,
    /// Convergence: rms(g).
    pub rms_gradient: f64,
    /// Convergence: max |s_i| of the last accepted step (Bohr).
    pub max_step: f64,
    /// Convergence: rms(s).
    pub rms_step: f64,
    /// Maximum number of accepted steps.
    pub max_steps: usize,
    /// Initial trust radius Δ (Bohr).
    pub initial_trust_radius: f64,
    /// Upper bound for Δ when it grows after untruncated steps.
    pub max_trust_radius: f64,
    /// The optimization fails when Δ falls below this.
    pub min_trust_radius: f64,
    /// Diagonal of the initial Hessian H_0 (Hartree/Bohr²).
    pub initial_hessian: f64,
    /// A step is rejected if E_new > E + this.
    pub energy_rise_tolerance: f64,
    /// If true, running out of steps is an error ([`QcError::OptimizationNotConverged`]);
    /// otherwise the result is returned with `converged == false`.
    pub error_if_not_converged: bool,
}

impl Default for OptimizerSettings {
    fn default() -> Self {
        OptimizerSettings {
            max_gradient: 4.5e-4,
            rms_gradient: 3e-4,
            max_step: 1.8e-3,
            rms_step: 1.2e-3,
            max_steps: 100,
            initial_trust_radius: 0.3,
            max_trust_radius: 0.5,
            min_trust_radius: 1e-5,
            initial_hessian: 0.5,
            energy_rise_tolerance: 1e-8,
            error_if_not_converged: true,
        }
    }
}

impl OptimizerSettings {
    /// All four thresholds 30× tighter than the defaults (max|g| < 1.5e-5 Hartree/Bohr),
    /// e.g. for comparing geometries with other programs.
    pub fn tight() -> Self {
        let d = Self::default();
        OptimizerSettings {
            max_gradient: d.max_gradient / 30.0,
            rms_gradient: d.rms_gradient / 30.0,
            max_step: d.max_step / 30.0,
            rms_step: d.rms_step / 30.0,
            ..d
        }
    }
}

/// One energy/gradient evaluation of the optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationStep {
    /// Number of accepted steps before this evaluation (0 for the starting point).
    pub step: usize,
    /// Energy at the evaluated point.
    pub energy: f64,
    /// max |g_i| at the evaluated point.
    pub max_gradient: f64,
    /// rms(g) at the evaluated point.
    pub rms_gradient: f64,
    /// max |s_i| of the step that led here (0 for the starting point).
    pub max_step: f64,
    /// rms(s) of that step.
    pub rms_step: f64,
    /// Trust radius used for that step.
    pub trust_radius: f64,
    /// Whether the point was accepted (false: energy rose, step rejected).
    pub accepted: bool,
}

/// Outcome of [`bfgs_minimize`].
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationResult {
    /// Final coordinates.
    pub coordinates: Vec<f64>,
    /// Energy at the final coordinates.
    pub energy: f64,
    /// Gradient at the final coordinates.
    pub gradient: Vec<f64>,
    /// Number of accepted steps.
    pub steps: usize,
    /// Number of energy/gradient evaluations (including the start and rejected steps).
    pub evaluations: usize,
    /// Whether all convergence criteria were met.
    pub converged: bool,
    /// Every evaluation in order.
    pub history: Vec<OptimizationStep>,
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Minimizes `f` (returning energy and gradient) from `x0` with the BFGS procedure
/// of theory.typ section 11:
///
/// 1. H₀ = `initial_hessian`·I, Δ = `initial_trust_radius`.
/// 2. s = −H⁻¹g (linear solve), scaled down to length Δ if longer.
/// 3. Evaluate E, g at x + s.
/// 4. If E rises by more than `energy_rise_tolerance`, reject the step, set
///    Δ ← min(Δ, |s|)/2 and go back to 2 with the same g, H; fail with
///    [`QcError::TrustRadiusCollapsed`] once Δ < `min_trust_radius`.
/// 5. Otherwise accept, update H by eq:bfgs (skipped if yᵀs ≤ 1e-10) and, if the step
///    was not truncated, set Δ ← min(2Δ, `max_trust_radius`).
/// 6. Converged when max|g|, rms(g) (new gradient) and max|s|, rms(s) (accepted step)
///    are all below their thresholds.
///
/// Errors from `f` are passed through.
pub fn bfgs_minimize<F>(x0: &[f64], settings: &OptimizerSettings, mut f: F) -> Result<OptimizationResult>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>)>,
{
    let n = x0.len();
    if n == 0 {
        return Err(QcError::InvalidInput("nothing to optimize".into()));
    }
    let mut x = x0.to_vec();
    let (mut e, mut g) = f(&x)?;
    if g.len() != n {
        return Err(QcError::InvalidInput("gradient length differs from coordinate length".into()));
    }
    let mut history = vec![OptimizationStep {
        step: 0,
        energy: e,
        max_gradient: max_abs(&g),
        rms_gradient: rms(&g),
        max_step: 0.0,
        rms_step: 0.0,
        trust_radius: settings.initial_trust_radius,
        accepted: true,
    }];
    let mut evaluations = 1;
    let mut h = Matrix::identity(n).scaled(settings.initial_hessian);
    let mut delta = settings.initial_trust_radius;
    let mut steps = 0;
    let finish = |x: Vec<f64>, e: f64, g: Vec<f64>, steps, evaluations, converged, history| OptimizationResult {
        coordinates: x,
        energy: e,
        gradient: g,
        steps,
        evaluations,
        converged,
        history,
    };
    loop {
        if steps >= settings.max_steps {
            if settings.error_if_not_converged {
                return Err(QcError::OptimizationNotConverged { steps, energy: e, max_gradient: max_abs(&g) });
            }
            return Ok(finish(x, e, g, steps, evaluations, false, history));
        }
        // Step 2: s = -H⁻¹ g, truncated to the trust radius.
        let mut s: Vec<f64> = solve(&h, &g)?.into_iter().map(|v| -v).collect();
        let len = norm(&s);
        let truncated = len > delta;
        if truncated {
            for v in &mut s {
                *v *= delta / len;
            }
        }
        // Step 3.
        let xn: Vec<f64> = x.iter().zip(&s).map(|(a, b)| a + b).collect();
        let (en, gn) = f(&xn)?;
        evaluations += 1;
        if gn.len() != n {
            return Err(QcError::InvalidInput("gradient length differs from coordinate length".into()));
        }
        let record = |accepted: bool, step: usize| OptimizationStep {
            step,
            energy: en,
            max_gradient: max_abs(&gn),
            rms_gradient: rms(&gn),
            max_step: max_abs(&s),
            rms_step: rms(&s),
            trust_radius: delta,
            accepted,
        };
        // Step 4: reject energy rises.
        if en > e + settings.energy_rise_tolerance {
            history.push(record(false, steps));
            delta = delta.min(norm(&s)) / 2.0;
            if delta < settings.min_trust_radius {
                return Err(QcError::TrustRadiusCollapsed { steps, trust_radius: delta });
            }
            continue;
        }
        // Step 5: accept and update H (theory.typ eq:bfgs).
        steps += 1;
        history.push(record(true, steps));
        let y: Vec<f64> = gn.iter().zip(&g).map(|(a, b)| a - b).collect();
        let ys: f64 = y.iter().zip(&s).map(|(a, b)| a * b).sum();
        if ys > 1e-10 {
            let hs: Vec<f64> = (0..n).map(|i| (0..n).map(|j| h[(i, j)] * s[j]).sum()).collect();
            let shs: f64 = s.iter().zip(&hs).map(|(a, b)| a * b).sum();
            for i in 0..n {
                for j in 0..n {
                    h[(i, j)] += y[i] * y[j] / ys - hs[i] * hs[j] / shs;
                }
            }
        }
        if !truncated {
            delta = (2.0 * delta).min(settings.max_trust_radius);
        }
        // Step 6: convergence test on the new gradient and the accepted step.
        let converged = max_abs(&gn) < settings.max_gradient
            && rms(&gn) < settings.rms_gradient
            && max_abs(&s) < settings.max_step
            && rms(&s) < settings.rms_step;
        x = xn;
        e = en;
        g = gn;
        if converged {
            return Ok(finish(x, e, g, steps, evaluations, true, history));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_converges_to_minimum() {
        // E = ½ (x-c)ᵀ A (x-c) with an anisotropic positive-definite A.
        let a = [[2.0, 0.3, 0.0], [0.3, 0.8, -0.2], [0.0, -0.2, 0.1]];
        let c = [0.4, -1.0, 2.0];
        let f = |x: &[f64]| {
            let d: Vec<f64> = (0..3).map(|i| x[i] - c[i]).collect();
            let g: Vec<f64> = (0..3).map(|i| (0..3).map(|j| a[i][j] * d[j]).sum()).collect();
            let e = 0.5 * d.iter().zip(&g).map(|(p, q)| p * q).sum::<f64>();
            Ok((e, g))
        };
        let res = bfgs_minimize(&[0.0, 0.0, 0.0], &OptimizerSettings::tight(), f).unwrap();
        assert!(res.converged);
        for (x, ci) in res.coordinates.iter().zip(c) {
            assert!((x - ci).abs() < 1e-3, "{:?}", res.coordinates);
        }
        assert!(res.history.windows(2).all(|w| !w[1].accepted || w[1].energy <= w[0].energy + 1e-8));
    }

    #[test]
    fn rosenbrock_converges() {
        // Scaled Rosenbrock function, minimum at (1, 1); needs rejected steps and BFGS curvature.
        let f = |x: &[f64]| {
            let (a, b) = (x[0], x[1]);
            let e = 0.01 * ((1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2));
            let g = vec![0.01 * (-2.0 * (1.0 - a) - 400.0 * a * (b - a * a)), 0.01 * 200.0 * (b - a * a)];
            Ok((e, g))
        };
        let settings = OptimizerSettings { max_steps: 500, ..OptimizerSettings::tight() };
        let res = bfgs_minimize(&[-1.2, 1.0], &settings, f).unwrap();
        println!("Rosenbrock: {} steps, {} evaluations, x = {:?}", res.steps, res.evaluations, res.coordinates);
        assert!(res.converged);
        assert!((res.coordinates[0] - 1.0).abs() < 1e-3 && (res.coordinates[1] - 1.0).abs() < 2e-3);
    }

    #[test]
    fn step_limit_is_reported() {
        let f = |x: &[f64]| Ok((x[0] * x[0] * 0.01, vec![0.02 * x[0]]));
        let settings = OptimizerSettings { max_steps: 2, ..OptimizerSettings::tight() };
        let err = bfgs_minimize(&[10.0], &settings, f).unwrap_err();
        assert!(matches!(err, QcError::OptimizationNotConverged { steps: 2, .. }));
        let settings = OptimizerSettings { error_if_not_converged: false, ..settings };
        let res = bfgs_minimize(&[10.0], &settings, f).unwrap();
        assert!(!res.converged);
        assert_eq!(res.steps, 2);
    }

    #[test]
    fn inconsistent_gradient_collapses_trust_radius() {
        // The "gradient" points uphill, so every step raises the energy.
        let f = |x: &[f64]| Ok((x[0] * x[0], vec![-2.0 * x[0]]));
        let err = bfgs_minimize(&[1.0], &OptimizerSettings::default(), f).unwrap_err();
        assert!(matches!(err, QcError::TrustRadiusCollapsed { .. }));
    }
}

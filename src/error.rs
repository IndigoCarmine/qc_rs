//! Crate-wide error type.

use std::fmt;

/// Errors returned by `qc_rs`.
#[derive(Debug, Clone, PartialEq)]
pub enum QcError {
    /// Malformed basis-set file or other text input.
    Parse(String),
    /// Element symbol or atomic number outside the supported range (H–Ar).
    UnknownElement(String),
    /// The basis set has no entry for this element.
    MissingBasis { basis: String, element: String },
    /// Shell angular momentum not supported for basis shells.
    UnsupportedAngularMomentum(usize),
    /// Invalid argument (dimension mismatch, odd electron count, coincident atoms, ...).
    InvalidInput(String),
    /// The overlap matrix is (numerically) singular; carries its smallest eigenvalue.
    LinearDependence(f64),
    /// An iterative eigensolver did not converge.
    EigenNotConverged,
    /// A linear system was singular.
    SingularMatrix,
    /// The SCF did not converge within the iteration limit.
    ScfNotConverged { iterations: usize, delta_e: f64, max_error: f64 },
    /// The geometry optimization reached its step limit before converging.
    OptimizationNotConverged { steps: usize, energy: f64, max_gradient: f64 },
    /// Repeated energy rises shrank the optimizer's trust radius below its minimum.
    TrustRadiusCollapsed { steps: usize, trust_radius: f64 },
}

impl fmt::Display for QcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QcError::Parse(s) => write!(f, "parse error: {s}"),
            QcError::UnknownElement(s) => write!(f, "unknown or unsupported element: {s}"),
            QcError::MissingBasis { basis, element } => {
                write!(f, "basis set '{basis}' has no functions for element {element}")
            }
            QcError::UnsupportedAngularMomentum(l) => {
                write!(f, "unsupported shell angular momentum L = {l}")
            }
            QcError::InvalidInput(s) => write!(f, "invalid input: {s}"),
            QcError::LinearDependence(s) => {
                write!(f, "basis is linearly dependent (smallest overlap eigenvalue {s:e})")
            }
            QcError::EigenNotConverged => write!(f, "Jacobi eigensolver did not converge"),
            QcError::SingularMatrix => write!(f, "singular linear system"),
            QcError::ScfNotConverged { iterations, delta_e, max_error } => write!(
                f,
                "SCF not converged after {iterations} iterations (|dE| = {delta_e:e}, max|e| = {max_error:e})"
            ),
            QcError::OptimizationNotConverged { steps, energy, max_gradient } => write!(
                f,
                "geometry optimization not converged after {steps} steps (E = {energy:.10}, max|g| = {max_gradient:e})"
            ),
            QcError::TrustRadiusCollapsed { steps, trust_radius } => write!(
                f,
                "geometry optimization failed after {steps} steps: energy keeps rising, trust radius {trust_radius:e} Bohr"
            ),
        }
    }
}

impl std::error::Error for QcError {}

/// Result alias using [`QcError`].
pub type Result<T> = std::result::Result<T, QcError>;

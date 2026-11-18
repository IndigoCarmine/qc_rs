//! `qc_rs`: closed-shell Hartree–Fock and Kohn–Sham DFT with Gaussian basis sets:
//! single-point energies, analytic nuclear gradients and geometry optimization.
//!
//! The mathematics follows `docs/theory.typ`; comments cite its equation labels.
//!
//! High-level use goes through [`Calculation`]:
//!
//! ```no_run
//! use qc_rs::{Calculation, Molecule, OptimizerSettings, Unit};
//! let mol = Molecule::from_symbols(&["H", "H"], &[[0.0; 3], [0.0, 0.0, 0.74]], Unit::Angstrom, 0)?;
//! let calc = Calculation::new(&mol, "hf", "sto-3g")?;
//! println!("E = {}", calc.energy()?.energy);
//! let opt = calc.optimize(&OptimizerSettings::default())?;
//! println!("E_opt = {}", opt.energy);
//! # Ok::<(), qc_rs::QcError>(())
//! ```
//!
//! The lower-level pieces ([`run_rhf`], [`run_rks`], [`gradient::rhf_gradient`], …)
//! remain available.

/// Compiles and runs the code blocks of README.md as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod ao_eval;
pub mod basis;
pub mod boys;
pub mod calculation;
pub mod dft;
pub mod dual;
pub mod error;
pub mod functional;
pub mod gradient;
pub mod grid;
pub mod hermite;
pub mod integrals;
pub mod linalg;
pub mod molecule;
pub mod optimize;
pub mod properties;
pub mod scf;

pub use basis::{Basis, BasisOptions, BasisSetDefinition};
pub use calculation::{Calculation, GeometryOptimization, GradientResult, SinglePoint};
pub use dft::{RksResult, XcIntegrator, run_rks};
pub use error::{QcError, Result};
pub use functional::{Functional, XcComponent};
pub use gradient::{GradientComponents, rhf_gradient, rks_gradient};
pub use grid::{GridSettings, MolecularGrid};
pub use optimize::{OptimizationResult, OptimizationStep, OptimizerSettings, bfgs_minimize};
pub use linalg::Matrix;
pub use molecule::{Atom, Molecule, Unit};
pub use properties::dipole_moment;
pub use scf::{EnergyComponents, ExchangeCorrelation, HartreeFock, ScfOptions, ScfResult, XcTerm, run_rhf, run_scf};

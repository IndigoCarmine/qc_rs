//! High-level driver: a molecule, a method ("hf" or a functional name) and a basis
//! set name, with single-point energy, analytic gradient and geometry optimization.
//!
//! ```no_run
//! use qc_rs::{Calculation, Molecule, OptimizerSettings, Unit};
//! let mol = Molecule::from_symbols(
//!     &["O", "H", "H"],
//!     &[[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
//!     Unit::Angstrom,
//!     0,
//! )?;
//! let calc = Calculation::new(&mol, "b3lyp", "6-31g")?;
//! println!("E = {}", calc.energy()?.energy);
//! println!("dE/dR = {:?}", calc.gradient()?.gradient);
//! let opt = calc.optimize(&OptimizerSettings::default())?;
//! println!("E_opt = {} after {} steps", opt.energy, opt.steps);
//! # Ok::<(), qc_rs::QcError>(())
//! ```

use crate::basis::{Basis, BasisOptions, BasisSetDefinition};
use crate::dft::{XcIntegrator, XcOptions};
use crate::error::Result;
use crate::functional::Functional;
use crate::gradient::{GradientComponents, rks_gradient};
use crate::grid::{GridSettings, MolecularGrid};
use crate::linalg::Matrix;
use crate::molecule::Molecule;
use crate::optimize::{OptimizationStep, OptimizerSettings, bfgs_minimize};
use crate::scf::{ScfOptions, ScfResult, run_scf};

/// SCF convergence (max |e| of the DIIS commutator) used for gradients and
/// optimizations when the user's [`ScfOptions::error_tolerance`] is looser.
///
/// The analytic gradient assumes a stationary SCF solution, so its error is first
/// order in the remaining density error: the default 1e-7 leaves errors of ~1e-7
/// Hartree/Bohr, 1e-10 removes them for a few extra SCF iterations.
pub const GRADIENT_SCF_ERROR_TOLERANCE: f64 = 1e-10;

/// A closed-shell RHF or RKS calculation setup.
#[derive(Debug, Clone)]
pub struct Calculation {
    molecule: Molecule,
    functional: Functional,
    basis_set: BasisSetDefinition,
    basis_options: BasisOptions,
    grid: GridSettings,
    scf_options: ScfOptions,
}

/// Result of [`Calculation::energy`].
#[derive(Debug, Clone)]
pub struct SinglePoint {
    /// Total energy (Hartree).
    pub energy: f64,
    /// The converged SCF (components, orbitals, density).
    pub scf: ScfResult,
    /// Σ_g w_g ρ_g (DFT only).
    pub electron_count: Option<f64>,
    /// Number of grid points (0 for HF).
    pub n_grid_points: usize,
}

/// Result of [`Calculation::gradient`].
#[derive(Debug, Clone)]
pub struct GradientResult {
    /// Total energy (Hartree).
    pub energy: f64,
    /// dE/dR per atom (Hartree/Bohr).
    pub gradient: Vec<[f64; 3]>,
    /// The individual terms of theory.typ eq:grad.
    pub components: GradientComponents,
    /// The SCF the gradient was computed from.
    pub scf: ScfResult,
}

/// Result of [`Calculation::optimize`].
#[derive(Debug, Clone)]
pub struct GeometryOptimization {
    /// Final geometry.
    pub molecule: Molecule,
    /// Energy at the final geometry.
    pub energy: f64,
    /// Gradient at the final geometry (Hartree/Bohr).
    pub gradient: Vec<[f64; 3]>,
    /// Number of accepted optimization steps.
    pub steps: usize,
    /// Number of energy+gradient evaluations.
    pub evaluations: usize,
    /// Whether the convergence criteria were met (always true unless
    /// [`OptimizerSettings::error_if_not_converged`] is false).
    pub converged: bool,
    /// Per-evaluation history.
    pub history: Vec<OptimizationStep>,
}

impl Calculation {
    /// `method` is "hf" or a functional name accepted by [`Functional::from_name`]
    /// (e.g. "b3lyp", "blyp", "svwn5"); `basis` is a built-in basis set name
    /// ("sto-3g", "6-31g", "6-31g*", "6-31g**"). Spherical d functions, the default DFT
    /// grid of theory.typ section 8.4 and default SCF options.
    pub fn new(molecule: &Molecule, method: &str, basis: &str) -> Result<Self> {
        let calc = Calculation {
            molecule: molecule.clone(),
            functional: Functional::from_name(method)?,
            basis_set: BasisSetDefinition::builtin(basis)?,
            basis_options: BasisOptions::default(),
            grid: GridSettings::default(),
            scf_options: ScfOptions::default(),
        };
        calc.basis_for(molecule)?; // fail early on missing elements
        Ok(calc)
    }

    /// Uses 6 Cartesian instead of 5 spherical d functions.
    pub fn with_cartesian_d(mut self, cartesian: bool) -> Self {
        self.basis_options.cartesian_d = cartesian;
        self
    }

    /// DFT integration grid (ignored for HF).
    pub fn with_grid(mut self, grid: GridSettings) -> Self {
        self.grid = grid;
        self
    }

    /// SCF options. For gradients and optimizations the commutator threshold is
    /// tightened to [`GRADIENT_SCF_ERROR_TOLERANCE`] if it is looser.
    pub fn with_scf_options(mut self, options: ScfOptions) -> Self {
        self.scf_options = options;
        self
    }

    /// Uses a different geometry with the same settings.
    pub fn with_molecule(mut self, molecule: &Molecule) -> Self {
        self.molecule = molecule.clone();
        self
    }

    /// The molecule.
    pub fn molecule(&self) -> &Molecule {
        &self.molecule
    }

    /// The functional (HF is the functional with f = 0, a_HF = 1).
    pub fn functional(&self) -> &Functional {
        &self.functional
    }

    /// The AO basis of `mol`.
    pub fn basis_for(&self, mol: &Molecule) -> Result<Basis> {
        Basis::new(mol, &self.basis_set, self.basis_options)
    }

    fn gradient_scf_options(&self) -> ScfOptions {
        let mut o = self.scf_options.clone();
        o.error_tolerance = o.error_tolerance.min(GRADIENT_SCF_ERROR_TOLERANCE);
        o
    }

    /// SCF (and optionally gradient) at geometry `mol`.
    fn run(
        &self,
        mol: &Molecule,
        options: &ScfOptions,
        gradient: bool,
    ) -> Result<(ScfResult, XcIntegrator, Option<GradientComponents>)> {
        let basis = self.basis_for(mol)?;
        let grid = if self.functional.has_density_part() {
            MolecularGrid::new(mol, self.grid)?
        } else {
            MolecularGrid { points: vec![], weights: vec![], blocks: vec![], settings: self.grid }
        };
        let mut xc = XcIntegrator::with_grid(&basis, &self.functional, grid, XcOptions::default())?;
        let scf = run_scf(mol, &basis, &mut xc, options)?;
        let grad = if gradient { Some(rks_gradient(mol, &basis, &scf, &xc)?) } else { None };
        Ok((scf, xc, grad))
    }

    /// Single-point energy.
    pub fn energy(&self) -> Result<SinglePoint> {
        let (scf, xc, _) = self.run(&self.molecule, &self.scf_options, false)?;
        Ok(SinglePoint {
            energy: scf.energy,
            electron_count: xc.last_electron_count(),
            n_grid_points: xc.grid().len(),
            scf,
        })
    }

    /// Energy and analytic gradient (theory.typ eq:grad).
    pub fn gradient(&self) -> Result<GradientResult> {
        let (scf, _, grad) = self.run(&self.molecule, &self.gradient_scf_options(), true)?;
        let components = grad.expect("gradient requested");
        Ok(GradientResult { energy: scf.energy, gradient: components.total(), components, scf })
    }

    /// Geometry optimization (theory.typ section 11) starting from this calculation's
    /// molecule. The SCF at each new geometry starts from the previous density.
    pub fn optimize(&self, settings: &OptimizerSettings) -> Result<GeometryOptimization> {
        let base = self.gradient_scf_options();
        let mut last_density: Option<Matrix> = base.initial_density.clone();
        let res = bfgs_minimize(&self.molecule.coordinates(), settings, |x| {
            let mol = self.molecule.with_coordinates(x)?;
            let opts = ScfOptions { initial_density: last_density.clone(), ..base.clone() };
            let (scf, _, grad) = self.run(&mol, &opts, true)?;
            last_density = Some(scf.density);
            let g = grad.expect("gradient requested").total();
            Ok((scf.energy, g.iter().flatten().copied().collect()))
        })?;
        Ok(GeometryOptimization {
            molecule: self.molecule.with_coordinates(&res.coordinates)?,
            energy: res.energy,
            gradient: res.gradient.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
            steps: res.steps,
            evaluations: res.evaluations,
            converged: res.converged,
            history: res.history,
        })
    }
}

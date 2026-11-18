//! High-level driver: a molecule, a method ("hf" or a functional name) and a basis
//! set name, with single-point energy, analytic gradient, geometry optimization and
//! harmonic frequencies with thermochemistry, optionally with the DFT-D3 dispersion
//! correction.
//!
//! ```no_run
//! use qc_rs::{Calculation, D3Variant, Molecule, OptimizerSettings, Unit};
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
//! // B3LYP-D3(BJ): the dispersion energy is added to energy, gradient and optimization.
//! let d3 = calc.with_dispersion(D3Variant::Bj)?;
//! let e = d3.energy()?;
//! println!("E = {} (E_disp = {:?})", e.energy, e.dispersion_energy);
//! # Ok::<(), qc_rs::QcError>(())
//! ```

use crate::basis::{Basis, BasisOptions, BasisSetDefinition};
use crate::dft::{XcIntegrator, XcOptions};
use crate::dispersion::{D3Parameters, D3Variant, dispersion_energy, dispersion_energy_and_gradient};
use crate::error::{QcError, Result};
use crate::functional::Functional;
use crate::gradient::{GradientComponents, rks_gradient};
use crate::grid::{GridSettings, MolecularGrid};
use crate::hessian::{DEFAULT_HESSIAN_STEP, central_difference_hessian};
use crate::linalg::Matrix;
use crate::molecule::Molecule;
use crate::optimize::{OptimizationStep, OptimizerSettings, bfgs_minimize};
use crate::scf::{IntegralStrategy, ScfOptions, ScfResult, run_scf};
use crate::thermo::{ThermoSettings, Thermochemistry, check_settings as check_thermo_settings, thermochemistry};
use crate::vibration::{HarmonicAnalysis, harmonic_analysis, resolve_masses};

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
    dispersion: Option<D3Parameters>,
}

/// Result of [`Calculation::energy`].
#[derive(Debug, Clone)]
pub struct SinglePoint {
    /// Total energy (Hartree): `scf.energy` plus `dispersion_energy`, if any
    /// (theory.typ eq:d3-total). UNCHECKED(Grimme2010)
    pub energy: f64,
    /// The converged SCF (components, orbitals, density).
    pub scf: ScfResult,
    /// Σ_g w_g ρ_g (DFT only).
    pub electron_count: Option<f64>,
    /// Number of grid points (0 for HF).
    pub n_grid_points: usize,
    /// DFT-D3 dispersion energy (Hartree, theory.typ eq:d3-e). UNCHECKED(Grimme2010)
    /// `None` without a dispersion correction ([`Calculation::with_dispersion`]).
    pub dispersion_energy: Option<f64>,
}

/// Result of [`Calculation::gradient`].
#[derive(Debug, Clone)]
pub struct GradientResult {
    /// Total energy (Hartree), including `dispersion_energy` if any.
    pub energy: f64,
    /// dE/dR per atom (Hartree/Bohr): `components.total()` plus
    /// `dispersion_gradient`, if any (theory.typ eq:d3-total).
    pub gradient: Vec<[f64; 3]>,
    /// The individual terms of theory.typ eq:grad (SCF part only). UNCHECKED(Pulay1969)
    pub components: GradientComponents,
    /// The SCF the gradient was computed from.
    pub scf: ScfResult,
    /// DFT-D3 dispersion energy (Hartree); `None` without a dispersion correction.
    pub dispersion_energy: Option<f64>,
    /// DFT-D3 dispersion gradient per atom (Hartree/Bohr, theory.typ eq:d3-grad). DERIVED
    /// `None` without a dispersion correction.
    pub dispersion_gradient: Option<Vec<[f64; 3]>>,
}

/// Result of [`Calculation::hessian`].
#[derive(Debug, Clone)]
pub struct HessianResult {
    /// Total energy at the reference geometry (Hartree), as [`GradientResult::energy`].
    pub energy: f64,
    /// Total gradient at the reference geometry (Hartree/Bohr), as
    /// [`GradientResult::gradient`]. Frequencies are meaningful only where it is ~0.
    pub gradient: Vec<[f64; 3]>,
    /// Symmetrised Cartesian Hessian of the total energy (Hartree/Bohr²), 3N × 3N,
    /// coordinates ordered atom by atom (x, y, z) like [`Molecule::coordinates`]
    /// (theory.typ eq:hess-fd, eq:hess-sym). DERIVED
    pub hessian: Matrix,
    /// δ_asym = max |H_ij − H_ji| before symmetrisation (Hartree/Bohr²): a measure of
    /// the numerical error of the Hessian.
    pub max_asymmetry: f64,
    /// Displacement h (Bohr).
    pub step: f64,
    /// Number of displaced SCF + gradient evaluations (6N), not counting the reference.
    pub n_gradient_evaluations: usize,
    /// SCF iterations summed over the displaced evaluations.
    pub displaced_scf_iterations: usize,
    /// The SCF at the reference geometry.
    pub scf: ScfResult,
}

/// Settings of [`Calculation::frequencies`].
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencySettings {
    /// Displacement h (Bohr) of the numerical Hessian (theory.typ eq:hess-fd); default
    /// [`DEFAULT_HESSIAN_STEP`].
    pub step: f64,
    /// Atomic masses (u, one per atom); `None` uses [`crate::standard_atomic_mass`]
    /// (theory.typ eq:mass-weight).
    pub masses: Option<Vec<f64>>,
    /// Temperature, pressure and the rotational symmetry number σ (given by the user,
    /// not determined automatically) of the thermochemistry.
    pub thermo: ThermoSettings,
}

impl Default for FrequencySettings {
    fn default() -> Self {
        Self { step: DEFAULT_HESSIAN_STEP, masses: None, thermo: ThermoSettings::default() }
    }
}

/// Result of [`Calculation::frequencies`].
#[derive(Debug, Clone)]
pub struct FrequencyResult {
    /// The numerical Hessian of the total energy, with the energy and gradient at the
    /// reference geometry.
    pub hessian: HessianResult,
    /// Harmonic wavenumbers (cm⁻¹, ascending, imaginary as negative), normal modes and
    /// reduced masses (theory.typ eq:mass-weight … eq:reduced-mass).
    pub analysis: HarmonicAnalysis,
    /// RRHO thermochemistry (theory.typ eq:thermo-total) with E_el = the total energy
    /// at the reference geometry, including the dispersion energy if any.
    pub thermo: Thermochemistry,
    /// max_A,α |dE/dR_Aα| at the reference geometry (Hartree/Bohr). The analysis is
    /// meaningful only at a stationary point, where this is ~0 (theory.typ sec:freq).
    pub max_abs_gradient: f64,
}

/// Result of [`Calculation::optimize`].
#[derive(Debug, Clone)]
pub struct GeometryOptimization {
    /// Final geometry.
    pub molecule: Molecule,
    /// Energy at the final geometry (including the dispersion energy, if any).
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
    /// (e.g. "b3lyp", "blyp", "svwn5", "cam-b3lyp"); `basis` is a built-in basis set name
    /// ("sto-3g", "6-31g", "6-31g*", "6-31g**"). Spherical d functions, the default DFT
    /// grid of theory.typ section 8.4, default SCF options and no dispersion correction.
    pub fn new(molecule: &Molecule, method: &str, basis: &str) -> Result<Self> {
        let calc = Calculation {
            molecule: molecule.clone(),
            functional: Functional::from_name(method)?,
            basis_set: BasisSetDefinition::builtin(basis)?,
            basis_options: BasisOptions::default(),
            grid: GridSettings::default(),
            scf_options: ScfOptions::default(),
            dispersion: None,
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

    /// Stored or integral-direct two-electron integrals (sets
    /// [`ScfOptions::integral_strategy`]; the default is automatic). Gradients and
    /// optimizations work with either.
    pub fn with_integral_strategy(mut self, strategy: IntegralStrategy) -> Self {
        self.scf_options.integral_strategy = strategy;
        self
    }

    /// Adds the DFT-D3 two-body dispersion correction (theory.typ sec:d3) with the
    /// published parameters of this calculation's functional
    /// ([`D3Parameters::for_functional`]: B3LYP, B3LYP5 and CAM-B3LYP) to the energy,
    /// the gradient and the geometry optimization (theory.typ eq:d3-total). Other
    /// methods, including HF, are an error: give parameters explicitly with
    /// [`Calculation::with_dispersion_parameters`]. Elements beyond Ar are an error.
    pub fn with_dispersion(self, variant: D3Variant) -> Result<Self> {
        let params = D3Parameters::for_functional(self.functional.name(), variant)?;
        let calc = self.with_dispersion_parameters(Some(params));
        dispersion_energy(&calc.molecule, &params)?; // fail early on elements beyond Ar
        Ok(calc)
    }

    /// Adds a DFT-D3 correction with explicitly given parameters (see
    /// [`D3Parameters::new`]) whatever the method; `None` removes the correction.
    /// Unlike [`Calculation::with_dispersion`], unsupported elements are reported
    /// only when the energy is computed.
    pub fn with_dispersion_parameters(mut self, params: Option<D3Parameters>) -> Self {
        self.dispersion = params;
        self
    }

    /// The dispersion correction, if any.
    pub fn dispersion(&self) -> Option<&D3Parameters> {
        self.dispersion.as_ref()
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

    /// Dispersion energy and gradient at geometry `mol`, if a correction is set.
    fn dispersion_at(&self, mol: &Molecule) -> Result<Option<(f64, Vec<[f64; 3]>)>> {
        self.dispersion.as_ref().map(|p| dispersion_energy_and_gradient(mol, p)).transpose()
    }

    /// Single-point energy.
    pub fn energy(&self) -> Result<SinglePoint> {
        // Dispersion first: it is cheap and fails on unsupported elements before the SCF.
        let e_disp = self.dispersion.as_ref().map(|p| dispersion_energy(&self.molecule, p)).transpose()?;
        let (scf, xc, _) = self.run(&self.molecule, &self.scf_options, false)?;
        Ok(SinglePoint {
            // eq:d3-total. Without dispersion this is scf.energy unchanged.
            energy: e_disp.map_or(scf.energy, |e| scf.energy + e),
            electron_count: xc.last_electron_count(),
            n_grid_points: xc.grid().len(),
            scf,
            dispersion_energy: e_disp,
        })
    }

    /// Energy and analytic gradient (theory.typ eq:grad, plus eq:d3-grad with a
    /// dispersion correction).
    pub fn gradient(&self) -> Result<GradientResult> {
        let disp = self.dispersion_at(&self.molecule)?; // before the SCF, as in energy()
        let (scf, _, grad) = self.run(&self.molecule, &self.gradient_scf_options(), true)?;
        let components = grad.expect("gradient requested");
        let (energy, gradient) = add_dispersion(scf.energy, components.total(), disp.as_ref());
        let (dispersion_energy, dispersion_gradient) = disp.unzip();
        Ok(GradientResult { energy, gradient, components, scf, dispersion_energy, dispersion_gradient })
    }

    /// Cartesian Hessian of the total energy (including the dispersion correction, if
    /// any) by central differences of the analytic gradient with displacement `step`
    /// in Bohr (theory.typ eq:hess-fd, [`crate::hessian::DEFAULT_HESSIAN_STEP`] is the recommended
    /// value), symmetrised by eq:hess-sym. DERIVED
    ///
    /// The reference point is computed like [`Calculation::gradient`]; each of the 6N
    /// displaced SCFs starts from the reference density and converges to
    /// [`GRADIENT_SCF_ERROR_TOLERANCE`] (or the user's tighter tolerance). The
    /// displacements run one after another.
    pub fn hessian(&self, step: f64) -> Result<HessianResult> {
        if !(step.is_finite() && step > 0.0) {
            // Checked here too so that no SCF runs for an invalid step.
            return Err(QcError::InvalidInput(format!("Hessian displacement must be finite and positive, got {step}")));
        }
        let reference = self.gradient()?;
        let base = ScfOptions { initial_density: Some(reference.scf.density.clone()), ..self.gradient_scf_options() };
        let mut displaced_scf_iterations = 0;
        let fd = central_difference_hessian(&self.molecule.coordinates(), step, |x| {
            let mol = self.molecule.with_coordinates(x)?;
            // The same total gradient as gradient(): SCF part plus eq:d3-grad.
            let disp = self.dispersion_at(&mol)?;
            let (scf, _, grad) = self.run(&mol, &base, true)?;
            displaced_scf_iterations += scf.iterations;
            let (_, g) = add_dispersion(scf.energy, grad.expect("gradient requested").total(), disp.as_ref());
            Ok(g.iter().flatten().copied().collect())
        })?;
        Ok(HessianResult {
            energy: reference.energy,
            gradient: reference.gradient,
            hessian: fd.hessian,
            max_asymmetry: fd.max_asymmetry,
            step,
            n_gradient_evaluations: fd.n_gradient_evaluations,
            displaced_scf_iterations,
            scf: reference.scf,
        })
    }

    /// Harmonic frequencies and thermochemistry (theory.typ sec:freq) at this
    /// calculation's geometry: the Hessian of [`Calculation::hessian`] with
    /// `settings.step` (6N + 1 SCF + gradient evaluations), [`crate::harmonic_analysis`]
    /// with `settings.masses`, and [`crate::thermochemistry`] with `settings.thermo` and
    /// E_el = the total energy (eq:thermo-total, including the dispersion energy if any).
    /// DERIVED
    ///
    /// The geometry should be a stationary point (e.g. from [`Calculation::optimize`]);
    /// check [`FrequencyResult::max_abs_gradient`]. Imaginary frequencies are negative
    /// wavenumbers and are left out of the thermochemistry
    /// ([`Thermochemistry::n_excluded_modes`]). The symmetry number is not determined
    /// automatically. Invalid settings are reported before any SCF runs.
    pub fn frequencies(&self, settings: &FrequencySettings) -> Result<FrequencyResult> {
        resolve_masses(&self.molecule, settings.masses.as_deref())?;
        check_thermo_settings(&settings.thermo)?;
        let hessian = self.hessian(settings.step)?;
        let masses = settings.masses.as_deref();
        let analysis = harmonic_analysis(&self.molecule, &hessian.hessian, masses)?;
        let thermo =
            thermochemistry(&self.molecule, masses, &analysis.frequencies_cm1, hessian.energy, &settings.thermo)?;
        let max_abs_gradient = hessian.gradient.iter().flatten().fold(0.0_f64, |m, g| m.max(g.abs()));
        Ok(FrequencyResult { hessian, analysis, thermo, max_abs_gradient })
    }

    /// Geometry optimization (theory.typ section 11) starting from this calculation's
    /// molecule, of the total energy including the dispersion correction if any
    /// (eq:d3-total). The SCF at each new geometry starts from the previous density.
    pub fn optimize(&self, settings: &OptimizerSettings) -> Result<GeometryOptimization> {
        let base = self.gradient_scf_options();
        let mut last_density: Option<Matrix> = base.initial_density.clone();
        let res = bfgs_minimize(&self.molecule.coordinates(), settings, |x| {
            let mol = self.molecule.with_coordinates(x)?;
            let disp = self.dispersion_at(&mol)?; // before the SCF, as in energy()
            let opts = ScfOptions { initial_density: last_density.clone(), ..base.clone() };
            let (scf, _, grad) = self.run(&mol, &opts, true)?;
            last_density = Some(scf.density);
            let (e, g) = add_dispersion(scf.energy, grad.expect("gradient requested").total(), disp.as_ref());
            Ok((e, g.iter().flatten().copied().collect()))
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

/// theory.typ eq:d3-total: SCF energy and gradient plus the dispersion part, if any.
/// Without dispersion both are returned unchanged.
fn add_dispersion(
    e_scf: f64,
    mut g: Vec<[f64; 3]>,
    disp: Option<&(f64, Vec<[f64; 3]>)>,
) -> (f64, Vec<[f64; 3]>) {
    match disp {
        None => (e_scf, g),
        Some((e_disp, g_disp)) => {
            for (a, b) in g.iter_mut().zip(g_disp) {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            }
            (e_scf + e_disp, g)
        }
    }
}

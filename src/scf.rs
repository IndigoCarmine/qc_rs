//! Closed-shell SCF (RHF now; RKS via a pluggable exchange-correlation term).
//! theory.typ sections 6–7.

use crate::basis::Basis;
use crate::error::{QcError, Result};
use crate::integrals::{EriTensor, core_hamiltonian, overlap};
use crate::linalg::{Matrix, solve, symmetric_eigen};
use crate::molecule::Molecule;
use std::collections::VecDeque;

/// Exchange-correlation contribution evaluated for a given density matrix.
#[derive(Debug, Clone)]
pub struct XcTerm {
    /// E_xc.
    pub energy: f64,
    /// V^xc_μν.
    pub potential: Matrix,
}

/// A model for the exchange-correlation part of the Fock matrix
/// F = h + J - (a_HF/2) K + V^xc (theory.typ eq:fock-rks).
pub trait ExchangeCorrelation {
    /// Fraction a_HF of exact (Hartree–Fock) exchange.
    fn exact_exchange_fraction(&self) -> f64;
    /// E_xc and V^xc for density matrix `density`, or `None` if there is no
    /// density-functional part.
    fn evaluate(&mut self, density: &Matrix) -> Result<Option<XcTerm>>;
}

/// Pure Hartree–Fock: a_HF = 1, no XC functional.
#[derive(Debug, Clone, Copy, Default)]
pub struct HartreeFock;

impl ExchangeCorrelation for HartreeFock {
    fn exact_exchange_fraction(&self) -> f64 {
        1.0
    }
    fn evaluate(&mut self, _density: &Matrix) -> Result<Option<XcTerm>> {
        Ok(None)
    }
}

/// SCF settings (defaults from theory.typ section 6.3).
#[derive(Debug, Clone)]
pub struct ScfOptions {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence threshold on |ΔE| (Hartree).
    pub energy_tolerance: f64,
    /// Convergence threshold on max |e| of the DIIS error (eq:diis-err).
    pub error_tolerance: f64,
    /// Number of Fock/error pairs kept for DIIS.
    pub diis_size: usize,
    /// Schwarz screening threshold for ERIs.
    pub schwarz_threshold: f64,
    /// Smallest allowed overlap eigenvalue.
    pub linear_dependence_threshold: f64,
    /// Starting density matrix (e.g. from a previous geometry); core guess if `None`.
    pub initial_density: Option<Matrix>,
}

impl Default for ScfOptions {
    fn default() -> Self {
        ScfOptions {
            max_iterations: 200,
            energy_tolerance: 1e-10,
            error_tolerance: 1e-7,
            diis_size: 8,
            schwarz_threshold: 1e-12,
            linear_dependence_threshold: 1e-8,
            initial_density: None,
        }
    }
}

/// Energy decomposition (Hartree). theory.typ eq:e-rks
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EnergyComponents {
    /// Σ P h.
    pub one_electron: f64,
    /// ½ Σ P J.
    pub coulomb: f64,
    /// -(a_HF/4) Σ P K.
    pub exchange: f64,
    /// E_xc.
    pub xc: f64,
    /// V_nn.
    pub nuclear_repulsion: f64,
}

impl EnergyComponents {
    /// Total energy.
    pub fn total(&self) -> f64 {
        self.one_electron + self.coulomb + self.exchange + self.xc + self.nuclear_repulsion
    }

    /// Two-electron part (Coulomb + exchange + xc).
    pub fn two_electron(&self) -> f64 {
        self.coulomb + self.exchange + self.xc
    }
}

/// Result of a converged SCF.
#[derive(Debug, Clone)]
pub struct ScfResult {
    /// Total energy.
    pub energy: f64,
    /// Energy components.
    pub components: EnergyComponents,
    /// Orbital energies, ascending.
    pub orbital_energies: Vec<f64>,
    /// MO coefficients (AOs × MOs, columns ordered like `orbital_energies`).
    pub coefficients: Matrix,
    /// Density matrix P = 2 C_occ C_occᵀ built from `coefficients` (eq:density).
    pub density: Matrix,
    /// Number of doubly occupied orbitals.
    pub n_occupied: usize,
    /// Number of SCF iterations performed.
    pub iterations: usize,
    /// |ΔE| at convergence.
    pub final_delta_e: f64,
    /// max |e| of the DIIS error at convergence.
    pub final_error: f64,
}

/// Closed-shell Hartree–Fock energy.
pub fn run_rhf(mol: &Molecule, basis: &Basis, options: &ScfOptions) -> Result<ScfResult> {
    run_scf(mol, basis, &mut HartreeFock, options)
}

/// Integrals and derived matrices shared by all SCF iterations.
struct ScfSystem<'a> {
    h: Matrix,
    s: Matrix,
    x: Matrix,
    eri: EriTensor,
    e_nuc: f64,
    xc: &'a mut dyn ExchangeCorrelation,
}

impl ScfSystem<'_> {
    /// Fock matrix and energy components for density `p`. theory.typ eq:fock-rks, eq:e-rks
    fn fock(&mut self, p: &Matrix) -> Result<(Matrix, EnergyComponents)> {
        let a_hf = self.xc.exact_exchange_fraction();
        let (j, k) = self.eri.coulomb_exchange(p, a_hf != 0.0);
        let mut f = &self.h + &j;
        let mut comps = EnergyComponents {
            one_electron: p.dot(&self.h),
            coulomb: 0.5 * p.dot(&j),
            nuclear_repulsion: self.e_nuc,
            ..Default::default()
        };
        if let Some(k) = k {
            f = &f - &k.scaled(0.5 * a_hf);
            comps.exchange = -0.25 * a_hf * p.dot(&k);
        }
        if let Some(xc) = self.xc.evaluate(p)? {
            f = &f + &xc.potential;
            comps.xc = xc.energy;
        }
        Ok((f, comps))
    }

    /// Solves F C = S C ε via X (theory.typ eq:roothaan, eq:orth).
    fn diagonalize(&self, f: &Matrix) -> Result<(Vec<f64>, Matrix)> {
        let fp = self.x.transpose().matmul(f).matmul(&self.x);
        let e = symmetric_eigen(&fp)?;
        Ok((e.values, self.x.matmul(&e.vectors)))
    }
}

/// P = 2 Σ_i^occ C_μi C_νi. theory.typ eq:density
fn density(c: &Matrix, nocc: usize) -> Matrix {
    let n = c.rows();
    let mut p = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..=mu {
            let v: f64 = (0..nocc).map(|i| c[(mu, i)] * c[(nu, i)]).sum::<f64>() * 2.0;
            p[(mu, nu)] = v;
            p[(nu, mu)] = v;
        }
    }
    p
}

/// Generic closed-shell SCF with DIIS for the Fock matrix of theory.typ eq:fock-rks.
pub fn run_scf(
    mol: &Molecule,
    basis: &Basis,
    xc: &mut dyn ExchangeCorrelation,
    options: &ScfOptions,
) -> Result<ScfResult> {
    let n = basis.nao();
    let nelec = mol.n_electrons();
    if !nelec.is_multiple_of(2) {
        return Err(QcError::InvalidInput(format!("closed-shell SCF needs an even electron count, got {nelec}")));
    }
    let nocc = nelec / 2;
    if nocc > n {
        return Err(QcError::InvalidInput("more occupied orbitals than basis functions".into()));
    }
    if options.diis_size == 0 {
        return Err(QcError::InvalidInput("diis_size must be at least 1".into()));
    }

    // Step 1: integrals and symmetric orthogonalization (theory.typ eq:orth).
    let s = overlap(basis);
    let se = symmetric_eigen(&s)?;
    let smin = se.values.first().copied().unwrap_or(0.0);
    if smin < options.linear_dependence_threshold {
        return Err(QcError::LinearDependence(smin));
    }
    let u = &se.vectors;
    let x = Matrix::from_fn(n, n, |i, j| (0..n).map(|k| u[(i, k)] * u[(j, k)] / se.values[k].sqrt()).sum());
    let mut sys = ScfSystem {
        h: core_hamiltonian(basis, mol),
        s,
        x,
        eri: EriTensor::compute(basis, options.schwarz_threshold),
        e_nuc: mol.nuclear_repulsion(),
        xc,
    };

    // Step 2: initial density.
    let mut p = match &options.initial_density {
        Some(p0) => {
            if p0.rows() != n || p0.cols() != n {
                return Err(QcError::InvalidInput(format!(
                    "initial density is {}x{}, basis has {n} functions",
                    p0.rows(),
                    p0.cols()
                )));
            }
            p0.clone()
        }
        None => {
            let (_, c) = sys.diagonalize(&sys.h)?;
            density(&c, nocc)
        }
    };

    let mut diis: VecDeque<(Matrix, Matrix)> = VecDeque::new();
    let mut e_old: Option<f64> = None;
    let (mut last_de, mut last_err) = (f64::INFINITY, f64::INFINITY);
    for iter in 1..=options.max_iterations {
        // Step 3: Fock matrix and energy.
        let (f, comps) = sys.fock(&p)?;
        let e = comps.total();
        if !e.is_finite() {
            return Err(QcError::ScfNotConverged { iterations: iter, delta_e: f64::NAN, max_error: f64::NAN });
        }
        // theory.typ eq:diis-err
        let fps = f.matmul(&p).matmul(&sys.s);
        let comm = &fps - &fps.transpose(); // S P F = (F P S)ᵀ for symmetric F, P, S
        let err = sys.x.transpose().matmul(&comm).matmul(&sys.x);
        let max_err = err.max_abs();
        last_err = max_err;
        if let Some(eo) = e_old {
            last_de = (e - eo).abs();
            // Step 6: convergence test.
            if last_de < options.energy_tolerance && max_err < options.error_tolerance {
                return finish(&mut sys, &f, nocc, iter, last_de, max_err);
            }
        }
        e_old = Some(e);
        // Step 4: DIIS extrapolation.
        diis.push_back((f.clone(), err));
        while diis.len() > options.diis_size {
            diis.pop_front();
        }
        let f_ext = if diis.len() >= 2 { diis_extrapolate(&mut diis)? } else { f };
        // Step 5: new density.
        let (_, c) = sys.diagonalize(&f_ext)?;
        p = density(&c, nocc);
    }
    Err(QcError::ScfNotConverged { iterations: options.max_iterations, delta_e: last_de, max_error: last_err })
}

/// After convergence: orbitals from the Fock matrix of the converged density, and
/// energy components re-evaluated for the density of those orbitals (so that energy,
/// density and orbitals are mutually consistent).
fn finish(sys: &mut ScfSystem, f: &Matrix, nocc: usize, iterations: usize, de: f64, err: f64) -> Result<ScfResult> {
    let (eps, c) = sys.diagonalize(f)?;
    let p = density(&c, nocc);
    let (_, comps) = sys.fock(&p)?;
    Ok(ScfResult {
        energy: comps.total(),
        components: comps,
        orbital_energies: eps,
        coefficients: c,
        density: p,
        n_occupied: nocc,
        iterations,
        final_delta_e: de,
        final_error: err,
    })
}

/// Pulay DIIS: solve theory.typ eq:diis, dropping the oldest pair while singular.
fn diis_extrapolate(diis: &mut VecDeque<(Matrix, Matrix)>) -> Result<Matrix> {
    loop {
        let m = diis.len();
        if m < 2 {
            return Ok(diis.back().expect("non-empty DIIS history").0.clone());
        }
        let mut b = Matrix::zeros(m + 1, m + 1);
        for k in 0..m {
            for l in 0..=k {
                let v = diis[k].1.dot(&diis[l].1);
                b[(k, l)] = v;
                b[(l, k)] = v;
            }
        }
        // Scaling B by a constant leaves the coefficients unchanged and keeps the
        // pivot test meaningful as the errors become small.
        let scale = (0..m).map(|k| b[(k, k)]).fold(0.0, f64::max);
        if scale > 0.0 {
            for k in 0..m {
                for l in 0..m {
                    b[(k, l)] /= scale;
                }
            }
        }
        for k in 0..m {
            b[(k, m)] = -1.0;
            b[(m, k)] = -1.0;
        }
        let mut rhs = vec![0.0; m + 1];
        rhs[m] = -1.0;
        match solve(&b, &rhs) {
            Ok(c) => {
                let n = diis[0].0.rows();
                let mut f = Matrix::zeros(n, n);
                for (k, (fk, _)) in diis.iter().enumerate() {
                    for (o, v) in f.as_mut_slice().iter_mut().zip(fk.as_slice()) {
                        *o += c[k] * v;
                    }
                }
                return Ok(f);
            }
            Err(QcError::SingularMatrix) => {
                diis.pop_front();
            }
            Err(e) => return Err(e),
        }
    }
}

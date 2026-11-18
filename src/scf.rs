//! Closed-shell SCF (RHF now; RKS via a pluggable exchange-correlation term).
//! theory.typ sections 6–7.

use crate::basis::Basis;
use crate::error::{QcError, Result};
use crate::integrals::{EriTensor, core_hamiltonian, overlap};
use crate::linalg::{Matrix, solve, symmetric_eigen};
use crate::molecule::{Atom, Molecule};
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

/// How the starting density is chosen when none is supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitialGuess {
    /// Superposition of spherically averaged atomic densities (theory.typ eq:sad).
    #[default]
    AtomicDensities,
    /// Orbitals of the core Hamiltonian. Cheap, but a poor start for larger molecules:
    /// DIIS can fail to converge from it.
    CoreHamiltonian,
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
    /// Starting density matrix (e.g. from a previous geometry); if `None`, the
    /// density is built according to `initial_guess`.
    pub initial_density: Option<Matrix>,
    /// Guess used when `initial_density` is `None`.
    pub initial_guess: InitialGuess,
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
            initial_guess: InitialGuess::default(),
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

/// Occupation numbers of a spherically averaged atom (theory.typ eq:sad-occ): orbitals
/// are filled in order of energy, and a set of degenerate orbitals shares its electrons
/// equally.
fn atomic_occupations(energies: &[f64], n_electrons: usize) -> Vec<f64> {
    let mut occ = vec![0.0; energies.len()];
    let mut remaining = n_electrons as f64;
    let mut start = 0;
    while remaining > 0.0 && start < energies.len() {
        let mut end = start + 1;
        while end < energies.len() && (energies[end] - energies[start]).abs() < 1e-6 {
            end += 1;
        }
        let g = (end - start) as f64;
        let n = remaining.min(2.0 * g);
        for o in &mut occ[start..end] {
            *o = n / g;
        }
        remaining -= n;
        start = end;
    }
    occ
}

/// Density of one neutral, spherically averaged atom in its own basis: a small
/// Hartree–Fock calculation with fractional occupations (theory.typ eq:sad-occ).
fn atomic_density(atom: Atom, basis: &Basis, schwarz_threshold: f64) -> Result<Matrix> {
    let mol = Molecule::new(vec![atom], 0)?;
    let n = basis.nao();
    let s = overlap(basis);
    let se = symmetric_eigen(&s)?;
    let u = &se.vectors;
    // Atomic basis sets are far from linear dependence; clamp instead of failing.
    let x = Matrix::from_fn(n, n, |i, j| (0..n).map(|k| u[(i, k)] * u[(j, k)] / se.values[k].max(1e-10).sqrt()).sum());
    let h = core_hamiltonian(basis, &mol);
    let eri = EriTensor::compute(basis, schwarz_threshold);
    let nelec = atom.z as usize;
    let mut p = Matrix::zeros(n, n);
    for iter in 0..100 {
        let f = if iter == 0 {
            h.clone()
        } else {
            let (j, k) = eri.coulomb_exchange(&p, true);
            &(&h + &j) - &k.expect("exchange requested").scaled(0.5)
        };
        let e = symmetric_eigen(&x.transpose().matmul(&f).matmul(&x))?;
        let c = x.matmul(&e.vectors);
        let occ = atomic_occupations(&e.values, nelec);
        let p_new = Matrix::from_fn(n, n, |a, b| (0..n).map(|i| occ[i] * c[(a, i)] * c[(b, i)]).sum());
        let change = (&p_new - &p).max_abs();
        // Plain 50 % damping is enough for atoms; the result only seeds the molecular SCF.
        p = if iter == 0 { p_new } else { &p.scaled(0.5) + &p_new.scaled(0.5) };
        if change < 1e-6 {
            break;
        }
    }
    Ok(p)
}

/// Initial density as a superposition of atomic densities: the block-diagonal matrix
/// of the densities of the neutral, spherically averaged atoms (theory.typ eq:sad).
/// It has the right number of electrons for a neutral molecule and is not idempotent;
/// the first diagonalization of the SCF restores both.
pub fn atomic_density_guess(mol: &Molecule, basis: &Basis, schwarz_threshold: f64) -> Result<Matrix> {
    let n = basis.nao();
    let mut p = Matrix::zeros(n, n);
    let mut cache: Vec<(u32, Matrix)> = Vec::new();
    let mut offset = 0;
    for (ia, atom) in mol.atoms().iter().enumerate() {
        let sub = basis.atom_basis(ia);
        let m = sub.nao();
        if !cache.iter().any(|(z, pa)| *z == atom.z && pa.rows() == m) {
            cache.push((atom.z, atomic_density(*atom, &sub, schwarz_threshold)?));
        }
        let pa = &cache.iter().find(|(z, pa)| *z == atom.z && pa.rows() == m).expect("just inserted").1;
        // AOs are ordered atom by atom, so the atom's block is contiguous.
        for a in 0..m {
            for b in 0..m {
                p[(offset + a, offset + b)] = pa[(a, b)];
            }
        }
        offset += m;
    }
    Ok(p)
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
        None => match options.initial_guess {
            InitialGuess::AtomicDensities => atomic_density_guess(mol, basis, options.schwarz_threshold)?,
            InitialGuess::CoreHamiltonian => {
                let (_, c) = sys.diagonalize(&sys.h)?;
                density(&c, nocc)
            }
        },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Unit;

    #[test]
    fn atomic_occupations_fill_degenerate_sets_equally() {
        // Carbon-like: 1s, 2s, three degenerate 2p, then higher orbitals.
        let eps = [-11.0, -0.7, -0.4, -0.4, -0.4, 0.5, 0.9];
        let occ = atomic_occupations(&eps, 6);
        let expect = [2.0, 2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0, 0.0];
        for (o, e) in occ.iter().zip(expect) {
            assert!((o - e).abs() < 1e-15);
        }
        assert_eq!(atomic_occupations(&[-0.5, 0.3], 1), vec![1.0, 0.0]);
        assert!((atomic_occupations(&eps, 10).iter().sum::<f64>() - 10.0).abs() < 1e-14);
    }

    #[test]
    fn atomic_guess_has_the_right_electron_count_and_same_converged_energy() {
        let mol = Molecule::from_symbols(
            &["C", "O", "H", "H"],
            &[[0.0, 0.0, -0.53], [0.0, 0.0, 0.68], [0.0, 0.94, -1.11], [0.0, -0.94, -1.11]],
            Unit::Angstrom,
            0,
        )
        .unwrap();
        let basis = Basis::from_name(&mol, "6-31g*").unwrap();
        let p0 = atomic_density_guess(&mol, &basis, 1e-12).unwrap();
        // Tr(P S) = number of electrons (theory.typ eq:sad).
        assert!((p0.dot(&overlap(&basis)) - mol.n_electrons() as f64).abs() < 1e-8);
        // Symmetric, and zero between different atoms.
        let ao_atom = basis.ao_atom();
        for a in 0..basis.nao() {
            for b in 0..basis.nao() {
                assert!((p0[(a, b)] - p0[(b, a)]).abs() < 1e-12);
                if ao_atom[a] != ao_atom[b] {
                    assert_eq!(p0[(a, b)], 0.0);
                }
            }
        }
        // The converged result does not depend on the guess, and the atomic guess is the better start.
        let tight = ScfOptions { error_tolerance: 1e-9, ..Default::default() };
        let sad = run_scf(&mol, &basis, &mut HartreeFock, &tight).unwrap();
        let core_opts = ScfOptions { initial_guess: InitialGuess::CoreHamiltonian, ..tight };
        let core = run_scf(&mol, &basis, &mut HartreeFock, &core_opts).unwrap();
        assert!((sad.energy - core.energy).abs() < 1e-9, "{} vs {}", sad.energy, core.energy);
        assert!(sad.iterations <= core.iterations, "{} vs {} iterations", sad.iterations, core.iterations);
    }
}

//! Closed-shell SCF (RHF now; RKS via a pluggable exchange-correlation term).
//! theory.typ sec:rhf, sec:rks.

use crate::basis::Basis;
use crate::direct::{DirectBuildStatistics, DirectJk};
use crate::error::{QcError, Result};
use crate::integrals::{EriTensor, ExchangeKernel, core_hamiltonian, overlap};
use crate::linalg::{Matrix, SymmetricEigen, solve, symmetric_eigen};
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
/// F = h + J - ½ K^x + V^xc (theory.typ eq:fock-rks, eq:cam-fock), where K^x is the UNCHECKED(Johnson1993) VERIFIED(Yanai2004, derived from Eqs. (6), (7), p. 52) VERIFIED(Kohn1965, derived from Eqs. (2.6)-(2.9), p. A1134; a_HF = 0)
/// exchange matrix of the kernel [α + β erf(ωr)]/r (eq:cam-k); K^x = a_HF K for an VERIFIED(Yanai2004, derived from Eqs. (6), (7), p. 52)
/// ordinary hybrid.
pub trait ExchangeCorrelation {
    /// Fraction a_HF of exact (Hartree–Fock) exchange (the short-range fraction α for a
    /// range-separated kernel).
    fn exact_exchange_fraction(&self) -> f64;
    /// The exact-exchange kernel (α, β, ω) of theory.typ eq:cam-k. The default is the
    /// global fraction [`ExchangeCorrelation::exact_exchange_fraction`] (β = 0); this
    /// is what the SCF uses.
    fn exchange_kernel(&self) -> ExchangeKernel {
        ExchangeKernel::global(self.exact_exchange_fraction())
    }
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
    /// Superposition of spherically averaged atomic densities (theory.typ eq:sad). UNCHECKED(VanLenthe2006)
    #[default]
    AtomicDensities,
    /// Orbitals of the core Hamiltonian. Cheap, but a poor start for larger molecules:
    /// DIIS can fail to converge from it.
    CoreHamiltonian,
}

/// How the two-electron integrals enter the Fock matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegralStrategy {
    /// Compute all unique integrals once and keep them in memory (8-fold packed,
    /// ~N⁴/8 doubles, twice that for a range-separated kernel; see
    /// [`EriTensor::memory_bytes`]). Fastest for small molecules.
    Stored,
    /// Integral-direct: recompute the screened shell quartets in every iteration and
    /// contract them at once into J and K (theory.typ eq:jk-quartet); memory O(N²).
    Direct,
    /// [`IntegralStrategy::Stored`] if the tensors fit within
    /// [`ScfOptions::eri_memory_budget`], [`IntegralStrategy::Direct`] otherwise.
    #[default]
    Auto,
}

/// Default for [`ScfOptions::eri_memory_budget`]: 0.5 GiB (about 150 basis functions).
/// Above roughly that size the direct build is faster than the stored tensor as
/// well as smaller: N = 168 took 5.4 s direct against 10.0 s stored (24 threads).
pub const DEFAULT_ERI_MEMORY_BUDGET: u64 = 1 << 29;

/// SCF settings. Defaults from theory.typ: `max_iterations`, `energy_tolerance` and
/// `error_tolerance` from sec:scf-procedure; `diis_size` from eq:diis;
/// `schwarz_threshold` from eq:schwarz; `linear_dependence_threshold` from eq:orth;
/// `direct_threshold`, `incremental_fock`, `full_rebuild_interval` and
/// `eri_memory_budget` from sec:direct.
#[derive(Debug, Clone)]
pub struct ScfOptions {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence threshold on |ΔE| (Hartree).
    pub energy_tolerance: f64,
    /// Convergence threshold on max |e| of the DIIS error (eq:diis-err). UNCHECKED(Pulay1982)
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
    /// Stored or direct two-electron integrals (default: automatic).
    pub integral_strategy: IntegralStrategy,
    /// Largest total size (bytes) of the stored integral tensors for which
    /// [`IntegralStrategy::Auto`] chooses the stored strategy.
    pub eri_memory_budget: u64,
    /// Direct strategy: shell quartets whose density-weighted Schwarz bound
    /// (theory.typ eq:schwarz-dens) is below this are skipped (ΔP builds use down to
    /// [`INCREMENTAL_THRESHOLD_FACTOR`] times this). The energy error is of the order
    /// of 10 times the threshold.
    pub direct_threshold: f64,
    /// Direct strategy: build J and K from the density change ΔP and add them to the
    /// previous ones (theory.typ eq:fock-incr), as long as the DIIS error is at least
    /// [`INCREMENTAL_MIN_ERROR_FACTOR`] times `direct_threshold`.
    pub incremental_fock: bool,
    /// Direct strategy: a full build after this many incremental ones (also the first
    /// build and the one for the final energy are full).
    pub full_rebuild_interval: usize,
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
            integral_strategy: IntegralStrategy::default(),
            eri_memory_budget: DEFAULT_ERI_MEMORY_BUDGET,
            direct_threshold: 1e-12,
            incremental_fock: true,
            full_rebuild_interval: 10,
        }
    }
}

/// Energy decomposition (Hartree). theory.typ eq:e-rks UNCHECKED(Johnson1993) VERIFIED(Kohn1965, derived from Eqs. (2.1)-(2.3), (2.10), pp. A1133-A1134; a_HF = 0)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EnergyComponents {
    /// Σ P h.
    pub one_electron: f64,
    /// ½ Σ P J.
    pub coulomb: f64,
    /// -(1/4) Σ P K^x (= -(a_HF/4) Σ P K for an ordinary hybrid). theory.typ eq:cam-fock
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
    /// Density matrix P = 2 C_occ C_occᵀ built from `coefficients` (eq:density). UNCHECKED(Roothaan1951)
    pub density: Matrix,
    /// Number of doubly occupied orbitals.
    pub n_occupied: usize,
    /// Number of SCF iterations performed.
    pub iterations: usize,
    /// |ΔE| at convergence.
    pub final_delta_e: f64,
    /// max |e| of the DIIS error at convergence.
    pub final_error: f64,
    /// The integral strategy used ([`IntegralStrategy::Stored`] or
    /// [`IntegralStrategy::Direct`], never `Auto`).
    pub integral_strategy: IntegralStrategy,
    /// Direct strategy: one entry per J/K build (the last one is the full build for the
    /// final energy). Empty for the stored strategy.
    pub direct_builds: Vec<DirectBuildStatistics>,
}

/// Closed-shell Hartree–Fock energy.
pub fn run_rhf(mol: &Molecule, basis: &Basis, options: &ScfOptions) -> Result<ScfResult> {
    run_scf(mol, basis, &mut HartreeFock, options)
}

/// Smallest screening threshold of the ΔP builds relative to
/// [`ScfOptions::direct_threshold`] τ (theory.typ eq:fock-incr): a ΔP build uses
/// τ · clamp(e / ([`INCREMENTAL_ERROR_RATIO`] τ), this, 1) with e the DIIS error of the
/// previous iteration. Each build may leave out terms up to the threshold, and in the
/// ΔP builds these errors add up from one iteration to the next instead of being the
/// same function of P every time: with τ = 1e-12 throughout, the Fock matrix carried
/// noise that kept DIIS at ~2e-10, above the 1e-10 used for gradients. Far from
/// convergence the noise does not matter and the full τ keeps the ΔP builds cheap.
pub const INCREMENTAL_THRESHOLD_FACTOR: f64 = 0.01;

/// See [`INCREMENTAL_THRESHOLD_FACTOR`].
pub const INCREMENTAL_ERROR_RATIO: f64 = 1e6;

/// ΔP builds are used only while the DIIS error (eq:diis-err) of the previous
/// iteration is at least this multiple of [`ScfOptions::direct_threshold`]; below,
/// every build is full. With the default threshold 1e-12 this never triggers for the
/// usual convergence targets; for a loose threshold it keeps the SCF converging.
pub const INCREMENTAL_MIN_ERROR_FACTOR: f64 = 100.0;

/// Integral-direct J/K with the state of the incremental build (theory.typ eq:fock-incr).
struct DirectState {
    jk: DirectJk,
    /// Screening threshold of the full builds.
    threshold: f64,
    incremental: bool,
    interval: usize,
    /// Density, J and K of the last build.
    last: Option<(Matrix, Matrix, Option<Matrix>)>,
    /// Incremental builds since the last full one.
    since_full: usize,
    /// DIIS error (eq:diis-err) of the previous iteration.
    diis_error: f64,
    stats: Vec<DirectBuildStatistics>,
}

impl DirectState {
    /// J and K for density `p` (K in the convention of [`DirectJk::coulomb_exchange`]).
    /// Incremental unless `full`, the first build, or `interval` incremental builds
    /// have been done since the last full one.
    fn build(&mut self, p: &Matrix, want_k: bool, full: bool) -> (Matrix, Option<Matrix>) {
        let use_delta = self.incremental && !full && self.since_full < self.interval;
        let (j, k) = match (&self.last, use_delta) {
            (Some((p_old, j_old, k_old)), true) => {
                // theory.typ eq:fock-incr
                let dp = p - p_old;
                // Tighter as the SCF converges: the noise of the ΔP builds (~100 τ_ΔP)
                // stays a hundredth of the DIIS error.
                let scale = (self.diis_error / (INCREMENTAL_ERROR_RATIO * self.threshold))
                    .clamp(INCREMENTAL_THRESHOLD_FACTOR, 1.0);
                let tau = self.threshold * scale;
                let (dj, dk, mut st) = self.jk.coulomb_exchange_screened(&dp, want_k, tau);
                st.incremental = true;
                self.stats.push(st);
                self.since_full += 1;
                let k = match (k_old, dk) {
                    (Some(k_old), Some(dk)) => Some(k_old + &dk),
                    _ => None,
                };
                (j_old + &dj, k)
            }
            _ => {
                let (j, k, st) = self.jk.coulomb_exchange(p, want_k);
                self.stats.push(st);
                self.since_full = 0;
                (j, k)
            }
        };
        self.last = Some((p.clone(), j.clone(), k.clone()));
        (j, k)
    }
}

/// Where J and K come from.
enum TwoElectron {
    Stored {
        eri: EriTensor,
        /// α (μν|λσ) + β (μν|λσ)_ω for a range-separated kernel (theory.typ eq:cam-k).
        exchange_eri: Option<EriTensor>,
    },
    Direct(Box<DirectState>),
}

/// Integrals and derived matrices shared by all SCF iterations.
struct ScfSystem<'a> {
    h: Matrix,
    s: Matrix,
    x: Matrix,
    two: TwoElectron,
    /// Exact-exchange kernel of `xc`.
    kernel: ExchangeKernel,
    e_nuc: f64,
    xc: &'a mut dyn ExchangeCorrelation,
}

impl ScfSystem<'_> {
    /// Fock matrix and energy components for density `p`. theory.typ eq:fock-rks,
    /// eq:e-rks, eq:cam-fock. `full` forbids an incremental direct build.
    fn fock(&mut self, p: &Matrix, full: bool) -> Result<(Matrix, EnergyComponents)> {
        let (j, k, a_hf) = match &mut self.two {
            // K^x from the combined tensor (theory.typ eq:cam-k), already scaled.
            TwoElectron::Stored { eri, exchange_eri: Some(x) } => {
                (eri.coulomb_exchange(p, false).0, Some(x.exchange(p)), 1.0)
            }
            // K^x = a_HF K (theory.typ eq:fock-rks).
            TwoElectron::Stored { eri, exchange_eri: None } => {
                let a_hf = self.kernel.alpha;
                let (j, k) = eri.coulomb_exchange(p, a_hf != 0.0);
                (j, k, a_hf)
            }
            // Same convention: K^x for a range-separated kernel, otherwise K.
            TwoElectron::Direct(d) => {
                if self.kernel.is_range_separated() {
                    let (j, k) = d.build(p, true, full);
                    (j, k, 1.0)
                } else {
                    let a_hf = self.kernel.alpha;
                    let (j, k) = d.build(p, a_hf != 0.0, full);
                    (j, k, a_hf)
                }
            }
        };
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

    /// Solves F C = S C ε via X (theory.typ eq:roothaan, eq:orth). UNCHECKED(Roothaan1951) UNCHECKED(Szabo1996)
    fn diagonalize(&self, f: &Matrix) -> Result<(Vec<f64>, Matrix)> {
        let fp = self.x.transpose().matmul(f).matmul(&self.x);
        let e = symmetric_eigen(&fp)?;
        Ok((e.values, self.x.matmul(&e.vectors)))
    }
}

/// P = 2 Σ_i^occ C_μi C_νi. theory.typ eq:density
///
/// The lower triangle is built in parallel over rows (each element summed over i in
/// order by one task, so the result does not depend on the number of threads) and then
/// mirrored.
fn density(c: &Matrix, nocc: usize) -> Matrix {
    use rayon::prelude::*;

    let n = c.rows();
    let mut p = Matrix::zeros(n, n);
    if n == 0 {
        return p;
    }
    let (cd, ldc) = (c.as_slice(), c.cols());
    p.as_mut_slice().par_chunks_mut(n).enumerate().for_each(|(mu, row)| {
        let cmu = &cd[mu * ldc..mu * ldc + nocc];
        for (nu, v) in row[..=mu].iter_mut().enumerate() {
            let cnu = &cd[nu * ldc..nu * ldc + nocc];
            *v = cmu.iter().zip(cnu).map(|(a, b)| a * b).sum::<f64>() * 2.0;
        }
    });
    for mu in 0..n {
        for nu in 0..mu {
            p[(nu, mu)] = p[(mu, nu)];
        }
    }
    p
}

/// X = U s^{-1/2} Uᵀ from the eigen-decomposition of S (theory.typ eq:orth), built in
/// parallel over rows; each element is summed over k in order by one task.
fn symmetric_orthogonalizer(se: &SymmetricEigen) -> Matrix {
    use rayon::prelude::*;

    let u = &se.vectors;
    let n = u.rows();
    let mut x = Matrix::zeros(n, n);
    if n == 0 {
        return x;
    }
    let sqrt_s: Vec<f64> = se.values.iter().map(|s| s.sqrt()).collect();
    let ud = u.as_slice();
    x.as_mut_slice().par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let ui = &ud[i * n..(i + 1) * n];
        for (j, v) in row.iter_mut().enumerate() {
            let uj = &ud[j * n..(j + 1) * n];
            *v = ui.iter().zip(uj).zip(&sqrt_s).map(|((a, b), r)| a * b / r).sum();
        }
    });
    x
}

/// Occupation numbers of a spherically averaged atom (theory.typ eq:sad-occ): orbitals UNCHECKED(VanLenthe2006)
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

/// Generic closed-shell SCF with DIIS for the Fock matrix of theory.typ eq:fock-rks
/// (eq:cam-fock for a range-separated exchange kernel).
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
    let x = symmetric_orthogonalizer(&se);
    let kernel = xc.exchange_kernel();
    let two = match resolve_strategy(options, n, kernel) {
        IntegralStrategy::Direct => TwoElectron::Direct(Box::new(DirectState {
            jk: DirectJk::new(basis, kernel, options.direct_threshold),
            threshold: options.direct_threshold,
            incremental: options.incremental_fock,
            interval: options.full_rebuild_interval,
            last: None,
            since_full: 0,
            diis_error: f64::INFINITY,
            stats: Vec::new(),
        })),
        _ => {
            let eri = EriTensor::compute(basis, options.schwarz_threshold);
            // A second tensor only for a range-separated kernel (theory.typ eq:cam-k).
            let exchange_eri = eri.exchange_tensor(basis, kernel, options.schwarz_threshold);
            TwoElectron::Stored { eri, exchange_eri }
        }
    };
    let mut sys = ScfSystem {
        h: core_hamiltonian(basis, mol),
        s,
        x,
        two,
        kernel,
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
        // Close to convergence the direct strategy builds the full J and K again
        // (theory.typ eq:fock-incr), so that the noise of the ΔP builds cannot stall DIIS.
        let full = last_err < INCREMENTAL_MIN_ERROR_FACTOR * options.direct_threshold;
        if let TwoElectron::Direct(d) = &mut sys.two {
            d.diis_error = last_err;
        }
        let (f, comps) = sys.fock(&p, full)?;
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
    // A full build, so that the energy carries no error accumulated by the
    // incremental builds (theory.typ eq:fock-incr).
    let (_, comps) = sys.fock(&p, true)?;
    let (integral_strategy, direct_builds) = match &mut sys.two {
        TwoElectron::Stored { .. } => (IntegralStrategy::Stored, Vec::new()),
        TwoElectron::Direct(d) => (IntegralStrategy::Direct, std::mem::take(&mut d.stats)),
    };
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
        integral_strategy,
        direct_builds,
    })
}

/// The strategy `options` asks for, with [`IntegralStrategy::Auto`] resolved: stored
/// if the tensor(s) for `nao` functions (two for a range-separated `kernel`) fit
/// within [`ScfOptions::eri_memory_budget`].
pub fn resolve_strategy(options: &ScfOptions, nao: usize, kernel: ExchangeKernel) -> IntegralStrategy {
    match options.integral_strategy {
        IntegralStrategy::Auto => {
            let tensors = if kernel.is_range_separated() { 2 } else { 1 };
            if EriTensor::memory_bytes(nao).saturating_mul(tensors) <= options.eri_memory_budget {
                IntegralStrategy::Stored
            } else {
                IntegralStrategy::Direct
            }
        }
        s => s,
    }
}

/// Pulay DIIS: solve theory.typ eq:diis, dropping the oldest pair while singular. UNCHECKED(Pulay1980)
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

    fn bits(m: &Matrix) -> Vec<u64> {
        m.as_slice().iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn density_and_orthogonalizer_match_serial_formulas_bitwise() {
        let n = 61;
        let a = Matrix::from_fn(n, n, |i, j| ((i * 37 + j * 11) % 29) as f64 / 13.0 - 1.0);
        let sym = &a + &a.transpose();
        let se = symmetric_eigen(&(&sym.matmul(&sym) + &Matrix::identity(n))).unwrap();
        let u = &se.vectors;
        // The plain serial formulas (theory.typ eq:orth, eq:density).
        let x_ref = Matrix::from_fn(n, n, |i, j| (0..n).map(|k| u[(i, k)] * u[(j, k)] / se.values[k].sqrt()).sum());
        let nocc = 17;
        let p_ref = Matrix::from_fn(n, n, |mu, nu| {
            let (r, c) = if nu <= mu { (mu, nu) } else { (nu, mu) };
            (0..nocc).map(|i| a[(r, i)] * a[(c, i)]).sum::<f64>() * 2.0
        });
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let four = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for x in [symmetric_orthogonalizer(&se), one.install(|| symmetric_orthogonalizer(&se)), four.install(|| symmetric_orthogonalizer(&se))] {
            assert_eq!(bits(&x), bits(&x_ref));
        }
        for p in [density(&a, nocc), one.install(|| density(&a, nocc)), four.install(|| density(&a, nocc))] {
            assert_eq!(bits(&p), bits(&p_ref));
        }
        assert_eq!(density(&Matrix::zeros(0, 0), 0).rows(), 0);
    }

    #[test]
    fn scf_energy_bitwise_independent_of_threads() {
        let mol = Molecule::from_symbols(
            &["O", "H", "H"],
            &[[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
            Unit::Angstrom,
            0,
        )
        .unwrap();
        let basis = Basis::from_name(&mol, "6-31g*").unwrap();
        let run = || run_scf(&mol, &basis, &mut HartreeFock, &ScfOptions::default()).unwrap();
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(run);
        let four = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(run);
        let default = run();
        assert_eq!(one.energy.to_bits(), default.energy.to_bits());
        assert_eq!(four.energy.to_bits(), default.energy.to_bits());
        assert_eq!(one.iterations, default.iterations);
    }
}

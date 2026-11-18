//! Integral-direct SCF (theory.typ sec:direct, eq:jk-quartet, eq:schwarz-dens,
//! eq:fock-incr) against the stored integrals: energies, orbital energies and
//! gradients, the screening threshold, the incremental Fock build and the automatic
//! choice of the strategy.

use qc_rs::{Calculation, IntegralStrategy, Molecule, ScfOptions, Unit};

fn h2co() -> Molecule {
    // Slightly distorted, so that no symmetry hides an error.
    Molecule::from_symbols(
        &["C", "O", "H", "H"],
        &[[0.0, 0.03, -0.53], [0.02, 0.0, 0.68], [0.0, 0.94, -1.11], [0.05, -0.92, -1.12]],
        Unit::Angstrom,
        0,
    )
    .unwrap()
}

fn water() -> Molecule {
    Molecule::from_symbols(
        &["O", "H", "H"],
        &[[0.0, 0.0, 0.10], [0.0, 0.80, -0.45], [0.0, -0.70, -0.52]],
        Unit::Angstrom,
        0,
    )
    .unwrap()
}

/// n-butane-like chain C4H10 (Å), long enough for the screening to matter.
fn butane() -> Molecule {
    let c = [[0.0, 0.0, 0.0], [1.53, 0.0, 0.0], [2.04, 1.44, 0.0], [3.57, 1.44, 0.0]];
    let h = [
        [-0.36, -1.03, 0.0],
        [-0.36, 0.51, 0.89],
        [-0.36, 0.51, -0.89],
        [1.89, -0.51, 0.89],
        [1.89, -0.51, -0.89],
        [1.68, 1.95, 0.89],
        [1.68, 1.95, -0.89],
        [3.93, 2.47, 0.0],
        [3.93, 0.93, 0.89],
        [3.93, 0.93, -0.89],
    ];
    let mut symbols = vec!["C"; 4];
    symbols.extend(["H"; 10]);
    let coords: Vec<[f64; 3]> = c.iter().chain(&h).copied().collect();
    Molecule::from_symbols(&symbols, &coords, Unit::Angstrom, 0).unwrap()
}

fn tight() -> ScfOptions {
    ScfOptions { error_tolerance: 1e-10, energy_tolerance: 1e-12, ..Default::default() }
}

fn max_dev(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn direct_matches_stored_energy_orbitals_gradient() {
    let cases: [(Molecule, &str, &str, bool); 6] = [
        (h2co(), "hf", "6-31g*", false),
        (h2co(), "hf", "6-31g*", true),
        (water(), "hf", "6-31g**", false),
        (water(), "b3lyp", "6-31g*", true),
        (h2co(), "b3lyp", "6-31g*", false),
        (water(), "cam-b3lyp", "6-31g*", false),
    ];
    for (mol, method, basis, cart) in cases {
        let calc = |s: IntegralStrategy| {
            Calculation::new(&mol, method, basis)
                .unwrap()
                .with_cartesian_d(cart)
                .with_scf_options(tight())
                .with_integral_strategy(s)
                .gradient()
                .unwrap()
        };
        let stored = calc(IntegralStrategy::Stored);
        let direct = calc(IntegralStrategy::Direct);
        assert_eq!(stored.scf.integral_strategy, IntegralStrategy::Stored);
        assert_eq!(direct.scf.integral_strategy, IntegralStrategy::Direct);
        assert!(stored.scf.direct_builds.is_empty());
        assert!(!direct.scf.direct_builds.is_empty());
        let de = (stored.energy - direct.energy).abs();
        let deps = max_dev(&stored.scf.orbital_energies, &direct.scf.orbital_energies);
        let gs: Vec<f64> = stored.gradient.iter().flatten().copied().collect();
        let gd: Vec<f64> = direct.gradient.iter().flatten().copied().collect();
        let dg = max_dev(&gs, &gd);
        println!("{method}/{basis} cart={cart}: |dE| {de:.1e} |d eps| {deps:.1e} |dg| {dg:.1e}");
        assert!(de <= 1e-9, "{method}/{basis} cart={cart}: |dE| = {de:.2e}");
        assert!(deps <= 1e-8, "{method}/{basis} cart={cart}: |d eps| = {deps:.2e}");
        assert!(dg <= 1e-8, "{method}/{basis} cart={cart}: |dg| = {dg:.2e}");
    }
}

#[test]
fn screening_threshold_controls_the_error() {
    let mol = butane();
    let energy = |opts: ScfOptions| {
        Calculation::new(&mol, "hf", "6-31g*").unwrap().with_scf_options(opts).energy().unwrap().scf
    };
    // Reference: stored integrals without screening.
    let exact = energy(ScfOptions { integral_strategy: IntegralStrategy::Stored, schwarz_threshold: 0.0, ..tight() });
    // Each skipped quartet is below τ; together they change the energy by ~10 τ here.
    let mut last_skipped = 1.0;
    for (tau, tol) in [(1e-6, 3e-5), (1e-8, 3e-7), (1e-10, 3e-9), (1e-12, 2e-11), (1e-14, 1e-11)] {
        let r = energy(ScfOptions { integral_strategy: IntegralStrategy::Direct, direct_threshold: tau, ..tight() });
        let de = (r.energy - exact.energy).abs();
        let skipped = r.direct_builds.last().unwrap().skipped_fraction();
        println!("tau = {tau:.0e}: |E - E_exact| = {de:.2e}, final build skipped {:.1} %", 100.0 * skipped);
        assert!(de < tol, "tau = {tau:.0e}: |dE| = {de:.2e} >= {tol:.0e}");
        // A tighter threshold never skips more.
        assert!(skipped <= last_skipped);
        last_skipped = skipped;
    }
    // A loose threshold does change the energy (the test above is not vacuous).
    let loose = energy(ScfOptions { integral_strategy: IntegralStrategy::Direct, direct_threshold: 1e-6, ..tight() });
    assert!((loose.energy - exact.energy).abs() > 1e-7);
}

#[test]
fn incremental_build_converges_to_the_full_build_result() {
    let mol = butane();
    let run = |incremental: bool| {
        let opts = ScfOptions { integral_strategy: IntegralStrategy::Direct, incremental_fock: incremental, ..tight() };
        Calculation::new(&mol, "hf", "6-31g*").unwrap().with_scf_options(opts).energy().unwrap().scf
    };
    let inc = run(true);
    let full = run(false);
    assert!(inc.direct_builds.iter().any(|b| b.incremental));
    assert!(full.direct_builds.iter().all(|b| !b.incremental));
    // The final energy comes from a full build.
    assert!(!inc.direct_builds.last().unwrap().incremental);
    // ΔP builds skip more quartets as the SCF converges.
    let incs: Vec<f64> = inc.direct_builds.iter().filter(|b| b.incremental).map(|b| b.skipped_fraction()).collect();
    assert!(incs.last().unwrap() > incs.first().unwrap(), "{incs:?}");
    let de = (inc.energy - full.energy).abs();
    println!("incremental vs full: |dE| = {de:.1e}, {} vs {} iterations", inc.iterations, full.iterations);
    assert!(de <= 1e-9, "|dE| = {de:.2e}");
    assert!(max_dev(&inc.orbital_energies, &full.orbital_energies) <= 1e-8);
}

#[test]
fn rebuild_interval_and_reproducibility() {
    // Every second build full; results identical from run to run (fixed reduction order).
    let mol = h2co();
    let opts = ScfOptions { integral_strategy: IntegralStrategy::Direct, full_rebuild_interval: 1, ..tight() };
    let calc = Calculation::new(&mol, "hf", "6-31g*").unwrap().with_scf_options(opts);
    let a = calc.energy().unwrap().scf;
    let b = calc.energy().unwrap().scf;
    assert_eq!(a.energy, b.energy);
    assert_eq!(a.density, b.density);
    let kinds: Vec<bool> = a.direct_builds.iter().map(|s| s.incremental).collect();
    for w in kinds[..kinds.len() - 1].windows(2) {
        assert!(!(w[0] && w[1]), "two incremental builds in a row: {kinds:?}");
    }
}

#[test]
fn automatic_choice_follows_the_memory_budget() {
    let mol = h2co();
    let nao = 32; // H2CO/6-31G* with spherical d
    let one = qc_rs::integrals::EriTensor::memory_bytes(nao);
    let run = |method: &str, budget: u64| {
        let opts = ScfOptions { eri_memory_budget: budget, ..tight() };
        Calculation::new(&mol, method, "6-31g*").unwrap().with_scf_options(opts).energy().unwrap().scf
    };
    // Default: small molecules are stored.
    let stored = Calculation::new(&mol, "hf", "6-31g*").unwrap().with_scf_options(tight()).energy().unwrap().scf;
    assert_eq!(stored.coefficients.rows(), nao);
    assert_eq!(stored.integral_strategy, IntegralStrategy::Stored);
    // A budget below the tensor size switches to direct, with the same energy.
    let direct = run("hf", one - 1);
    assert_eq!(direct.integral_strategy, IntegralStrategy::Direct);
    assert!((direct.energy - stored.energy).abs() <= 1e-9, "{} vs {}", direct.energy, stored.energy);
    assert_eq!(run("hf", one).integral_strategy, IntegralStrategy::Stored);
    // A range-separated kernel needs two tensors.
    assert_eq!(run("cam-b3lyp", 2 * one - 1).integral_strategy, IntegralStrategy::Direct);
    assert_eq!(run("b3lyp", 2 * one - 1).integral_strategy, IntegralStrategy::Stored);
}

#[test]
fn optimization_with_direct_integrals() {
    let mol = water();
    let calc = |s| Calculation::new(&mol, "hf", "6-31g*").unwrap().with_integral_strategy(s);
    let settings = qc_rs::OptimizerSettings::default();
    let a = calc(IntegralStrategy::Stored).optimize(&settings).unwrap();
    let b = calc(IntegralStrategy::Direct).optimize(&settings).unwrap();
    assert!((a.energy - b.energy).abs() < 1e-9, "{} vs {}", a.energy, b.energy);
}

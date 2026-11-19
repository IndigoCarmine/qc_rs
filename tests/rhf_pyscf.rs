//! RHF single points compared against PySCF reference values
//! (tests/reference/pyscf_reference.json).

use qc_rs::basis::{Basis, BasisOptions, BasisSetDefinition};
use qc_rs::integrals::{kinetic, nuclear_attraction, overlap};
use qc_rs::linalg::symmetric_eigen;
use qc_rs::{Molecule, ScfOptions, Unit, dipole_moment, run_rhf};
use serde_json::Value;
use std::time::Instant;

fn load_reference() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/reference/pyscf_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("reference JSON")).expect("valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn max_dev(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Tracks the largest deviation per quantity and every tolerance violation.
#[derive(Default)]
struct Report {
    maxima: Vec<(&'static str, f64)>,
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, case: &str, what: &'static str, dev: f64, tol: f64) {
        match self.maxima.iter_mut().find(|(n, _)| *n == what) {
            Some((_, m)) => *m = m.max(dev),
            None => self.maxima.push((what, dev)),
        }
        if dev.is_nan() || dev > tol {
            self.failures.push(format!("{case}: {what} deviates by {dev:.3e} (tol {tol:.0e})"));
        }
    }
}

#[test]
fn rhf_single_points_match_pyscf() {
    let reference = load_reference();
    let mut report = Report::default();
    let mut n_cases = 0;
    for entry in reference["single_points"].as_array().unwrap() {
        if entry["method"] != "rhf" {
            continue;
        }
        n_cases += 1;
        let name = entry["name"].as_str().unwrap();
        let basis_name = entry["basis"].as_str().unwrap();
        let cart = entry["cartesian"].as_bool().unwrap();
        let case = format!("{name}/{basis_name}{}", if cart { "/cart" } else { "" });

        let symbols: Vec<&str> = entry["atoms"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
        let coords: Vec<[f64; 3]> = entry["coords_bohr"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let v = f64s(c);
                [v[0], v[1], v[2]]
            })
            .collect();
        let mol = Molecule::from_symbols(&symbols, &coords, Unit::Bohr, 0).unwrap();
        let def = BasisSetDefinition::builtin(basis_name).unwrap();
        let basis = Basis::new(&mol, &def, BasisOptions { cartesian_d: cart }).unwrap();

        assert_eq!(basis.nao() as u64, entry["nao"].as_u64().unwrap(), "{case}: nao");
        report.check(&case, "e_nuc", (mol.nuclear_repulsion() - entry["e_nuc"].as_f64().unwrap()).abs(), 1e-10);

        let t0 = Instant::now();
        let res = run_rhf(&mol, &basis, &ScfOptions::default()).unwrap();
        let elapsed = t0.elapsed();

        report.check(&case, "e_tot", (res.energy - entry["e_tot"].as_f64().unwrap()).abs(), 1e-8);
        report.check(&case, "mo_energy", max_dev(&res.orbital_energies, &f64s(&entry["mo_energy"])), 1e-6);

        let p = &res.density;
        report.check(&case, "e_kin", (p.dot(&kinetic(&basis)) - entry["e_kin"].as_f64().unwrap()).abs(), 1e-6);
        report.check(
            &case,
            "e_nuc_attr",
            (p.dot(&nuclear_attraction(&basis, &mol)) - entry["e_nuc_attr"].as_f64().unwrap()).abs(),
            1e-6,
        );
        let summary = &entry["scf_summary"];
        let c = &res.components;
        report.check(&case, "scf_summary.e1", (c.one_electron - summary["e1"].as_f64().unwrap()).abs(), 1e-6);
        report.check(&case, "scf_summary.e2", (c.two_electron() - summary["e2"].as_f64().unwrap()).abs(), 1e-6);
        report.check(&case, "scf_summary.coul", (c.coulomb - summary["coul"].as_f64().unwrap()).abs(), 1e-6);
        report.check(&case, "scf_summary.exc", (c.exchange - summary["exc"].as_f64().unwrap()).abs(), 1e-6);

        if !cart {
            // PySCF does not normalize individual Cartesian d and f components, so S differs there.
            let s_eig = symmetric_eigen(&overlap(&basis)).unwrap().values;
            report.check(&case, "ovlp_eigvals", max_dev(&s_eig, &f64s(&entry["ovlp_eigvals"])), 1e-8);
        }

        let mu = dipole_moment(&mol, &basis, p);
        report.check(&case, "dipole_au", max_dev(&mu, &f64s(&entry["dipole_au"])), 1e-6);

        println!(
            "{case:24} nao={:3} iter={:3} E={:.10} dE={:+.2e} time={:.3}s",
            basis.nao(),
            res.iterations,
            res.energy,
            res.energy - entry["e_tot"].as_f64().unwrap(),
            elapsed.as_secs_f64()
        );
    }
    assert_eq!(n_cases, 17, "expected 17 RHF reference entries (11 + 6 with 6-311G** or f functions)");
    println!("max deviations over all RHF cases:");
    for (what, m) in &report.maxima {
        println!("  {what:18} {m:.3e}");
    }
    assert!(report.failures.is_empty(), "failures:\n{}", report.failures.join("\n"));
}

#[test]
fn initial_density_restart_converges_immediately() {
    let mol = Molecule::from_symbols(
        &["O", "H", "H"],
        &[[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
        Unit::Angstrom,
        0,
    )
    .unwrap();
    let basis = Basis::from_name(&mol, "6-31g").unwrap();
    let first = run_rhf(&mol, &basis, &ScfOptions::default()).unwrap();
    let opts = ScfOptions { initial_density: Some(first.density.clone()), ..Default::default() };
    let second = run_rhf(&mol, &basis, &opts).unwrap();
    assert!((first.energy - second.energy).abs() < 1e-10);
    assert!(second.iterations <= 3, "restart took {} iterations", second.iterations);
    assert!(second.iterations < first.iterations);
}

#[test]
fn bad_inputs_are_errors() {
    // Odd electron count.
    let mol = Molecule::from_symbols(&["H"], &[[0.0; 3]], Unit::Bohr, 0).unwrap();
    let basis = Basis::from_name(&mol, "sto-3g").unwrap();
    assert!(run_rhf(&mol, &basis, &ScfOptions::default()).is_err());
    // Wrong initial density shape.
    let h2 = Molecule::from_symbols(&["H", "H"], &[[0.0; 3], [0.0, 0.0, 1.4]], Unit::Bohr, 0).unwrap();
    let basis = Basis::from_name(&h2, "sto-3g").unwrap();
    let opts = ScfOptions { initial_density: Some(qc_rs::Matrix::zeros(3, 3)), ..Default::default() };
    assert!(run_rhf(&h2, &basis, &opts).is_err());
    // Iteration limit.
    let opts = ScfOptions { max_iterations: 1, ..Default::default() };
    assert!(matches!(run_rhf(&h2, &basis, &opts), Err(qc_rs::QcError::ScfNotConverged { .. })));
    // Unknown basis.
    assert!(Basis::from_name(&h2, "cc-pvdz").is_err());
}

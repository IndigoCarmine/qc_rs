//! RKS single points compared against PySCF reference values
//! (tests/reference/pyscf_reference.json; PySCF used an unpruned (150, 1202 Lebedev)
//! grid, we use the fine product grid of theory.typ `<sec:rks>`).

use qc_rs::basis::{Basis, BasisOptions, BasisSetDefinition};
use qc_rs::{Functional, GridSettings, Molecule, ScfOptions, Unit, dipole_moment, run_rks};
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

fn molecule(entry: &Value) -> Molecule {
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
    Molecule::from_symbols(&symbols, &coords, Unit::Bohr, 0).unwrap()
}

/// Tracks the largest deviation per quantity and every tolerance violation.
#[derive(Default)]
struct Report {
    maxima: Vec<(&'static str, f64)>,
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, case: &str, what: &'static str, dev: f64, tol: f64) -> f64 {
        match self.maxima.iter_mut().find(|(n, _)| *n == what) {
            Some((_, m)) => *m = m.max(dev),
            None => self.maxima.push((what, dev)),
        }
        if dev.is_nan() || dev > tol {
            self.failures.push(format!("{case}: {what} deviates by {dev:.3e} (tol {tol:.0e})"));
        }
        dev
    }
}

#[test]
fn rks_single_points_match_pyscf() {
    let reference = load_reference();
    let mut report = Report::default();
    let mut n_cases = 0;
    // The reference lists the pure components (Slater, VWN, B88, LYP combinations)
    // before the hybrids, so a wrong component shows up first in its own case.
    for entry in reference["single_points"].as_array().unwrap() {
        let method = entry["method"].as_str().unwrap();
        if method == "rhf" {
            continue;
        }
        n_cases += 1;
        let name = entry["name"].as_str().unwrap();
        let basis_name = entry["basis"].as_str().unwrap();
        let cart = entry["cartesian"].as_bool().unwrap();
        let case = format!("{name}/{basis_name}/{method}");
        let functional = Functional::from_name(method).unwrap();

        let mol = molecule(entry);
        let def = BasisSetDefinition::builtin(basis_name).unwrap();
        let basis = Basis::new(&mol, &def, BasisOptions { cartesian_d: cart }).unwrap();
        assert_eq!(basis.nao() as u64, entry["nao"].as_u64().unwrap(), "{case}: nao");

        // PySCF converged to conv_tol 1e-12 / conv_tol_grad 1e-8. With our default
        // max|e| < 1e-7 the energy is converged (it is variational) but e1 and coul
        // still carry first-order density errors of up to ~8e-7; converging the
        // commutator to 1e-10 removes those, so the printed deviations reflect the
        // grid difference rather than SCF convergence.
        let opts = ScfOptions { error_tolerance: 1e-10, ..Default::default() };
        let t0 = Instant::now();
        let res = run_rks(&mol, &basis, &functional, GridSettings::fine(), &opts).unwrap();
        let elapsed = t0.elapsed();
        let scf = &res.scf;
        let c = &scf.components;
        let summary = &entry["scf_summary"];

        report.check(&case, "e_tot", (scf.energy - entry["e_tot"].as_f64().unwrap()).abs(), 1e-6);
        let dmo = report.check(&case, "mo_energy", max_dev(&scf.orbital_energies, &f64s(&entry["mo_energy"])), 1e-5);
        let de1 = report.check(&case, "scf_summary.e1", (c.one_electron - summary["e1"].as_f64().unwrap()).abs(), 1e-5);
        let dcoul =
            report.check(&case, "scf_summary.coul", (c.coulomb - summary["coul"].as_f64().unwrap()).abs(), 1e-5);
        // PySCF's "exc" is E_xc plus the exact-exchange energy -(a_HF/4) Tr(PK), for
        // CAM-B3LYP -(1/4) Tr(P K^x) with the kernel [α + β erf(ωr)]/r (eq:cam-fock).
        let dexc =
            report.check(&case, "scf_summary.exc", (c.xc + c.exchange - summary["exc"].as_f64().unwrap()).abs(), 1e-5);
        let mu = dipole_moment(&mol, &basis, &scf.density);
        let ddip = report.check(&case, "dipole_au", max_dev(&mu, &f64s(&entry["dipole_au"])), 1e-5);
        let nel = res.electron_count.expect("DFT has a grid");
        let dn = report.check(&case, "electron_count", (nel - mol.n_electrons() as f64).abs(), 1e-6);

        println!(
            "{case:26} pts={:7} iter={:3} E={:.10} dE={:+.2e} dMO={:.1e} de1={:.1e} dJ={:.1e} dExc={:.1e} \
             dmu={:.1e} dN={:.1e} time={:.2}s",
            res.n_grid_points,
            scf.iterations,
            scf.energy,
            scf.energy - entry["e_tot"].as_f64().unwrap(),
            dmo,
            de1,
            dcoul,
            dexc,
            ddip,
            dn,
            elapsed.as_secs_f64()
        );
    }
    assert_eq!(n_cases, 19, "expected 19 RKS reference entries (13 + 6 CAM-B3LYP)");
    println!("max deviations over all RKS cases:");
    for (what, m) in &report.maxima {
        println!("  {what:18} {m:.3e}");
    }
    assert!(report.failures.is_empty(), "failures:\n{}", report.failures.join("\n"));
}

/// Grid convergence: B3LYP total energies of H2O/6-31G and H2CO/6-31G on the
/// default, fine and two other grids, relative to PySCF.
#[test]
fn grid_convergence_b3lyp() {
    let reference = load_reference();
    for name in ["h2o", "h2co"] {
        let entry = reference["single_points"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["name"] == name && e["basis"] == "6-31g" && e["method"] == "b3lypg")
            .unwrap();
        let e_ref = entry["e_tot"].as_f64().unwrap();
        let mol = molecule(entry);
        let nelec = mol.n_electrons() as f64;
        let basis = Basis::from_name(&mol, "6-31g").unwrap();
        let f = Functional::b3lyp();
        let mut energies = Vec::new();
        for (label, grid) in [
            ("default (75, 18x36)", GridSettings::default()),
            ("(100, 24x48)", GridSettings { n_radial: 100, n_theta: 24 }),
            ("fine (150, 30x60)", GridSettings::fine()),
            ("(200, 36x72)", GridSettings { n_radial: 200, n_theta: 36 }),
        ] {
            let t0 = Instant::now();
            let r = run_rks(&mol, &basis, &f, grid, &ScfOptions::default()).unwrap();
            println!(
                "{name:5} {label:20} pts={:7} E={:.10} E-E_pyscf={:+.2e} N-Nelec={:+.2e} time={:.2}s",
                r.n_grid_points,
                r.scf.energy,
                r.scf.energy - e_ref,
                r.electron_count.unwrap() - nelec,
                t0.elapsed().as_secs_f64()
            );
            energies.push(r.scf.energy);
        }
        println!("{name:5} E(default) - E(fine) = {:+.2e}", energies[0] - energies[2]);
        assert!((energies[0] - energies[2]).abs() < 1e-5, "{name}: default grid far from fine grid");
        assert!((energies[2] - e_ref).abs() < 1e-6, "{name}: fine grid far from PySCF");
    }
}

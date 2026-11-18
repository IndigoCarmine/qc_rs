//! Geometry optimizations compared with PySCF + geomeTRIC
//! (tests/reference/pyscf_reference.json, "optimizations"). Energies and all
//! interatomic distances are compared; Cartesian coordinates are not (orientation differs).

use qc_rs::{Calculation, GridSettings, Molecule, OptimizerSettings, Unit};
use serde_json::Value;
use std::time::Instant;

fn load_reference() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/reference/pyscf_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("reference JSON")).expect("valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn distances(mol: &Molecule) -> Vec<Vec<f64>> {
    let a = mol.atoms();
    a.iter()
        .map(|x| {
            a.iter()
                .map(|y| (0..3).map(|k| (x.position[k] - y.position[k]).powi(2)).sum::<f64>().sqrt())
                .collect()
        })
        .collect()
}

/// Convergence well beyond the defaults of theory.typ section 11 (all four thresholds
/// 100× tighter: max|g| < 4.5e-6 Hartree/Bohr), since PySCF/geomeTRIC was converged to
/// max|g| ~ 1e-7–1e-6.
fn tight_settings() -> OptimizerSettings {
    let d = OptimizerSettings::default();
    OptimizerSettings {
        max_gradient: d.max_gradient / 100.0,
        rms_gradient: d.rms_gradient / 100.0,
        max_step: d.max_step / 100.0,
        rms_step: d.rms_step / 100.0,
        ..d
    }
}

#[test]
fn optimizations_match_pyscf() {
    let reference = load_reference();
    let mut failures = Vec::new();
    let entries = reference["optimizations"].as_array().unwrap();
    assert_eq!(entries.len(), 5);
    for entry in entries {
        let name = entry["name"].as_str().unwrap();
        let basis = entry["basis"].as_str().unwrap();
        let method = entry["method"].as_str().unwrap();
        let case = format!("{name}/{basis}/{method}");
        let symbols: Vec<&str> = entry["atoms"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
        let coords: Vec<[f64; 3]> = entry["initial_coords_bohr"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let v = f64s(c);
                [v[0], v[1], v[2]]
            })
            .collect();
        let mol = Molecule::from_symbols(&symbols, &coords, Unit::Bohr, 0).unwrap();
        let is_hf = method == "rhf";
        // PySCF used a (99, 590) grid for the DFT optimizations (B3LYP and CAM-B3LYP);
        // we use our fine grid.
        let calc = Calculation::new(&mol, if is_hf { "hf" } else { method }, basis)
            .unwrap()
            .with_grid(GridSettings::fine());
        let t0 = Instant::now();
        let opt = calc.optimize(&tight_settings()).unwrap();
        let elapsed = t0.elapsed().as_secs_f64();
        assert!(opt.converged);
        let e_ref = entry["e_opt"].as_f64().unwrap();
        let de = opt.energy - e_ref;
        let dref: Vec<Vec<f64>> = entry["final_distances_bohr"].as_array().unwrap().iter().map(f64s).collect();
        let dours = distances(&opt.molecule);
        let mut ddist: f64 = 0.0;
        for (r1, r2) in dours.iter().zip(&dref) {
            for (a, b) in r1.iter().zip(r2) {
                ddist = ddist.max((a - b).abs());
            }
        }
        let gmax = opt.gradient.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        let rejected = opt.history.iter().filter(|s| !s.accepted).count();
        println!(
            "{case:22} steps={:3} evals={:3} (rejected {rejected}) E={:.10} E-E_pyscf={de:+.2e} max|dR_ij|={ddist:.2e} \
             max|g|={gmax:.1e} (pyscf {:.1e}) time={elapsed:.1}s",
            opt.steps,
            opt.evaluations,
            opt.energy,
            entry["final_max_abs_gradient"].as_f64().unwrap()
        );
        let etol = if is_hf { 1e-7 } else { 5e-6 };
        if de.abs() > etol {
            failures.push(format!("{case}: energy deviates by {de:.3e} (tol {etol:.0e})"));
        }
        if ddist > 1e-3 {
            failures.push(format!("{case}: distances deviate by {ddist:.3e} Bohr (tol 1e-3)"));
        }
    }
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
}

/// Default settings converge too, and a non-converging run is reported, not hidden.
#[test]
fn default_settings_and_step_limit() {
    let mol = Molecule::from_symbols(
        &["O", "H", "H"],
        &[[0.0, 0.0, 0.0], [0.0, 0.9, 0.6], [0.0, -0.8, 0.5]],
        Unit::Angstrom,
        0,
    )
    .unwrap();
    let calc = Calculation::new(&mol, "hf", "sto-3g").unwrap();
    let opt = calc.optimize(&OptimizerSettings::default()).unwrap();
    println!("H2O/STO-3G default settings: {} steps, E = {:.8}", opt.steps, opt.energy);
    assert!(opt.converged && opt.energy < calc.energy().unwrap().energy);
    let limited = OptimizerSettings { max_steps: 2, ..OptimizerSettings::default() };
    assert!(matches!(calc.optimize(&limited), Err(qc_rs::QcError::OptimizationNotConverged { steps: 2, .. })));
    let flagged = calc.optimize(&OptimizerSettings { error_if_not_converged: false, ..limited }).unwrap();
    assert!(!flagged.converged && flagged.steps == 2);
}

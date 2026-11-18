//! Every molecule in tests/molecules/*.xyz against PySCF
//! (tests/reference/molecules_reference.json).
//!
//! The directory is scanned at run time, so adding a test molecule needs no change
//! here: drop the XYZ file into tests/molecules/ and regenerate the reference with
//! tests/reference/gen_molecule_reference.py. A molecule without a reference entry,
//! or whose geometry differs from the one the reference was computed for, fails the
//! test instead of being skipped.
//!
//! Set QC_TEST_MOLECULES=name1,name2 (file stems) to run only some of them.
//!
//! RHF of every second molecule (the 1st, 3rd, ... in name order) runs with the
//! integral-direct strategy (theory.typ sec:direct), the rest with the stored
//! integrals, so that both are checked against PySCF without doubling the run time.

use qc_rs::{Calculation, IntegralStrategy, Molecule};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

/// Maximum deviations allowed from PySCF, per method: (energy in Hartree, gradient in
/// Hartree/Bohr). The DFT tolerances are set by the integration grids: qc_rs uses
/// its default grid here and PySCF a (99, 590) Lebedev grid (observed for 19-24 atoms:
/// 3e-6 Hartree and 3e-5 Hartree/Bohr; RHF agrees to 1e-11 and 2e-9).
fn tolerances(method: &str) -> (f64, f64) {
    if method == "rhf" { (1e-7, 1e-6) } else { (2e-5, 1e-4) }
}

fn molecules_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("molecules")
}

/// File stem -> XYZ text for every *.xyz in tests/molecules, sorted by name.
fn xyz_files() -> BTreeMap<String, String> {
    let dir = molecules_dir();
    let mut out = BTreeMap::new();
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("xyz")) {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            out.insert(name, std::fs::read_to_string(&path).unwrap());
        }
    }
    out
}

fn reference() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/reference/molecules_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("molecules_reference.json")).expect("valid JSON")
}

const REGENERATE: &str = "run tests/reference/gen_molecule_reference.py";

/// Checks that the reference entry was computed for exactly this geometry.
fn same_geometry(mol: &Molecule, entry: &Value) -> Result<(), String> {
    let atoms = entry["atoms"].as_array().ok_or("reference entry has no atoms")?;
    if atoms.len() != mol.atoms().len() {
        return Err(format!("{} atoms in the XYZ file, {} in the reference", mol.atoms().len(), atoms.len()));
    }
    if entry["charge"].as_i64() != Some(mol.charge() as i64) {
        return Err(format!("charge {} in the XYZ file, {} in the reference", mol.charge(), entry["charge"]));
    }
    for (i, (atom, symbol)) in mol.atoms().iter().zip(atoms).enumerate() {
        let z = symbol.as_str().and_then(qc_rs::molecule::atomic_number);
        if z != Some(atom.z) {
            return Err(format!("atom {} is Z={} in the XYZ file, {symbol} in the reference", i + 1, atom.z));
        }
        for k in 0..3 {
            let r = entry["coords_bohr"][i][k].as_f64().ok_or("bad coords_bohr")?;
            if (r - atom.position[k]).abs() > 1e-8 {
                return Err(format!("atom {} moved by {:.2e} Bohr since the reference was made", i + 1, (r - atom.position[k]).abs()));
            }
        }
    }
    Ok(())
}

#[test]
fn every_xyz_molecule_matches_pyscf() {
    let files = xyz_files();
    assert!(!files.is_empty(), "no XYZ files in {}", molecules_dir().display());
    let reference = reference();
    let entries = reference["molecules"].as_object().expect("molecules object");
    let only: Option<Vec<String>> =
        std::env::var("QC_TEST_MOLECULES").ok().map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    if let Some(only) = &only {
        for name in only {
            assert!(files.contains_key(name), "QC_TEST_MOLECULES names '{name}', but tests/molecules/{name}.xyz does not exist");
        }
    }

    let mut failures = Vec::new();
    let mut checked = 0;
    // A reference entry whose XYZ file is gone means the two are out of sync.
    for name in entries.keys() {
        if !files.contains_key(name) {
            failures.push(format!("{name}: reference entry without tests/molecules/{name}.xyz; {REGENERATE}"));
        }
    }
    for (index, (name, text)) in files.iter().enumerate() {
        if only.as_ref().is_some_and(|o| !o.contains(name)) {
            continue;
        }
        let mol = match Molecule::from_xyz(text) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{name}: cannot read the XYZ file: {e}"));
                continue;
            }
        };
        let Some(entry) = entries.get(name) else {
            failures.push(format!("{name}: no reference entry; {REGENERATE}"));
            continue;
        };
        if let Err(why) = same_geometry(&mol, entry) {
            failures.push(format!("{name}: {why}; {REGENERATE}"));
            continue;
        }
        for result in entry["results"].as_array().expect("results") {
            let (method, basis) = (result["method"].as_str().unwrap(), result["basis"].as_str().unwrap());
            let case = format!("{name} {method}/{basis}");
            let (tol_e, tol_g) = tolerances(method);
            let start = Instant::now();
            // PySCF's "rhf" is our "hf"; the xc names are accepted as they are.
            let ours = if method == "rhf" { "hf" } else { method };
            let strategy = if method == "rhf" && index % 2 == 0 { IntegralStrategy::Direct } else { IntegralStrategy::Auto };
            let res = match Calculation::new(&mol, ours, basis).and_then(|c| c.with_integral_strategy(strategy).gradient()) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("{case}: calculation failed: {e}"));
                    continue;
                }
            };
            checked += 1;
            let nao = res.scf.coefficients.rows();
            let de = (res.energy - result["e_tot"].as_f64().unwrap()).abs();
            let mut dg: f64 = 0.0;
            let mut gmax: f64 = 0.0;
            for (i, g) in res.gradient.iter().enumerate() {
                for k in 0..3 {
                    let r = result["gradient"][i][k].as_f64().unwrap();
                    dg = dg.max((g[k] - r).abs());
                    gmax = gmax.max(r.abs());
                }
            }
            println!(
                "{case:32} nao={nao:4} {:?} it={:3} E={:.8} |dE|={de:.2e} max|g|={gmax:.3} max|dg|={dg:.2e} ({:.1} s)",
                res.scf.integral_strategy,
                res.scf.iterations,
                res.energy,
                start.elapsed().as_secs_f64()
            );
            if strategy == IntegralStrategy::Direct && res.scf.integral_strategy != IntegralStrategy::Direct {
                failures.push(format!("{case}: the direct strategy was not used"));
            }
            if result["nao"].as_u64() != Some(nao as u64) {
                failures.push(format!("{case}: nao {nao} vs {}", result["nao"]));
            }
            if de > tol_e {
                failures.push(format!("{case}: |dE| = {de:.2e} > {tol_e:.0e}"));
            }
            if dg > tol_g {
                failures.push(format!("{case}: max|dg| = {dg:.2e} > {tol_g:.0e}"));
            }
        }
    }
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
    assert!(checked > 0, "nothing was checked");
}

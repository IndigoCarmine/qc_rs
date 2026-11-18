//! DFT-D3 two-body energies and gradients against dftd3 1.7.0
//! (tests/reference/d3_reference.json, made by tests/reference/gen_d3_reference.py
//! with no distance cutoff and no three-body term), for B3LYP and CAM-B3LYP with BJ
//! and zero damping, on every molecule of the reference file (all elements H–Ar and
//! tests/molecules/*.xyz). The analytic gradient (theory.typ eq:d3-grad) is also
//! checked against central differences of the energy.

use qc_rs::dispersion::{D3Damping, D3Parameters, D3Variant, dispersion_energy, dispersion_energy_and_gradient};
use qc_rs::{Molecule, QcError, Unit};
use serde_json::Value;
use std::path::PathBuf;

fn reference() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/reference/d3_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("d3_reference.json")).expect("valid JSON")
}

/// Relative tolerances |E - E_ref| / |E_ref|, set by the precision of the data, not
/// by the implementation. The embedded D3 data (data/dispersion/README.md) are
/// float32 values written as decimals, so every reference C6, r4r2, reference CN and
/// r0ab can differ from the values inside dftd3 by up to ~6e-8 relative.
///
/// * BJ: C6 carries up to 6e-8 and C8 = 3 C6 r4r2_A r4r2_B (eq:d3-c8) up to 3 × 6e-8;
///   the damping radius a1 sqrt(3 r4r2_A r4r2_B) + a2 enters with weight below one.
///   Bound ≈ 2e-7; observed: at most 3.8e-8 (B3LYP) and 2.7e-8 (CAM-B3LYP). An
///   absolute 1e-10 Hartree is not reachable: 6e-8 of the 0.05 Hartree of the
///   largest molecule (BarPh) is already 3e-9 (observed 5.2e-10).
/// * Zero: in addition, the tabulated R0^AB enters eq:d3-zero through
///   (r / (s_r R0))^(-α_n), and ∂ln f_n / ∂ln R0 = α_n (1 - f_n) ≤ α_8 = 16, so the
///   6e-8 of r0ab becomes up to ~1e-6. Observed: at most 6.9e-7 (both functionals),
///   with both signs, as rounding noise would give.
const REL_TOL_BJ: f64 = 2e-7;
const REL_TOL_ZERO: f64 = 1.5e-6;

/// (reference label, functional, variant)
const CASES: [(&str, &str, D3Variant); 4] = [
    ("b3lyp-d3bj", "b3lyp", D3Variant::Bj),
    ("cam-b3lyp-d3bj", "cam-b3lyp", D3Variant::Bj),
    ("b3lyp-d3zero", "b3lyp", D3Variant::Zero),
    ("cam-b3lyp-d3zero", "cam-b3lyp", D3Variant::Zero),
];

fn molecule(entry: &Value) -> Molecule {
    let symbols: Vec<&str> = entry["atoms"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    Molecule::from_symbols(&symbols, &vectors(&entry["coords_bohr"]), Unit::Bohr, 0).unwrap()
}

/// A JSON array of 3-vectors (coordinates or gradient).
fn vectors(v: &Value) -> Vec<[f64; 3]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|c| std::array::from_fn(|k| c[k].as_f64().unwrap()))
        .collect()
}

/// The reference was computed with exactly the parameters of `for_functional`.
fn check_parameters(label: &str, params: &D3Parameters, p: &Value) {
    let get = |k: &str| p[k].as_f64().unwrap_or_else(|| panic!("{label}: no '{k}' in the reference"));
    assert_eq!(get("s9"), 0.0, "{label}: the reference must have no three-body term");
    match params.damping {
        D3Damping::Bj { s6, a1, s8, a2 } => {
            assert_eq!(p["damping"], "bj");
            assert_eq!((s6, a1, s8, a2), (get("s6"), get("a1"), get("s8"), get("a2")), "{label}");
        }
        D3Damping::Zero { s6, sr6, s8, sr8, alpha6 } => {
            assert_eq!(p["damping"], "zero");
            assert_eq!((s6, sr6, s8, sr8, alpha6), (get("s6"), get("rs6"), get("s8"), get("rs8"), get("alp")), "{label}");
        }
        _ => panic!("{label}: unexpected damping"),
    }
}

fn rel_tol(variant: D3Variant) -> f64 {
    match variant {
        D3Variant::Bj => REL_TOL_BJ,
        D3Variant::Zero => REL_TOL_ZERO,
        _ => panic!("no tolerance for {variant:?}"),
    }
}

#[test]
fn energies_match_dftd3() {
    let r = reference();
    let molecules = r["molecules"].as_object().unwrap();
    assert!(molecules.len() >= 20, "reference has only {} molecules", molecules.len());
    let mut failures = Vec::new();
    for (label, func, variant) in CASES {
        let params = D3Parameters::for_functional(func, variant).unwrap();
        check_parameters(label, &params, &r["parameters"][label]);
        let tol = rel_tol(variant);
        for (name, entry) in molecules {
            let e_ref = entry["results"][label]["energy"].as_f64().unwrap();
            let e = dispersion_energy(&molecule(entry), &params).unwrap();
            let rel = (e - e_ref).abs() / e_ref.abs();
            println!("{label:16} {name:14} E = {e:.12e}  ref {e_ref:.12e}  rel {rel:.1e}");
            // A NaN energy must fail too.
            if rel.is_nan() || rel > tol {
                failures.push(format!("{label} {name}: {e:.15e} vs {e_ref:.15e} (rel {rel:.2e} > {tol:e})"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Size of the force of one atom pair (direct and CN terms of eq:d3-grad) in the
/// reference molecules, used to turn the relative data error into an absolute
/// gradient tolerance. The largest gradient component in d3_reference.json is
/// 7.1e-4 Hartree/Bohr (BarPh, zero damping).
const PAIR_FORCE_SCALE: f64 = 1e-3;

/// Gradient against dftd3: |g - g_ref| ≤ rel_tol × PAIR_FORCE_SCALE per component,
/// with the relative tolerance of the energy. Same argument as for the energy: each
/// pair term of eq:d3-grad is proportional to C6 (data error ~6e-8 relative) and, for
/// zero damping, depends on R0^AB through (r / (s_r R0))^(-α_n) with a sensitivity
/// of order α_n. So the error of a component is the relative data error times the
/// pair forces that add up to it, not times the component itself: a component can
/// be a near-cancelling sum (SiH4 with zero damping: error 1.7e-6 of its largest
/// component, although every pair term is within the energy tolerance), or zero by
/// symmetry. Tolerances: BJ 2e-10, zero 1.5e-9 Hartree/Bohr. Observed: at most
/// 2.2e-11 (BJ, BarPh) and 6.4e-10 (zero, BarPh).
#[test]
fn gradients_match_dftd3() {
    let r = reference();
    let molecules = r["molecules"].as_object().unwrap();
    let mut failures = Vec::new();
    for (label, func, variant) in CASES {
        let params = D3Parameters::for_functional(func, variant).unwrap();
        let tol = rel_tol(variant) * PAIR_FORCE_SCALE;
        for (name, entry) in molecules {
            let g_ref = vectors(&entry["results"][label]["gradient"]);
            let (_, g) = dispersion_energy_and_gradient(&molecule(entry), &params).unwrap();
            assert_eq!(g.len(), g_ref.len(), "{label} {name}: number of atoms");
            // A NaN component must fail too.
            let worst = g.iter().flatten().zip(g_ref.iter().flatten()).fold(0.0_f64, |m, (a, b)| {
                let d = (a - b).abs();
                if d.is_nan() || m.is_nan() { f64::NAN } else { m.max(d) }
            });
            println!("{label:16} {name:14} max |g - g_ref| = {worst:.2e}");
            if worst.is_nan() || worst > tol {
                failures.push(format!("{label} {name}: max |g - g_ref| = {worst:.3e} > {tol:e} Hartree/Bohr"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Central differences of the energy, h = 1e-4 Bohr: truncation error h²/6 E''' and
/// rounding ~1e-16 / h = 1e-12 Hartree/Bohr. Observed: at most 6e-12.
#[test]
fn gradients_match_central_differences() {
    let r = reference();
    let h = 1e-4;
    for name in ["water_dimer", "PhNHCOCH3"] {
        let mol = molecule(&r["molecules"][name]);
        for (label, func, variant) in CASES {
            let params = D3Parameters::for_functional(func, variant).unwrap();
            let (e, g) = dispersion_energy_and_gradient(&mol, &params).unwrap();
            // The energy is bit for bit the one of dispersion_energy.
            assert_eq!(e, dispersion_energy(&mol, &params).unwrap(), "{label} {name}");
            let x = mol.coordinates();
            let mut worst = 0.0_f64;
            for i in 0..x.len() {
                let shifted = |s: f64| {
                    let mut y = x.clone();
                    y[i] += s;
                    dispersion_energy(&mol.with_coordinates(&y).unwrap(), &params).unwrap()
                };
                let num = (shifted(h) - shifted(-h)) / (2.0 * h);
                let d = (num - g[i / 3][i % 3]).abs();
                assert!(d.is_finite(), "{label} {name}: coordinate {i}");
                worst = worst.max(d);
            }
            println!("{label:16} {name:14} max |analytic - numeric| = {worst:.2e}");
            assert!(worst < 1e-8, "{label} {name}: max |analytic - numeric| = {worst:e}");
        }
    }
}

/// Translational invariance (theory.typ, below eq:d3-grad): Σ_A dE/dR_A = 0.
#[test]
fn gradient_sums_to_zero() {
    let r = reference();
    for (name, entry) in r["molecules"].as_object().unwrap() {
        for (label, func, variant) in CASES {
            let params = D3Parameters::for_functional(func, variant).unwrap();
            let (_, g) = dispersion_energy_and_gradient(&molecule(entry), &params).unwrap();
            for k in 0..3 {
                let s: f64 = g.iter().map(|v| v[k]).sum();
                assert!(s.abs() < 1e-12, "{label} {name}: Σ_A dE/dR_A[{k}] = {s:e}");
            }
        }
    }
}

/// Every tests/molecules/*.xyz must have a D3 reference entry for the same geometry,
/// so that a molecule added there is not silently left out of the tests above.
#[test]
fn every_test_molecule_has_a_reference() {
    let r = reference();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("molecules");
    let mut n = 0;
    for f in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display())) {
        let path = f.unwrap().path();
        if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("xyz")) {
            continue;
        }
        n += 1;
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let entry = &r["molecules"][&name];
        assert!(
            entry.is_object(),
            "tests/molecules/{name}.xyz has no entry in d3_reference.json; run tests/reference/gen_d3_reference.py"
        );
        let mol = Molecule::from_xyz(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let reference = molecule(entry);
        assert_eq!(mol.atoms().len(), reference.atoms().len(), "{name}: number of atoms");
        for (a, b) in mol.atoms().iter().zip(reference.atoms()) {
            assert_eq!(a.z, b.z, "{name}: elements differ");
            for k in 0..3 {
                assert!((a.position[k] - b.position[k]).abs() < 1e-9, "{name}: geometry differs from the reference");
            }
        }
    }
    assert!(n > 0, "no XYZ files in {}", dir.display());
}

#[test]
fn functional_names_and_variants() {
    let bj = D3Parameters::for_functional("B3LYP", D3Variant::Bj).unwrap();
    assert_eq!(D3Parameters::for_functional("b3lyp5", D3Variant::Bj).unwrap(), bj);
    assert_eq!(
        D3Parameters::for_functional("CAM-B3LYP", D3Variant::Zero).unwrap(),
        D3Parameters::for_functional("cam_b3lyp", D3Variant::Zero).unwrap()
    );
    assert_ne!(bj, D3Parameters::for_functional("b3lyp", D3Variant::Zero).unwrap());
    for name in ["pbe", "hf", "blyp", ""] {
        assert!(matches!(D3Parameters::for_functional(name, D3Variant::Bj), Err(QcError::InvalidInput(_))), "{name}");
    }
}

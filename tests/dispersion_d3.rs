//! DFT-D3 two-body energies against dftd3 1.7.0 (tests/reference/d3_reference.json,
//! made by tests/reference/gen_d3_reference.py with no distance cutoff and no
//! three-body term), for B3LYP and CAM-B3LYP with BJ and zero damping, on every
//! molecule of the reference file (all elements H–Ar and tests/molecules/*.xyz).

use qc_rs::dispersion::{D3Damping, D3Parameters, D3Variant, dispersion_energy};
use qc_rs::{Molecule, QcError, Unit};
use serde_json::Value;

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
    let coords: Vec<[f64; 3]> = entry["coords_bohr"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| std::array::from_fn(|k| c[k].as_f64().unwrap()))
        .collect();
    Molecule::from_symbols(&symbols, &coords, Unit::Bohr, 0).unwrap()
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

#[test]
fn energies_match_dftd3() {
    let r = reference();
    let molecules = r["molecules"].as_object().unwrap();
    assert!(molecules.len() >= 20, "reference has only {} molecules", molecules.len());
    let mut failures = Vec::new();
    for (label, func, variant) in CASES {
        let params = D3Parameters::for_functional(func, variant).unwrap();
        check_parameters(label, &params, &r["parameters"][label]);
        let tol = match variant {
            D3Variant::Bj => REL_TOL_BJ,
            D3Variant::Zero => REL_TOL_ZERO,
        };
        for (name, entry) in molecules {
            let e_ref = entry["results"][label]["energy"].as_f64().unwrap();
            let e = dispersion_energy(&molecule(entry), &params).unwrap();
            let rel = (e - e_ref).abs() / e_ref.abs();
            println!("{label:16} {name:14} E = {e:.12e}  ref {e_ref:.12e}  rel {rel:.1e}");
            // A NaN energy must fail too.
            if rel.is_nan() || rel > tol {
                failures.push(format!("{label} {name}: {e:.15e} vs {e_ref:.15e} (rel {rel:.2e} > {tol:.0e})"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
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

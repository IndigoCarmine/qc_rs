//! Harmonic analysis (theory.typ sec:freq: eq:mass-weight, eq:tr-vectors, eq:freq,
//! eq:reduced-mass) fed with the PySCF Hessians of tests/reference/pyscf_reference.json
//! ("hessians"), so that only the analysis is tested, not the SCF.

use qc_rs::{Matrix, Molecule, Unit, harmonic_analysis, standard_atomic_mass};
use serde_json::Value;

fn load_reference() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/reference/pyscf_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("reference JSON")).expect("valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(f64s).collect()
}

fn molecule(entry: &Value) -> Molecule {
    let symbols: Vec<&str> = entry["atoms"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    let coords: Vec<[f64; 3]> = rows(&entry["coords_bohr"]).iter().map(|c| [c[0], c[1], c[2]]).collect();
    Molecule::from_symbols(&symbols, &coords, Unit::Bohr, 0).unwrap()
}

fn label(entry: &Value) -> String {
    format!("{}/{}/{}", entry["name"], entry["basis"], entry["method"])
}

/// Projector onto the span of the columns `cols` of `v` (n × k), assumed orthonormal.
fn projector(v: &[Vec<f64>], cols: &[usize]) -> Matrix {
    let n = v.len();
    Matrix::from_fn(n, n, |i, j| cols.iter().map(|&k| v[i][k] * v[j][k]).sum())
}

/// Groups of indices of ascending `freqs` closer than `tol` to their neighbour.
fn degenerate_groups(freqs: &[f64], tol: f64) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (k, f) in freqs.iter().enumerate() {
        match groups.last_mut() {
            Some(g) if (f - freqs[*g.last().unwrap()]).abs() < tol => g.push(k),
            _ => groups.push(vec![k]),
        }
    }
    groups
}

/// The default masses equal the masses PySCF used (theory.typ eq:mass-weight).
#[test]
fn default_masses_match_pyscf() {
    let reference = load_reference();
    for entry in reference["hessians"].as_array().unwrap() {
        let mol = molecule(entry);
        for (atom, m) in mol.atoms().iter().zip(f64s(&entry["masses_u"])) {
            assert_eq!(standard_atomic_mass(atom.z), Some(m), "{} Z = {}", label(entry), atom.z);
        }
    }
}

/// Wavenumbers agree with PySCF to 1e-3 cm⁻¹ for every Hessian (HCN: 3N − 5 = 4 modes,
/// two degenerate); the mass-weighted normal modes span the same (degenerate)
/// subspaces, compared through projectors because signs and degenerate bases are
/// arbitrary.
#[test]
fn frequencies_and_modes_match_pyscf() {
    let reference = load_reference();
    let entries = reference["hessians"].as_array().unwrap();
    assert!(!entries.is_empty());
    let mut saw_linear = false;
    for entry in entries {
        let name = label(entry);
        let mol = molecule(entry);
        let n = 3 * mol.atoms().len();
        let h = rows(&entry["hessian"]);
        let hessian = Matrix::from_fn(n, n, |i, j| h[i][j]);
        let r = harmonic_analysis(&mol, &hessian, None).unwrap();

        let freq_ref = f64s(&entry["freq_wavenumber"]);
        assert_eq!(r.frequencies_cm1.len(), freq_ref.len(), "{name}");
        assert_eq!(r.n_trans_rot, n - freq_ref.len(), "{name}");
        if entry["name"] == "hcn" {
            assert!(r.is_linear && r.n_trans_rot == 5 && r.frequencies_cm1.len() == 4, "{name}");
            saw_linear = true;
        } else {
            assert!(!r.is_linear && r.n_trans_rot == 6, "{name}");
        }
        for (k, (a, b)) in r.frequencies_cm1.iter().zip(&freq_ref).enumerate() {
            assert!((a - b).abs() < 1e-3, "{name} mode {k}: {a} vs PySCF {b}");
        }

        // PySCF's norm_mode_cartesian rows are Cartesian displacements d_k; √m d_k = l_k.
        let masses = f64s(&entry["masses_u"]);
        let modes_ref = rows(&entry["norm_mode_cartesian"]);
        let l_ref: Vec<Vec<f64>> =
            (0..n).map(|i| modes_ref.iter().map(|d| d[i] * masses[i / 3].sqrt()).collect()).collect();
        let l: Vec<Vec<f64>> = (0..n).map(|i| (0..freq_ref.len()).map(|k| r.normal_modes[(i, k)]).collect()).collect();
        for group in degenerate_groups(&freq_ref, 1e-2) {
            let diff = (&projector(&l, &group) - &projector(&l_ref, &group)).max_abs();
            assert!(diff < 1e-6, "{name} modes {group:?}: projector difference {diff:e}");
            if group.len() == 1 {
                // Reduced mass is sign-independent (theory.typ eq:reduced-mass).
                let k = group[0];
                let mu_ref = 1.0 / modes_ref[k].iter().map(|x| x * x).sum::<f64>();
                assert!((r.reduced_masses[k] - mu_ref).abs() < 1e-6 * mu_ref, "{name} mode {k}");
            }
        }
    }
    assert!(saw_linear);
}

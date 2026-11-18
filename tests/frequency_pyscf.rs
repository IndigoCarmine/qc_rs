//! Harmonic analysis (theory.typ sec:freq: eq:mass-weight, eq:tr-vectors, eq:freq,
//! eq:reduced-mass) fed with the PySCF Hessians of tests/reference/pyscf_reference.json
//! ("hessians"), so that only the analysis is tested, not the SCF; and thermochemistry
//! (eq:thermo-trans … eq:thermo-total) fed with PySCF's wavenumbers.

use qc_rs::{Matrix, Molecule, ThermoSettings, Unit, harmonic_analysis, standard_atomic_mass, thermochemistry};
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

/// Absolute and relative difference of `a` from the reference `b`.
fn diff(a: f64, b: f64) -> (f64, f64) {
    ((a - b).abs(), (a - b).abs() / b.abs())
}

/// Thermochemistry (theory.typ eq:thermo-trans … eq:thermo-total) against PySCF's
/// thermo.thermo for every Hessian entry at both (T, p), with PySCF's wavenumbers,
/// masses, symmetry number and E0 as input.
///
/// PySCF's k_B T, read from its output as H − U, corresponds to E_h/k_B larger by
/// 3.3e-7 than CODATA 2018 (theory.typ, "PySCF との比較"). All temperature dependence
/// enters through k_B T, so we evaluate at the T' that gives PySCF's k_B T and compare
/// energies directly and S, C_v as T·S, T·C_v (E_h). T' / T − 1 is checked to be that
/// constant difference (measured −3.30e-7), so the rescaling cannot absorb a formula error.
///
/// Tolerances from the measured worst cases over all entries: |Δ| 7.1e-11 E_h (the
/// totals near −76 E_h, via the ZPE), relative 8.8e-9 (S_vib, a 1e-6 E_h term whose
/// sensitivity to Θ_k amplifies the ~1e-9 rounding of PySCF's stored wavenumbers),
/// Θ_a relative 9.1e-10. Limits: |Δ| ≤ 2e-10 E_h and relative ≤ 5e-8 for every
/// quantity, Θ_a relative ≤ 5e-9.
#[test]
fn thermochemistry_matches_pyscf() {
    let reference = load_reference();
    let mut saw_linear = false;
    for entry in reference["hessians"].as_array().unwrap() {
        let name = label(entry);
        let mol = molecule(entry);
        let masses = f64s(&entry["masses_u"]);
        let freq = f64s(&entry["freq_wavenumber"]);
        for t in entry["thermo"].as_array().unwrap() {
            let g = |k: &str| t[k].as_f64().unwrap_or_else(|| panic!("{name}: {k}"));
            let temp = g("temperature_k");
            let settings = ThermoSettings {
                temperature: temp,
                pressure: g("pressure_pa"),
                symmetry_number: t["sym_number"].as_u64().unwrap() as u32,
            };
            let at_t = thermochemistry(&mol, Some(&masses), &freq, g("E0"), &settings).unwrap();
            let kt_ref = g("H_tot") - g("E_tot");
            let t_prime = temp * kt_ref / (at_t.enthalpy - at_t.internal_energy);
            assert!((t_prime / temp - 1.0 + 3.30e-7).abs() < 1e-9, "{name}: T'/T − 1 = {:e}", t_prime / temp - 1.0);
            let r = thermochemistry(&mol, Some(&masses), &freq, g("E0"), &ThermoSettings { temperature: t_prime, ..settings })
                .unwrap();
            assert_eq!(r.n_excluded_modes, 0, "{name}");
            if entry["name"] == "hcn" {
                assert!(r.is_linear && r.rotational_temperatures.len() == 2, "{name}");
                saw_linear = true;
            } else {
                assert!(!r.is_linear && r.rotational_temperatures.len() == 3, "{name}");
            }

            let pairs = [
                ("E_trans", r.translation.energy, g("E_trans")),
                ("E_rot", r.rotation.energy, g("E_rot")),
                ("E_vib", r.vibration.energy, g("E_vib")),
                ("ZPE", r.zero_point_energy, g("ZPE")),
                ("E_0K", r.energy_at_0k, g("E_0K")),
                ("E_tot", r.internal_energy, g("E_tot")),
                ("H_tot", r.enthalpy, g("H_tot")),
                ("G_tot", r.gibbs_energy, g("G_tot")),
                ("S_trans", t_prime * r.translation.entropy, temp * g("S_trans")),
                ("S_rot", t_prime * r.rotation.entropy, temp * g("S_rot")),
                ("S_vib", t_prime * r.vibration.entropy, temp * g("S_vib")),
                ("S_tot", t_prime * r.entropy, temp * g("S_tot")),
                ("Cv_trans", t_prime * r.translation.heat_capacity, temp * g("Cv_trans")),
                ("Cv_rot", t_prime * r.rotation.heat_capacity, temp * g("Cv_rot")),
                ("Cv_vib", t_prime * r.vibration.heat_capacity, temp * g("Cv_vib")),
                ("Cv_tot", t_prime * r.heat_capacity_v, temp * g("Cv_tot")),
                ("Cp_tot", t_prime * r.heat_capacity_p, temp * g("Cp_tot")),
            ];
            for (key, a, b) in pairs {
                let (abs, rel) = diff(a, b);
                assert!(abs <= 2e-10 && rel <= 5e-8, "{name} T={temp} {key}: {a} vs PySCF {b} (abs {abs:e}, rel {rel:e})");
            }
            // Rotational constants (GHz; null for the zero moment of a linear molecule)
            // as temperatures Θ = h B / k_B (SI, exact h and k_B).
            let b_ghz: Vec<f64> = t["rot_const"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
            assert_eq!(b_ghz.len(), r.rotational_temperatures.len(), "{name}");
            for (b, theta) in b_ghz.iter().zip(&r.rotational_temperatures) {
                let (_, rel) = diff(*theta, b * 1e9 * 6.62607015e-34 / 1.380649e-23);
                assert!(rel <= 5e-9, "{name}: Θ {theta} vs PySCF B {b} GHz (rel {rel:e})");
            }
        }
    }
    assert!(saw_linear);
}

//! Analytic gradients: comparison with PySCF (tests/reference/pyscf_reference.json),
//! with central finite differences of our own energies, and translational /
//! rotational invariance.

use qc_rs::scf::run_scf;
use qc_rs::{
    Basis, Calculation, Functional, GridSettings, MolecularGrid, Molecule, ScfOptions, Unit, XcIntegrator,
    rks_gradient,
};
use serde_json::Value;
use std::time::Instant;

fn load_reference() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/reference/pyscf_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("reference JSON")).expect("valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn molecule(entry: &Value, key: &str) -> Molecule {
    let symbols: Vec<&str> = entry["atoms"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    let coords: Vec<[f64; 3]> = entry[key]
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

fn find<'a>(reference: &'a Value, name: &str, basis: &str, method: &str) -> &'a Value {
    reference["single_points"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == name && e["basis"] == basis && e["method"] == method && e["cartesian"] == false)
        .unwrap_or_else(|| panic!("no reference entry {name}/{basis}/{method}"))
}

/// |Σ_A g_A| (net force) and |Σ_A R_A × g_A| (net torque).
fn invariance(mol: &Molecule, g: &[[f64; 3]]) -> (f64, f64) {
    let mut f = [0.0; 3];
    let mut t = [0.0; 3];
    for (a, ga) in mol.atoms().iter().zip(g) {
        let r = a.position;
        for k in 0..3 {
            f[k] += ga[k];
        }
        t[0] += r[1] * ga[2] - r[2] * ga[1];
        t[1] += r[2] * ga[0] - r[0] * ga[2];
        t[2] += r[0] * ga[1] - r[1] * ga[0];
    }
    let n = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (n(f), n(t))
}

#[test]
fn gradients_match_pyscf() {
    let reference = load_reference();
    let mut failures = Vec::new();
    let (mut worst_rhf, mut worst_dft) = (0.0f64, 0.0f64);
    let mut n_cases = 0;
    for entry in reference["single_points"].as_array().unwrap() {
        let Some(gref) = entry.get("gradient") else { continue };
        n_cases += 1;
        let name = entry["name"].as_str().unwrap();
        let basis = entry["basis"].as_str().unwrap();
        let method = entry["method"].as_str().unwrap();
        let cart = entry["cartesian"].as_bool().unwrap();
        let case = format!("{name}/{basis}/{method}{}", if cart { "/cart" } else { "" });
        let mol = molecule(entry, "coords_bohr");
        let is_hf = method == "rhf";
        // PySCF: (150, 1202 Lebedev) unpruned grid, no grid response; ours: fine grid.
        let calc = Calculation::new(&mol, if is_hf { "hf" } else { method }, basis)
            .unwrap()
            .with_cartesian_d(cart)
            .with_grid(GridSettings::fine());
        let t0 = Instant::now();
        let res = calc.gradient().unwrap();
        let elapsed = t0.elapsed().as_secs_f64();
        let gref: Vec<f64> = gref.as_array().unwrap().iter().flat_map(f64s).collect();
        let dev = res.gradient.iter().flatten().zip(&gref).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let gmax = gref.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let (force, torque) = invariance(&mol, &res.gradient);
        let tol = if is_hf { 1e-6 } else { 1e-5 };
        if is_hf {
            worst_rhf = worst_rhf.max(dev);
        } else {
            worst_dft = worst_dft.max(dev);
        }
        println!(
            "{case:28} max|g|={gmax:.4} max|g - g_pyscf|={dev:.2e} |sum F|={force:.1e} |torque|={torque:.1e} \
             E-E_pyscf={:+.1e} time={elapsed:.2}s",
            res.energy - entry["e_tot"].as_f64().unwrap()
        );
        if dev.is_nan() || dev > tol {
            failures.push(format!("{case}: gradient deviates by {dev:.3e} (tol {tol:.0e})"));
        }
        if is_hf && (force > 1e-8 || torque > 1e-8) {
            failures.push(format!("{case}: RHF gradient not invariant (force {force:.1e}, torque {torque:.1e})"));
        }
    }
    assert_eq!(n_cases, 23, "expected 23 reference gradients");
    println!("max |g - g_pyscf|: RHF {worst_rhf:.2e}, DFT {worst_dft:.2e}");
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
}

/// Central differences of the energy (h = 1e-3 Bohr) on the asymmetric h2o_dist geometry.
///
/// For DFT the grid moves with the atoms in the finite differences, while the analytic
/// gradient neglects the grid-weight derivatives (theory.typ section 10.5); the
/// difference is reported for the default and the fine grid, together with the
/// translational non-invariance of the analytic DFT gradient, which has the same origin.
#[test]
fn gradients_match_finite_differences() {
    let reference = load_reference();
    let mol = molecule(find(&reference, "h2o_dist", "6-31g*", "rhf"), "coords_bohr");
    let tight = ScfOptions { error_tolerance: 1e-10, energy_tolerance: 1e-12, ..Default::default() };
    let h = 1e-3;
    for (method, basis, grid, tol) in [
        ("hf", "6-31g*", GridSettings::default(), 1e-6),
        ("b3lyp", "6-31g", GridSettings::default(), 2e-4),
        ("b3lyp", "6-31g", GridSettings::fine(), 2e-5),
    ] {
        let calc = Calculation::new(&mol, method, basis).unwrap().with_grid(grid).with_scf_options(tight.clone());
        let g = calc.gradient().unwrap().gradient;
        let x0 = mol.coordinates();
        let central = |i: usize, step: f64| {
            let mut e = [0.0; 2];
            for (s, sign) in [1.0, -1.0].iter().enumerate() {
                let mut x = x0.clone();
                x[i] += sign * step;
                e[s] = calc.clone().with_molecule(&mol.with_coordinates(&x).unwrap()).energy().unwrap().energy;
            }
            (e[0] - e[1]) / (2.0 * step)
        };
        let (mut worst, mut worst_rich): (f64, f64) = (0.0, 0.0);
        for i in 0..x0.len() {
            let fd = central(i, h);
            // Richardson extrapolation removes the O(h²) truncation error of the plain
            // central difference, exposing the remaining analytic-vs-energy mismatch.
            let rich = (4.0 * central(i, h / 2.0) - fd) / 3.0;
            worst = worst.max((fd - g[i / 3][i % 3]).abs());
            worst_rich = worst_rich.max((rich - g[i / 3][i % 3]).abs());
        }
        let (force, torque) = invariance(&mol, &g);
        let label = if method == "hf" { format!("RHF/{basis}") } else { format!("B3LYP/{basis} grid {grid:?}") };
        println!(
            "{label:62} max|analytic - FD(h=1e-3)| = {worst:.2e}, Richardson {worst_rich:.2e}; |sum F| = {force:.1e}, \
             |torque| = {torque:.1e}"
        );
        assert!(worst < tol, "{label}: analytic vs FD {worst:.2e} (tol {tol:.0e})");
    }
}

/// Timings of single gradients (printed): H2CO RHF/6-31G* and H2CO B3LYP/6-31G on the fine grid.
#[test]
fn gradient_timings() {
    let reference = load_reference();
    for (basis, method, grid) in [("6-31g*", "rhf", GridSettings::default()), ("6-31g", "b3lypg", GridSettings::fine())] {
        let entry = find(&reference, "h2co", basis, method);
        let mol = molecule(entry, "coords_bohr");
        let functional = Functional::from_name(if method == "rhf" { "hf" } else { method }).unwrap();
        let t0 = Instant::now();
        let b = Basis::from_name(&mol, basis).unwrap();
        let mut xc = if functional.has_density_part() {
            XcIntegrator::new(&mol, &b, &functional, grid).unwrap()
        } else {
            let empty = MolecularGrid { points: vec![], weights: vec![], blocks: vec![], settings: grid };
            XcIntegrator::with_grid(&b, &functional, empty, Default::default()).unwrap()
        };
        let opts = ScfOptions { error_tolerance: 1e-10, ..Default::default() };
        let scf = run_scf(&mol, &b, &mut xc, &opts).unwrap();
        let t_scf = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let g = rks_gradient(&mol, &b, &scf, &xc).unwrap();
        let t_grad = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let _ = qc_rs::gradient::two_electron_gradient(&mol, &b, &scf.density, functional.exact_exchange_fraction());
        let t_2e = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let _ = qc_rs::gradient::xc_gradient(&mol, &b, &xc, &scf.density);
        let t_xc = t0.elapsed().as_secs_f64();
        println!(
            "h2co/{basis}/{method} (nao {}, {} grid points): grid+SCF {t_scf:.2}s ({} iterations), gradient {t_grad:.2}s \
             (two-electron {t_2e:.2}s, xc {t_xc:.2}s), max|g| {:.4}",
            b.nao(),
            xc.grid().len(),
            scf.iterations,
            g.total().iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()))
        );
    }
}

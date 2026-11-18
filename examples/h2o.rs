//! Water: RHF/6-31G* and B3LYP/6-31G single points, analytic gradients and
//! geometry optimizations.
//!
//! Run with `cargo run --release --example h2o`.

use qc_rs::molecule::BOHR_IN_ANGSTROM;
use qc_rs::{Calculation, Molecule, OptimizerSettings, Result, Unit};
use std::time::Instant;

/// O–H bond lengths (Å) and H–O–H angle (degrees); atom 0 is O.
fn geometry(mol: &Molecule) -> (f64, f64, f64) {
    let p: Vec<[f64; 3]> = mol.atoms().iter().map(|a| a.position).collect();
    let v1: Vec<f64> = (0..3).map(|k| p[1][k] - p[0][k]).collect();
    let v2: Vec<f64> = (0..3).map(|k| p[2][k] - p[0][k]).collect();
    let n = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cos = v1.iter().zip(&v2).map(|(a, b)| a * b).sum::<f64>() / (n(&v1) * n(&v2));
    (n(&v1) * BOHR_IN_ANGSTROM, n(&v2) * BOHR_IN_ANGSTROM, cos.acos().to_degrees())
}

fn print_vectors(label: &str, g: &[[f64; 3]]) {
    println!("  {label} (Hartree/Bohr):");
    let symbols = ["O", "H", "H"];
    for (i, gi) in g.iter().enumerate() {
        let s = symbols.get(i).copied().unwrap_or("?");
        println!("    {s} {:>14.8} {:>14.8} {:>14.8}", gi[0], gi[1], gi[2]);
    }
}

fn main() -> Result<()> {
    // A distorted water molecule (Å).
    let mol = Molecule::from_symbols(
        &["O", "H", "H"],
        &[[0.0, 0.0, 0.10], [0.0, 0.80, -0.45], [0.0, -0.70, -0.52]],
        Unit::Angstrom,
        0,
    )?;
    let (r1, r2, a) = geometry(&mol);
    println!("start geometry: r(OH) = {r1:.4} / {r2:.4} Å, angle(HOH) = {a:.2}°\n");

    for (method, basis) in [("hf", "6-31g*"), ("b3lyp", "6-31g")] {
        let label = format!("{}/{}", method.to_uppercase(), basis.to_uppercase());
        println!("=== {label} ===");
        let calc = Calculation::new(&mol, method, basis)?;

        let t0 = Instant::now();
        let sp = calc.energy()?;
        print!("  single-point energy: {:.10} Eh  ({} SCF iterations", sp.energy, sp.scf.iterations);
        if let Some(n) = sp.electron_count {
            print!(", {} grid points, ∫ρ = {n:.6}", sp.n_grid_points);
        }
        println!(", {:.2} s)", t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        let gr = calc.gradient()?;
        println!("  energy + gradient: {:.2} s", t0.elapsed().as_secs_f64());
        print_vectors("gradient dE/dR", &gr.gradient);

        let t0 = Instant::now();
        let opt = calc.optimize(&OptimizerSettings::default())?;
        let (r1, r2, a) = geometry(&opt.molecule);
        let gmax = opt.gradient.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        println!(
            "  optimized in {} steps ({} evaluations, {:.2} s): E = {:.10} Eh, max|g| = {gmax:.1e}",
            opt.steps,
            opt.evaluations,
            t0.elapsed().as_secs_f64(),
            opt.energy
        );
        println!("  r(OH) = {r1:.4} / {r2:.4} Å, angle(HOH) = {a:.2}°");
        println!("  final coordinates (Å):");
        for (s, at) in ["O", "H", "H"].iter().zip(opt.molecule.atoms()) {
            let x = at.position.map(|v| v * BOHR_IN_ANGSTROM);
            println!("    {s} {:>12.6} {:>12.6} {:>12.6}", x[0], x[1], x[2]);
        }
        println!();
    }
    Ok(())
}

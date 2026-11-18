//! Times one SCF with the stored or the integral-direct two-electron integrals.
//!
//! ```sh
//! cargo run --release --example scf_benchmark -- <molecule> [options]
//! ```
//!
//! `<molecule>` is an XYZ file (Å) or one of the built-in `h2co`, `benzene`. Options:
//! `--method hf` (or a functional name), `--basis 6-31g*`, `--cart` (Cartesian d),
//! `--strategy stored|direct|auto`, `--threshold 1e-12` (direct screening),
//! `--no-incremental`, `--gradient`.

use qc_rs::{Calculation, IntegralStrategy, Molecule, ScfOptions, Unit};
use std::time::Instant;

fn builtin(name: &str) -> Option<Molecule> {
    match name {
        "h2co" => Some(
            Molecule::from_symbols(
                &["C", "O", "H", "H"],
                &[[0.0, 0.0, -0.53], [0.0, 0.0, 0.68], [0.0, 0.94, -1.11], [0.0, -0.94, -1.11]],
                Unit::Angstrom,
                0,
            )
            .unwrap(),
        ),
        "benzene" => {
            let mut symbols = Vec::new();
            let mut coords = Vec::new();
            for (sym, r) in [("C", 1.39), ("H", 2.47)] {
                for k in 0..6 {
                    let phi = k as f64 * std::f64::consts::PI / 3.0;
                    symbols.push(sym);
                    coords.push([r * phi.cos(), r * phi.sin(), 0.0]);
                }
            }
            Some(Molecule::from_symbols(&symbols, &coords, Unit::Angstrom, 0).unwrap())
        }
        _ => None,
    }
}

fn main() -> qc_rs::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().expect("usage: scf_benchmark <molecule> [options]").clone();
    let mol = match builtin(&name) {
        Some(m) => m,
        None => Molecule::from_xyz(&std::fs::read_to_string(&name).expect("cannot read the XYZ file"))?,
    };
    let value = |flag: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone());
    let method = value("--method").unwrap_or_else(|| "hf".into());
    let basis = value("--basis").unwrap_or_else(|| "6-31g*".into());
    let mut opts = ScfOptions::default();
    opts.integral_strategy = match value("--strategy").as_deref() {
        Some("stored") => IntegralStrategy::Stored,
        Some("direct") => IntegralStrategy::Direct,
        _ => IntegralStrategy::Auto,
    };
    if let Some(t) = value("--threshold") {
        opts.direct_threshold = t.parse().expect("threshold");
    }
    opts.incremental_fock = !args.iter().any(|a| a == "--no-incremental");
    let calc = Calculation::new(&mol, &method, &basis)?
        .with_cartesian_d(args.iter().any(|a| a == "--cart"))
        .with_scf_options(opts);
    let t0 = Instant::now();
    let (energy, scf) = if args.iter().any(|a| a == "--gradient") {
        let g = calc.gradient()?;
        (g.energy, g.scf)
    } else {
        let sp = calc.energy()?;
        (sp.energy, sp.scf)
    };
    let total = t0.elapsed().as_secs_f64();
    println!(
        "{name} {method}/{basis}{} nao={} threads={} strategy={:?}",
        if args.iter().any(|a| a == "--cart") { " (6d)" } else { "" },
        scf.coefficients.rows(),
        rayon::current_num_threads(),
        scf.integral_strategy
    );
    println!("E = {energy:.10}  iterations = {}  total {total:.2} s", scf.iterations);
    if !scf.direct_builds.is_empty() {
        let b = &scf.direct_builds;
        let jk: f64 = b.iter().map(|s| s.seconds).sum();
        println!(
            "direct: {} J/K builds ({} incremental), {jk:.2} s in J/K, {:.2} s per build",
            b.len(),
            b.iter().filter(|s| s.incremental).count(),
            jk / b.len() as f64
        );
        for (i, s) in b.iter().enumerate() {
            println!(
                "  build {:2} {} skipped {:6.2}% of {} quartets, {:.3} s",
                i + 1,
                if s.incremental { "ΔP  " } else { "full" },
                100.0 * s.skipped_fraction(),
                s.quartets_total,
                s.seconds
            );
        }
    }
    Ok(())
}

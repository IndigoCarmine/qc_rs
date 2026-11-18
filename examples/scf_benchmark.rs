//! Times one SCF with the stored or the integral-direct two-electron integrals.
//!
//! ```sh
//! cargo run --release --example scf_benchmark -- <molecule> [options]
//! ```
//!
//! `<molecule>` is an XYZ file (Å) or one of the built-in `h2co`, `benzene`. Options:
//! `--method hf` (or a functional name), `--basis 6-31g*`, `--cart` (Cartesian d),
//! `--strategy stored|direct|auto`, `--threshold 1e-12` (direct screening),
//! `--no-incremental`, `--gradient`, `--repeat N` (run the calculation N times and
//! report the shortest wall time; the detailed output is from the last run).
//!
//! # Measuring the single-point energy of a 50-100 atom molecule
//!
//! The reference case is `examples/molecules/barph_acetanilide_2dimer.xyz` (86 atoms,
//! nao = 536 in 6-31G, E(HF) = -2383.5085396025, 22 iterations). Measure it as
//!
//! ```sh
//! cargo run --release --example scf_benchmark -- \
//!     examples/molecules/barph_acetanilide_2dimer.xyz --basis 6-31g --repeat 3
//! ```
//!
//! i.e. the energy only (no `--gradient`), `--basis 6-31g`, 24 threads (the rayon
//! default on the 24-thread reference machine; set `RAYON_NUM_THREADS=24` elsewhere),
//! and take the `min` time. Run it when no other jobs are running: a concurrent build
//! or calculation can slow a run by up to a factor of two, which the minimum over
//! repeats only partly removes.

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
    let mut opts = ScfOptions {
        integral_strategy: match value("--strategy").as_deref() {
            Some("stored") => IntegralStrategy::Stored,
            Some("direct") => IntegralStrategy::Direct,
            _ => IntegralStrategy::Auto,
        },
        ..ScfOptions::default()
    };
    if let Some(t) = value("--threshold") {
        opts.direct_threshold = t.parse().expect("threshold");
    }
    opts.incremental_fock = !args.iter().any(|a| a == "--no-incremental");
    let calc = Calculation::new(&mol, &method, &basis)?
        .with_cartesian_d(args.iter().any(|a| a == "--cart"))
        .with_scf_options(opts);
    let repeat: usize = value("--repeat").map_or(1, |n| n.parse().expect("--repeat takes a positive integer"));
    assert!(repeat >= 1, "--repeat takes a positive integer");
    let mut times = Vec::with_capacity(repeat);
    let mut result = None;
    for _ in 0..repeat {
        let t0 = Instant::now();
        result = Some(if args.iter().any(|a| a == "--gradient") {
            let g = calc.gradient()?;
            (g.energy, g.scf)
        } else {
            let sp = calc.energy()?;
            (sp.energy, sp.scf)
        });
        times.push(t0.elapsed().as_secs_f64());
    }
    let (energy, scf) = result.expect("at least one run");
    let total = times.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "{name} {method}/{basis}{} nao={} threads={} strategy={:?}",
        if args.iter().any(|a| a == "--cart") { " (6d)" } else { "" },
        scf.coefficients.rows(),
        rayon::current_num_threads(),
        scf.integral_strategy
    );
    if repeat == 1 {
        println!("E = {energy:.10}  iterations = {}  total {total:.2} s", scf.iterations);
    } else {
        let all: Vec<String> = times.iter().map(|t| format!("{t:.2}")).collect();
        println!(
            "E = {energy:.10}  iterations = {}  min {total:.2} s of {repeat} runs ({} s)",
            scf.iterations,
            all.join(", ")
        );
    }
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

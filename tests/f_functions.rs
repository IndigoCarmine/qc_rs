//! f functions (L = 3, theory.typ eq:sph-f) through the whole program: the overlap of
//! f shells, the AO values of the individual spherical f functions and their
//! derivatives (eq:ao-deriv), stored vs integral-direct SCF (also the erf-attenuated
//! integrals of CAM-B3LYP), and analytic RHF / B3LYP gradients vs central differences
//! of the energy. Small molecules only.

use qc_rs::ao_eval::{D_X, eval_ao, hessian_component};
use qc_rs::basis::primitive_norm;
use qc_rs::integrals::overlap;
use qc_rs::{
    Basis, BasisOptions, BasisSetDefinition, Calculation, GridSettings, IntegralStrategy, Molecule, ScfOptions, Unit,
};

/// Slightly distorted water, so that no symmetry hides an error.
fn water() -> Molecule {
    Molecule::from_symbols(
        &["O", "H", "H"],
        &[[0.02, -0.01, 0.10], [0.05, 0.80, -0.45], [-0.03, -0.70, -0.52]],
        Unit::Angstrom,
        0,
    )
    .unwrap()
}

/// Hydrogen fluoride, tilted against the axes.
fn hf_molecule() -> Molecule {
    Molecule::from_symbols(&["F", "H"], &[[0.0, 0.0, 0.0], [0.45, -0.38, 0.62]], Unit::Angstrom, 0).unwrap()
}

fn tight() -> ScfOptions {
    ScfOptions { error_tolerance: 1e-10, energy_tolerance: 1e-12, ..Default::default() }
}

fn max_dev(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// One primitive f shell (exponent `a`) on a single atom at `center`.
fn single_f(center: [f64; 3], a: f64, cart: bool) -> Basis {
    let mol = Molecule::from_symbols(&["H"], &[center], Unit::Bohr, -1).unwrap();
    let def = BasisSetDefinition::from_gbs("f", &format!("H 0\nF 1 1.00\n {a} 1.0\n****\n")).unwrap();
    Basis::new(&mol, &def, BasisOptions { cartesian_d: cart }).unwrap()
}

/// The overlap of the f shells of a built-in basis: unit diagonal, and the
/// one-centre block of one spherical f shell is I_7 (eq:sph-f is orthonormal).
#[test]
fn f_shell_overlap() {
    for name in ["6-31g(2df,p)", "6-311g(2df,2pd)", "6-311++g(3df,3pd)"] {
        for cart in [false, true] {
            let def = BasisSetDefinition::builtin(name).unwrap();
            let b = Basis::new(&water(), &def, BasisOptions { cartesian_d: cart }).unwrap();
            let s = overlap(&b);
            let f_shells: Vec<_> = b.shells().iter().filter(|sh| sh.l == 3).collect();
            assert!(!f_shells.is_empty(), "{name}");
            for sh in f_shells {
                assert_eq!(sh.n_functions(), if cart { 10 } else { 7 });
                let o = sh.ao_offset;
                for i in 0..sh.n_functions() {
                    assert!((s[(o + i, o + i)] - 1.0).abs() < 1e-12, "{name} cart={cart}: S[{i}][{i}]");
                    if !cart {
                        for j in 0..7 {
                            let expect = if i == j { 1.0 } else { 0.0 };
                            let v = s[(o + i, o + j)];
                            assert!((v - expect).abs() < 1e-12, "{name}: f block [{i}][{j}] = {v}");
                        }
                    }
                }
            }
        }
    }
}

/// The real solid harmonics of eq:sph-f written out as polynomials (theory.typ
/// derivation: each row with the monomials χ_lmn / c_lmn), in the order m = -3..3:
/// value and gradient of K_m p_m(x, y, z).
fn solid_harmonic_f(m: usize, r: [f64; 3]) -> (f64, [f64; 3]) {
    let [x, y, z] = r;
    let a = (5.0_f64 / 8.0).sqrt();
    let b = (3.0_f64 / 8.0).sqrt();
    let r15 = 15.0_f64.sqrt();
    match m {
        // f(-3) = a y (3x² - y²)
        0 => (a * y * (3.0 * x * x - y * y), [a * 6.0 * x * y, a * (3.0 * x * x - 3.0 * y * y), 0.0]),
        // f(-2) = √15 xyz
        1 => (r15 * x * y * z, [r15 * y * z, r15 * x * z, r15 * x * y]),
        // f(-1) = b y (4z² - x² - y²)
        2 => (
            b * y * (4.0 * z * z - x * x - y * y),
            [-2.0 * b * x * y, b * (4.0 * z * z - x * x - 3.0 * y * y), 8.0 * b * y * z],
        ),
        // f(0) = z (2z² - 3x² - 3y²) / 2
        3 => (
            0.5 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y),
            [-3.0 * x * z, -3.0 * y * z, 0.5 * (6.0 * z * z - 3.0 * x * x - 3.0 * y * y)],
        ),
        // f(+1) = b x (4z² - x² - y²)
        4 => (
            b * x * (4.0 * z * z - x * x - y * y),
            [b * (4.0 * z * z - 3.0 * x * x - y * y), -2.0 * b * x * y, 8.0 * b * x * z],
        ),
        // f(+2) = √15/2 z (x² - y²)
        5 => (0.5 * r15 * z * (x * x - y * y), [r15 * x * z, -r15 * y * z, 0.5 * r15 * (x * x - y * y)]),
        // f(+3) = a x (x² - 3y²)
        6 => (a * x * (x * x - 3.0 * y * y), [a * (3.0 * x * x - 3.0 * y * y), -6.0 * a * x * y, 0.0]),
        _ => unreachable!(),
    }
}

/// Every spherical f AO equals N e^{-a r²} K_m p_m(r) of eq:sph-f, with the m order
/// -3..3 (a swapped or rotated row would fail here, unlike the orthonormality test),
/// and so does its gradient; Cartesian f AOs are c_lmn N x^l y^m z^n e^{-a r²}.
#[test]
fn f_ao_values_match_solid_harmonics() {
    let center = [0.3, -0.2, 0.1];
    let a = 0.8;
    let n = primitive_norm(a, [3, 0, 0]);
    let points: Vec<[f64; 3]> =
        vec![[0.9, 0.4, -0.7], [-0.5, 1.1, 0.6], [0.2, -0.9, 1.3], [1.4, 0.3, 0.5], [-0.8, -0.6, -1.0]];
    let sph = eval_ao(&single_f(center, a, false), &points, 1);
    assert_eq!(sph.n_ao, 7);
    for (g, p) in points.iter().enumerate() {
        let r = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let e = (-a * r2).exp();
        for m in 0..7 {
            let (v, dv) = solid_harmonic_f(m, r);
            let expect = n * e * v;
            let got = sph.get(0, m)[g];
            assert!((got - expect).abs() < 1e-13, "f({}) at {p:?}: {got} vs {expect}", m as i64 - 3);
            for i in 0..3 {
                let expect = n * e * (dv[i] - 2.0 * a * r[i] * v);
                let got = sph.get(D_X + i, m)[g];
                assert!((got - expect).abs() < 1e-12, "d_{i} f({}) at {p:?}: {got} vs {expect}", m as i64 - 3);
            }
        }
    }
    // Cartesian: lexicographic xxx, xxy, …, zzz with c_lmn of eq:cart-norm.
    let cart = eval_ao(&single_f(center, a, true), &points, 0);
    assert_eq!(cart.n_ao, 10);
    let comps = qc_rs::basis::cartesian_components(3);
    for (g, p) in points.iter().enumerate() {
        let r = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
        let e = (-a * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2])).exp();
        for (k, &lmn) in comps.iter().enumerate() {
            let mono = r[0].powi(lmn[0] as i32) * r[1].powi(lmn[1] as i32) * r[2].powi(lmn[2] as i32);
            let expect = qc_rs::basis::cartesian_norm(lmn) * n * mono * e;
            assert!((cart.get(0, k)[g] - expect).abs() < 1e-13, "{lmn:?} at {p:?}");
        }
    }
}

/// First and second derivatives of contracted f AOs (spherical and Cartesian) vs
/// central differences of the values and of the first derivatives.
#[test]
fn f_ao_derivatives_match_finite_differences() {
    let def = BasisSetDefinition::builtin("6-31g(2df,p)").unwrap();
    let mol = hf_molecule();
    let points = [[0.4, -0.3, 0.9], [-0.6, 0.5, -0.2], [1.1, 0.8, 0.3]];
    let h = 1e-4;
    for cart in [false, true] {
        let b = Basis::new(&mol, &def, BasisOptions { cartesian_d: cart }).unwrap();
        let f_aos: Vec<usize> = b
            .shells()
            .iter()
            .filter(|s| s.l == 3)
            .flat_map(|s| s.ao_offset..s.ao_offset + s.n_functions())
            .collect();
        assert_eq!(f_aos.len(), if cart { 10 } else { 7 });
        let at = eval_ao(&b, &points, 2);
        for (g, p) in points.iter().enumerate() {
            for i in 0..3 {
                let shifted = |s: f64| {
                    let mut q = *p;
                    q[i] += s;
                    eval_ao(&b, &[q], 1)
                };
                let (plus, minus) = (shifted(h), shifted(-h));
                for &mu in &f_aos {
                    let fd = (plus.get(0, mu)[0] - minus.get(0, mu)[0]) / (2.0 * h);
                    let an = at.get(D_X + i, mu)[g];
                    assert!((fd - an).abs() < 1e-7 * (1.0 + an.abs()), "cart={cart} AO {mu} d_{i}: {an} vs {fd}");
                    for j in 0..3 {
                        let fd = (plus.get(D_X + j, mu)[0] - minus.get(D_X + j, mu)[0]) / (2.0 * h);
                        let an = at.get(hessian_component(i, j), mu)[g];
                        assert!(
                            (fd - an).abs() < 1e-6 * (1.0 + an.abs()),
                            "cart={cart} AO {mu} d_{i}d_{j}: {an} vs {fd}"
                        );
                    }
                }
            }
        }
    }
}

/// Stored and integral-direct SCF agree with f functions, for the Coulomb and for the
/// erf-attenuated integrals (CAM-B3LYP), spherical and Cartesian; so do the gradients.
#[test]
fn direct_matches_stored_with_f() {
    let cases = [
        ("hf", "6-31g(2df,p)", false),
        ("hf", "6-31g(2df,p)", true),
        ("cam-b3lyp", "6-31g(2df,p)", false),
        ("cam-b3lyp", "6-31g(2df,p)", true),
    ];
    for (method, basis, cart) in cases {
        let calc = |s: IntegralStrategy| {
            Calculation::new(&water(), method, basis)
                .unwrap()
                .with_cartesian_d(cart)
                .with_scf_options(tight())
                .with_integral_strategy(s)
                .gradient()
                .unwrap()
        };
        let stored = calc(IntegralStrategy::Stored);
        let direct = calc(IntegralStrategy::Direct);
        assert_eq!(stored.scf.integral_strategy, IntegralStrategy::Stored);
        assert_eq!(direct.scf.integral_strategy, IntegralStrategy::Direct);
        let de = (stored.energy - direct.energy).abs();
        let gs: Vec<f64> = stored.gradient.iter().flatten().copied().collect();
        let gd: Vec<f64> = direct.gradient.iter().flatten().copied().collect();
        let dg = max_dev(&gs, &gd);
        println!("{method}/{basis} cart={cart}: E = {:.10}, |dE| {de:.1e}, |dg| {dg:.1e}", stored.energy);
        assert!(de <= 1e-10, "{method}/{basis} cart={cart}: |dE| = {de:.2e}");
        assert!(dg <= 1e-8, "{method}/{basis} cart={cart}: |dg| = {dg:.2e}");
    }
}

/// The f functions do change the result: spherical (7f) and Cartesian (10f) energies
/// differ, and adding the f shell lowers the RHF energy (variational).
#[test]
fn f_functions_contribute() {
    let mol = hf_molecule();
    let e = |basis: &str, cart: bool| {
        Calculation::new(&mol, "hf", basis)
            .unwrap()
            .with_cartesian_d(cart)
            .with_scf_options(tight())
            .energy()
            .unwrap()
            .energy
    };
    let (sph, cart) = (e("6-31g(2df,p)", false), e("6-31g(2df,p)", true));
    assert!((sph - cart).abs() > 1e-5, "7f vs 10f: {sph} {cart}");
    assert!(cart < sph);
    // 6-311G(2df,2pd) contains all of 6-311G (same valence functions) plus polarization.
    assert!(e("6-311g(2df,2pd)", false) < e("6-311g", false));
}

/// Analytic RHF and B3LYP gradients with f functions vs central differences of the
/// energy (h = 1e-3 Bohr). B3LYP uses the fine grid, which keeps the neglected
/// grid-weight derivative (theory.typ section 10.5) below the tolerance here.
/// Observed (water): 1.5e-7 for all three, the O(h²) truncation error.
#[test]
fn f_gradients_match_finite_differences() {
    let mol = water();
    let h = 1e-3;
    for (method, cart, grid) in [
        ("hf", false, GridSettings::default()),
        ("hf", true, GridSettings::default()),
        ("b3lyp", false, GridSettings::fine()),
    ] {
        let tol = 1e-6;
        let calc = Calculation::new(&mol, method, "6-31g(2df,p)")
            .unwrap()
            .with_cartesian_d(cart)
            .with_grid(grid)
            .with_scf_options(tight());
        let g = calc.gradient().unwrap().gradient;
        let x0 = mol.coordinates();
        let mut worst: f64 = 0.0;
        for i in 0..x0.len() {
            let mut e = [0.0; 2];
            for (s, sign) in [1.0, -1.0].iter().enumerate() {
                let mut x = x0.clone();
                x[i] += sign * h;
                e[s] = calc.clone().with_molecule(&mol.with_coordinates(&x).unwrap()).energy().unwrap().energy;
            }
            let fd = (e[0] - e[1]) / (2.0 * h);
            worst = worst.max((fd - g[i / 3][i % 3]).abs());
        }
        println!("{method} cart={cart}: max|analytic - FD| = {worst:.2e}");
        assert!(worst < tol, "{method} cart={cart}: analytic vs FD {worst:.2e} (tol {tol:.0e})");
    }
}

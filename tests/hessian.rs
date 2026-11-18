//! Numerical Hessian of `Calculation::hessian` (theory.typ eq:hess-fd, eq:hess-sym)
//! checked against second differences of the energy, translational invariance and
//! the DFT-D3 part.

use qc_rs::dispersion::dispersion_energy_and_gradient;
use qc_rs::{
    Calculation, D3Damping, D3Parameters, DEFAULT_HESSIAN_STEP, Molecule, QcError, ScfOptions, Unit,
    central_difference_hessian,
};

fn h2() -> Molecule {
    Molecule::from_symbols(&["H", "H"], &[[0.0, 0.0, 0.0], [0.05, -0.03, 0.74]], Unit::Angstrom, 0).unwrap()
}

fn water() -> Molecule {
    Molecule::from_symbols(
        &["O", "H", "H"],
        &[[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
        Unit::Angstrom,
        0,
    )
    .unwrap()
}

/// Energy with a tight SCF, at `mol` with coordinate `i` shifted by `di` and `j` by `dj`.
fn energy(calc: &Calculation, mol: &Molecule, (i, di): (usize, f64), (j, dj): (usize, f64)) -> f64 {
    let mut x = mol.coordinates();
    x[i] += di;
    x[j] += dj;
    calc.clone().with_molecule(&mol.with_coordinates(&x).unwrap()).energy().unwrap().energy
}

/// H2/STO-3G RHF: the Hessian equals second differences of the energy
/// (d²E/dx_i dx_j by a 4-point formula with a smaller step), diagonal and mixed.
#[test]
fn rhf_h2_matches_energy_second_differences() {
    let mol = h2();
    let tight = ScfOptions { error_tolerance: 1e-10, ..ScfOptions::default() };
    let calc = Calculation::new(&mol, "hf", "sto-3g").unwrap().with_scf_options(tight);
    let res = calc.hessian(DEFAULT_HESSIAN_STEP).unwrap();
    assert_eq!(res.hessian.rows(), 6);
    assert_eq!(res.n_gradient_evaluations, 12);
    assert_eq!(res.step, DEFAULT_HESSIAN_STEP);
    // O(h²) truncation error, see rhf_water_translational_invariance.
    assert!(res.max_asymmetry < 2e-5, "delta_asym = {:e}", res.max_asymmetry);

    let h = 2e-3;
    let mut worst: f64 = 0.0;
    for (i, j) in [(5, 5), (2, 5), (3, 5), (0, 0), (0, 4)] {
        let fd = if i == j {
            (energy(&calc, &mol, (i, h), (j, 0.0)) - 2.0 * energy(&calc, &mol, (i, 0.0), (j, 0.0))
                + energy(&calc, &mol, (i, -h), (j, 0.0)))
                / (h * h)
        } else {
            (energy(&calc, &mol, (i, h), (j, h)) - energy(&calc, &mol, (i, h), (j, -h))
                - energy(&calc, &mol, (i, -h), (j, h))
                + energy(&calc, &mol, (i, -h), (j, -h)))
                / (4.0 * h * h)
        };
        worst = worst.max((res.hessian[(i, j)] - fd).abs());
    }
    // Truncation errors: h²/6 · g''' with h = 5e-3 for the Hessian, smaller for the energies.
    assert!(worst < 2e-5, "max |H - d2E| = {worst:e}");
    assert!(res.hessian[(5, 5)] > 0.1, "bond stretch curvature {}", res.hessian[(5, 5)]);
}

/// The reference point is the one of `gradient()`, bit for bit.
#[test]
fn reference_point_is_gradient() {
    let calc = Calculation::new(&h2(), "hf", "sto-3g").unwrap();
    let g = calc.gradient().unwrap();
    let res = calc.hessian(DEFAULT_HESSIAN_STEP).unwrap();
    assert_eq!(res.energy, g.energy);
    assert_eq!(res.gradient, g.gradient);
    assert_eq!(res.scf.density, g.scf.density);
    assert!(res.displaced_scf_iterations > 0);
}

/// Water/STO-3G RHF: Σ_A H_(A_α, j) = 0 (translational invariance, theory.typ sec:freq)
/// and H_ij = H_ji, both up to the O(h²) truncation error of eq:hess-fd.
///
/// The HF gradient sums to zero at every geometry, so each unsymmetrised column sums
/// to zero; the symmetrised row sum also contains the transposed part, whose
/// truncation errors (h²/6 · ∂³g_j/∂x_i³, different for each i) do not cancel.
/// Halving h must therefore divide both the row sums and δ_asym by about 4.
#[test]
fn rhf_water_translational_invariance() {
    let calc = Calculation::new(&water(), "hf", "sto-3g").unwrap();
    let measure = |step: f64| {
        let res = calc.hessian(step).unwrap();
        let mut worst: f64 = 0.0;
        for alpha in 0..3 {
            for j in 0..9 {
                let s: f64 = (0..3).map(|a| res.hessian[(3 * a + alpha, j)]).sum();
                worst = worst.max(s.abs());
            }
        }
        assert!(res.hessian.max_abs() > 0.1);
        (worst, res.max_asymmetry)
    };
    let (sum1, asym1) = measure(DEFAULT_HESSIAN_STEP);
    let (sum2, asym2) = measure(DEFAULT_HESSIAN_STEP / 2.0);
    assert!(sum1 < 1e-5, "max |sum_A H| = {sum1:e}");
    assert!(asym1 < 2e-5, "delta_asym = {asym1:e}");
    for (a, b) in [(sum1, sum2), (asym1, asym2)] {
        assert!((a / b - 4.0).abs() < 0.2, "not O(h^2): {a:e} -> {b:e}");
    }
}

/// With a D3 correction the Hessian is that of the total energy: it differs from the
/// Hessian without D3 by the central-difference Hessian of the D3 gradient alone.
#[test]
fn dispersion_is_included() {
    let mol = h2();
    let params = D3Parameters::new(D3Damping::Bj { s6: 1.0, a1: 0.4, s8: 1.0, a2: 5.0 }).unwrap();
    let plain = Calculation::new(&mol, "hf", "sto-3g").unwrap();
    let with_d3 = plain.clone().with_dispersion_parameters(Some(params));
    let h_plain = plain.hessian(DEFAULT_HESSIAN_STEP).unwrap().hessian;
    let h_d3 = with_d3.hessian(DEFAULT_HESSIAN_STEP).unwrap().hessian;
    let h_disp = central_difference_hessian(&mol.coordinates(), DEFAULT_HESSIAN_STEP, |x| {
        let (_, g) = dispersion_energy_and_gradient(&mol.with_coordinates(x)?, &params)?;
        Ok(g.iter().flatten().copied().collect())
    })
    .unwrap()
    .hessian;
    let diff = (&(&h_d3 - &h_plain) - &h_disp).max_abs();
    assert!(diff < 1e-12, "max |H_D3 - H - H_disp| = {diff:e}");
    assert!(h_disp.max_abs() > 1e-6, "D3 Hessian too small to test: {:e}", h_disp.max_abs());
}

#[test]
fn invalid_step_is_an_error() {
    let calc = Calculation::new(&h2(), "hf", "sto-3g").unwrap();
    for h in [0.0, -1e-3, f64::NAN] {
        assert!(matches!(calc.hessian(h), Err(QcError::InvalidInput(_))), "{h}");
    }
}

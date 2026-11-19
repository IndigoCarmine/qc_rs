//! Numerical Hessian of `Calculation::hessian` (theory.typ eq:hess-fd, eq:hess-sym)
//! checked against second differences of the energy, translational invariance and
//! the DFT-D3 part; and `Calculation::frequencies` built on it.

use qc_rs::dispersion::dispersion_energy_and_gradient;
use qc_rs::{
    Calculation, D3Damping, D3Parameters, D3Variant, DEFAULT_HESSIAN_STEP, FrequencySettings, Molecule, QcError,
    ScfOptions, ThermoSettings, Unit, central_difference_hessian, harmonic_analysis, thermochemistry,
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

/// Dipole derivatives (theory.typ eq:ir-intensity): translating the whole molecule by
/// δ changes μ by q δ (eq:dipole with the origin fixed), so Σ_A ∂μ_α/∂x_Aβ = q δ_αβ,
/// for neutral LiH (q = 0) and for the cation HeH⁺ (q = 1). Each sum is a sum of central
/// differences of a smooth function, so it holds up to O(h²) and SCF convergence.
#[test]
fn dipole_derivatives_sum_to_the_charge() {
    let lih = Molecule::from_symbols(&["Li", "H"], &[[0.0, 0.0, 0.0], [0.05, -0.03, 1.6]], Unit::Angstrom, 0).unwrap();
    let heh = Molecule::from_symbols(&["He", "H"], &[[0.0, 0.0, 0.0], [0.05, -0.03, 0.78]], Unit::Angstrom, 1).unwrap();
    for (mol, q) in [(lih, 0.0), (heh, 1.0)] {
        let res = Calculation::new(&mol, "hf", "sto-3g").unwrap().hessian(DEFAULT_HESSIAN_STEP).unwrap();
        let d = &res.dipole_derivatives;
        assert_eq!((d.rows(), d.cols()), (6, 3));
        let mut worst: f64 = 0.0;
        for alpha in 0..3 {
            for beta in 0..3 {
                let s: f64 = (0..2).map(|a| d[(3 * a + beta, alpha)]).sum();
                worst = worst.max((s - if alpha == beta { q } else { 0.0 }).abs());
            }
        }
        assert!(worst < 1e-6, "q = {q}: max |Σ_A ∂μ/∂x_A − q δ| = {worst:e}");
        assert!(d.max_abs() > 0.1, "q = {q}: dipole derivatives too small to test");
    }
}

#[test]
fn invalid_step_is_an_error() {
    let calc = Calculation::new(&h2(), "hf", "sto-3g").unwrap();
    for h in [0.0, -1e-3, f64::NAN] {
        assert!(matches!(calc.hessian(h), Err(QcError::InvalidInput(_))), "{h}");
    }
}

/// `frequencies()` with DFT-D3 (B3LYP-D3(BJ)/STO-3G, H2): the Hessian is that of the
/// total energy, and the thermochemistry uses E_el = E_SCF + E_disp (theory.typ
/// eq:thermo-total, eq:d3-total) with the wavenumbers of that Hessian.
#[test]
fn frequencies_include_dispersion() {
    let mol = h2();
    let plain = Calculation::new(&mol, "b3lyp", "sto-3g").unwrap();
    let with_d3 = plain.clone().with_dispersion(D3Variant::Bj).unwrap();
    let settings = FrequencySettings::default();
    let r = with_d3.frequencies(&settings).unwrap();
    let r0 = plain.frequencies(&settings).unwrap();
    let g = with_d3.gradient().unwrap();
    let e_disp = g.dispersion_energy.unwrap();
    assert!(e_disp.abs() > 1e-5, "E_disp too small to test: {e_disp:e}");

    assert!((r.hessian.energy - (g.scf.energy + e_disp)).abs() < 1e-10);
    assert_eq!(r.thermo.electronic_energy, r.hessian.energy);
    let de = r.thermo.electronic_energy - r0.thermo.electronic_energy;
    assert!((de - e_disp).abs() < 1e-10, "ΔE_el = {de:e}, E_disp = {e_disp:e}");
    assert_eq!(r.analysis, harmonic_analysis(&mol, &r.hessian.hessian, None).unwrap());
    let direct = thermochemistry(&mol, None, &r.analysis.frequencies_cm1, r.hessian.energy, &settings.thermo).unwrap();
    assert_eq!(r.thermo, direct);
    // H and G move by E_disp plus the (small) change of the vibrational terms.
    let d_vib = r.thermo.vibration.energy - r0.thermo.vibration.energy;
    let dh = r.thermo.enthalpy - r0.thermo.enthalpy;
    assert!((dh - e_disp - d_vib).abs() < 1e-10, "ΔH = {dh:e}");
    let dw = (r.analysis.frequencies_cm1[0] - r0.analysis.frequencies_cm1[0]).abs();
    assert!(dw > 1e-3, "D3 does not change the H2 wavenumber: {dw:e} cm⁻¹");
    let g_max = g.gradient.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!((r.max_abs_gradient - g_max).abs() < 1e-10, "{} vs {g_max}", r.max_abs_gradient);
}

/// User masses reach both the harmonic analysis and the thermochemistry.
#[test]
fn frequencies_use_given_masses() {
    let mol = h2();
    let calc = Calculation::new(&mol, "hf", "sto-3g").unwrap();
    let masses = vec![2.014101778, 2.014101778]; // D2
    let settings = FrequencySettings { masses: Some(masses.clone()), ..FrequencySettings::default() };
    let r = calc.frequencies(&settings).unwrap();
    assert_eq!(r.analysis, harmonic_analysis(&mol, &r.hessian.hessian, Some(&masses)).unwrap());
    let direct =
        thermochemistry(&mol, Some(&masses), &r.analysis.frequencies_cm1, r.hessian.energy, &settings.thermo).unwrap();
    assert_eq!(r.thermo, direct);
    let h = harmonic_analysis(&mol, &r.hessian.hessian, None).unwrap();
    let ratio = r.analysis.frequencies_cm1[0] / h.frequencies_cm1[0];
    assert!((ratio - (1.008_f64 / 2.014101778).sqrt()).abs() < 1e-10, "ν(D2)/ν(H2) = {ratio}");
}

/// Invalid steps, masses and thermochemistry settings are errors.
#[test]
fn invalid_frequency_settings_are_errors() {
    let calc = Calculation::new(&h2(), "hf", "sto-3g").unwrap();
    let d = FrequencySettings::default;
    let thermo = ThermoSettings::default;
    let bad = [
        FrequencySettings { step: 0.0, ..d() },
        FrequencySettings { step: f64::NAN, ..d() },
        FrequencySettings { masses: Some(vec![1.008]), ..d() },
        FrequencySettings { masses: Some(vec![1.008, -1.0]), ..d() },
        FrequencySettings { thermo: ThermoSettings { symmetry_number: 0, ..thermo() }, ..d() },
        FrequencySettings { thermo: ThermoSettings { temperature: 0.0, ..thermo() }, ..d() },
        FrequencySettings { thermo: ThermoSettings { pressure: f64::NAN, ..thermo() }, ..d() },
    ];
    for s in bad {
        assert!(matches!(calc.frequencies(&s), Err(QcError::InvalidInput(_))), "{s:?}");
    }
}

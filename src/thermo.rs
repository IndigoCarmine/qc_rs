//! Thermochemistry in the ideal-gas, rigid-rotor, harmonic-oscillator approximation
//! (theory.typ sec:freq, "熱化学": eq:thermo-trans, eq:thermo-rot, eq:thermo-vib,
//! eq:thermo-elec, eq:thermo-total). Everything is per molecule, in E_h and E_h/K.
//!
//! The input is a geometry, masses and harmonic wavenumbers, e.g. from
//! [`crate::harmonic_analysis`]; no electronic-structure work is done here.

use crate::error::{QcError, Result};
use crate::molecule::Molecule;
use crate::vibration::{ELECTRON_MASS_IN_U, HARTREE_IN_WAVENUMBER, inertia_frame, resolve_masses};

/// E_h / k_B in K (CODATA 2018). theory.typ eq:thermo-trans UNCHECKED(CODATA2018)
const HARTREE_IN_KELVIN: f64 = 315775.02480407;

/// Planck constant h in J s (SI, exact). theory.typ eq:thermo-trans UNCHECKED(CODATA2018)
const PLANCK_SI: f64 = 6.62607015e-34;

/// Boltzmann constant k_B in J/K (SI, exact). theory.typ eq:thermo-trans UNCHECKED(CODATA2018)
const BOLTZMANN_SI: f64 = 1.380649e-23;

/// Atomic mass constant m_u in kg (CODATA 2018). theory.typ eq:thermo-trans UNCHECKED(CODATA2018)
const ATOMIC_MASS_UNIT_SI: f64 = 1.66053906660e-27;

/// Settings of [`thermochemistry`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermoSettings {
    /// Temperature T in K (default 298.15).
    pub temperature: f64,
    /// Pressure p in Pa (default 101325).
    pub pressure: f64,
    /// Rotational symmetry number σ (default 1; not determined automatically).
    pub symmetry_number: u32,
}

impl Default for ThermoSettings {
    fn default() -> Self {
        Self { temperature: 298.15, pressure: 101325.0, symmetry_number: 1 }
    }
}

/// Energy (E_h), entropy (E_h/K) and constant-volume heat capacity (E_h/K) of one part
/// of the partition function.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ThermoContribution {
    /// Thermal energy E (E_h). For the vibrations it includes the zero-point energy.
    pub energy: f64,
    /// Entropy S (E_h/K).
    pub entropy: f64,
    /// Heat capacity at constant volume C_v (E_h/K).
    pub heat_capacity: f64,
}

/// Result of [`thermochemistry`] (theory.typ eq:thermo-total), per molecule.
#[derive(Debug, Clone, PartialEq)]
pub struct Thermochemistry {
    /// The settings used.
    pub settings: ThermoSettings,
    /// Electronic energy E_el (E_h), the energy at the bottom of the potential.
    pub electronic_energy: f64,
    /// Translation (eq:thermo-trans).
    pub translation: ThermoContribution,
    /// Rotation (eq:thermo-rot); zero for one atom.
    pub rotation: ThermoContribution,
    /// Vibration (eq:thermo-vib); its energy includes the zero-point energy.
    pub vibration: ThermoContribution,
    /// Electronic excitation (eq:thermo-elec): zero for a closed-shell singlet.
    pub electronic: ThermoContribution,
    /// Zero-point energy E_ZPE (E_h).
    pub zero_point_energy: f64,
    /// E_el + E_ZPE (E_h).
    pub energy_at_0k: f64,
    /// Internal energy U (E_h).
    pub internal_energy: f64,
    /// Enthalpy H = U + k_B T (E_h).
    pub enthalpy: f64,
    /// Entropy S (E_h/K).
    pub entropy: f64,
    /// Gibbs energy G = H − T S (E_h).
    pub gibbs_energy: f64,
    /// C_v (E_h/K).
    pub heat_capacity_v: f64,
    /// C_p = C_v + k_B (E_h/K).
    pub heat_capacity_p: f64,
    /// Rotational temperatures Θ_a (K) of the kept rotations, in the order of the
    /// principal moments I_1 ≤ I_2 ≤ I_3 (so descending): 3 values, 2 for a linear
    /// molecule (Θ_2 = Θ_3), none for one atom.
    pub rotational_temperatures: Vec<f64>,
    /// Whether the linear-rotor expression was used (exactly two rotations kept,
    /// theory.typ eq:tr-vectors).
    pub is_linear: bool,
    /// Number of wavenumbers ≤ 0 (imaginary, or exactly zero) left out of the
    /// vibrational sums.
    pub n_excluded_modes: usize,
}

/// Thermochemistry of `mol` (theory.typ eq:thermo-total) from its harmonic
/// wavenumbers `frequencies_cm1` (cm⁻¹, imaginary as negative, 3 N − n_tr of them as
/// returned by [`crate::harmonic_analysis`]) and electronic energy `electronic_energy`
/// (E_h). `masses` (u, one per atom) defaults to [`crate::standard_atomic_mass`].
///
/// Linearity and the principal moments follow the rotation rule of theory.typ
/// eq:tr-vectors, the same rule as [`crate::harmonic_analysis`].
pub fn thermochemistry(
    mol: &Molecule,
    masses: Option<&[f64]>,
    frequencies_cm1: &[f64],
    electronic_energy: f64,
    settings: &ThermoSettings,
) -> Result<Thermochemistry> {
    let t = settings.temperature;
    let p = settings.pressure;
    if !(t.is_finite() && t > 0.0) {
        return Err(QcError::InvalidInput(format!("temperature must be finite and positive, got {t}")));
    }
    if !(p.is_finite() && p > 0.0) {
        return Err(QcError::InvalidInput(format!("pressure must be finite and positive, got {p}")));
    }
    if settings.symmetry_number == 0 {
        return Err(QcError::InvalidInput("symmetry number must be at least 1".into()));
    }
    if !electronic_energy.is_finite() {
        return Err(QcError::InvalidInput(format!("electronic energy must be finite, got {electronic_energy}")));
    }
    if let Some(bad) = frequencies_cm1.iter().find(|x| !x.is_finite()) {
        return Err(QcError::InvalidInput(format!("wavenumbers must be finite, got {bad}")));
    }
    let masses = resolve_masses(mol, masses)?;
    let frame = inertia_frame(mol, &masses)?;
    let n_modes = 3 * masses.len() - 3 - frame.kept_rotations.len();
    if frequencies_cm1.len() != n_modes {
        return Err(QcError::InvalidInput(format!(
            "{} wavenumbers given, expected 3N − n_tr = {n_modes}",
            frequencies_cm1.len()
        )));
    }

    let k = 1.0 / HARTREE_IN_KELVIN; // k_B in E_h/K
    let kt = k * t;

    // eq:thermo-trans: q_t in SI.
    let mass_kg = masses.iter().sum::<f64>() * ATOMIC_MASS_UNIT_SI;
    let kt_si = BOLTZMANN_SI * t;
    let q_t = (2.0 * std::f64::consts::PI * mass_kg * kt_si / (PLANCK_SI * PLANCK_SI)).powf(1.5) * kt_si / p;
    let translation = ThermoContribution { energy: 1.5 * kt, entropy: k * (q_t.ln() + 2.5), heat_capacity: 1.5 * k };

    // eq:thermo-rot: Θ_a = (E_h/k_B) / (2 I_a m_u/m_e), I_a in u Bohr².
    let rotational_temperatures: Vec<f64> = frame
        .kept_rotations
        .iter()
        .map(|&a| HARTREE_IN_KELVIN * ELECTRON_MASS_IN_U / (2.0 * frame.principal.values[a]))
        .collect();
    let sigma = f64::from(settings.symmetry_number);
    let rotation = match rotational_temperatures.as_slice() {
        [] => ThermoContribution::default(),
        [_, theta] => {
            // Linear: Θ = Θ_2 = Θ_3 (the larger moment of the two kept).
            let q_r = t / (sigma * theta);
            ThermoContribution { energy: kt, entropy: k * (q_r.ln() + 1.0), heat_capacity: k }
        }
        [t1, t2, t3] => {
            let q_r = std::f64::consts::PI.sqrt() / sigma * t.powf(1.5) / (t1 * t2 * t3).sqrt();
            ThermoContribution { energy: 1.5 * kt, entropy: k * (q_r.ln() + 1.5), heat_capacity: 1.5 * k }
        }
        // I_2 ≥ I_3 / 2 for any body, so exactly one kept rotation cannot occur.
        other => {
            return Err(QcError::InvalidInput(format!("{} rotations kept", other.len())));
        }
    };

    // eq:thermo-vib over ν̃_k > 0; Θ_k = ν̃_k (E_h/k_B) / (E_h/hc).
    let mut vibration = ThermoContribution::default();
    let mut zero_point_energy = 0.0;
    let mut n_excluded_modes = 0;
    for &nu in frequencies_cm1 {
        if nu <= 0.0 {
            n_excluded_modes += 1;
            continue;
        }
        let theta = nu * HARTREE_IN_KELVIN / HARTREE_IN_WAVENUMBER;
        let x = theta / t;
        let em1 = x.exp_m1(); // e^x − 1
        let one_minus_emx = -(-x).exp_m1(); // 1 − e^{−x}
        zero_point_energy += 0.5 * k * theta;
        vibration.energy += k * theta * (0.5 + 1.0 / em1);
        vibration.entropy += k * (x / em1 - (-(-x).exp()).ln_1p());
        // e^x (x / (e^x − 1))² = x² e^{−x} / (1 − e^{−x})²
        vibration.heat_capacity += k * x * x * (-x).exp() / (one_minus_emx * one_minus_emx);
    }

    // eq:thermo-elec: q_e = 1.
    let electronic = ThermoContribution::default();

    // eq:thermo-total.
    let parts = [translation, rotation, vibration, electronic];
    let internal_energy = electronic_energy + parts.iter().map(|c| c.energy).sum::<f64>();
    let enthalpy = internal_energy + kt;
    let entropy: f64 = parts.iter().map(|c| c.entropy).sum();
    let heat_capacity_v: f64 = parts.iter().map(|c| c.heat_capacity).sum();
    Ok(Thermochemistry {
        settings: *settings,
        electronic_energy,
        translation,
        rotation,
        vibration,
        electronic,
        zero_point_energy,
        energy_at_0k: electronic_energy + zero_point_energy,
        internal_energy,
        enthalpy,
        entropy,
        gibbs_energy: enthalpy - t * entropy,
        heat_capacity_v,
        heat_capacity_p: heat_capacity_v + k,
        rotational_temperatures,
        is_linear: frame.kept_rotations.len() == 2,
        n_excluded_modes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Unit;

    fn water() -> Molecule {
        Molecule::from_symbols(
            &["O", "H", "H"],
            &[[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
            Unit::Angstrom,
            0,
        )
        .unwrap()
    }

    const WATER_FREQ: [f64; 3] = [1600.0, 3800.0, 3900.0];

    /// σ → 2σ lowers S_r by k_B ln 2 and raises G by k_B T ln 2 (eq:thermo-rot).
    #[test]
    fn symmetry_number_shifts_gibbs_by_kt_ln2() {
        let mol = water();
        let s1 = ThermoSettings { symmetry_number: 1, ..Default::default() };
        let s2 = ThermoSettings { symmetry_number: 2, ..Default::default() };
        let a = thermochemistry(&mol, None, &WATER_FREQ, -76.0, &s1).unwrap();
        let b = thermochemistry(&mol, None, &WATER_FREQ, -76.0, &s2).unwrap();
        let kt_ln2 = 298.15 / HARTREE_IN_KELVIN * 2f64.ln();
        assert!((b.gibbs_energy - a.gibbs_energy - kt_ln2).abs() < 1e-14, "{}", b.gibbs_energy - a.gibbs_energy);
        assert_eq!(a.enthalpy, b.enthalpy);
    }

    /// One atom: translation only, no rotation and no vibration.
    #[test]
    fn single_atom_is_translation_only() {
        let mol = Molecule::from_symbols(&["Ar"], &[[0.3, -0.2, 1.0]], Unit::Bohr, 0).unwrap();
        let r = thermochemistry(&mol, None, &[], -527.0, &ThermoSettings::default()).unwrap();
        assert!(r.rotational_temperatures.is_empty() && !r.is_linear);
        assert_eq!(r.rotation, ThermoContribution::default());
        assert_eq!(r.vibration, ThermoContribution::default());
        let kt = 298.15 / HARTREE_IN_KELVIN;
        assert!((r.enthalpy - (-527.0 + 2.5 * kt)).abs() < 1e-12);
    }

    /// Imaginary and zero wavenumbers are left out of the sums and counted.
    #[test]
    fn nonpositive_wavenumbers_are_excluded() {
        let mol = water();
        let st = ThermoSettings::default();
        let with = thermochemistry(&mol, None, &[-300.0, 0.0, 3900.0], -76.0, &st).unwrap();
        let only = thermochemistry(&mol, None, &[3900.0, -1.0, -2.0], -76.0, &st).unwrap();
        assert_eq!(with.n_excluded_modes, 2);
        assert_eq!(with.vibration, only.vibration);
    }

    /// Very high wavenumbers / low temperature: no overflow, C_v,v → 0, E_v → ZPE.
    #[test]
    fn stiff_modes_do_not_overflow() {
        let mol = water();
        let st = ThermoSettings { temperature: 1.0, ..Default::default() };
        let r = thermochemistry(&mol, None, &WATER_FREQ, -76.0, &st).unwrap();
        assert!(r.vibration.heat_capacity == 0.0 && r.vibration.entropy == 0.0);
        assert_eq!(r.vibration.energy, r.zero_point_energy);
        assert!(r.gibbs_energy.is_finite());
    }

    #[test]
    fn rejects_bad_input() {
        let mol = water();
        let st = ThermoSettings::default();
        assert!(thermochemistry(&mol, None, &[1000.0, 2000.0], -76.0, &st).is_err());
        assert!(thermochemistry(&mol, None, &[1000.0, 2000.0, f64::NAN], -76.0, &st).is_err());
        assert!(thermochemistry(&mol, None, &WATER_FREQ, f64::INFINITY, &st).is_err());
        assert!(thermochemistry(&mol, Some(&[1.0, 1.0]), &WATER_FREQ, -76.0, &st).is_err());
        for bad in [
            ThermoSettings { temperature: 0.0, ..st },
            ThermoSettings { pressure: -1.0, ..st },
            ThermoSettings { symmetry_number: 0, ..st },
        ] {
            assert!(thermochemistry(&mol, None, &WATER_FREQ, -76.0, &bad).is_err());
        }
    }
}

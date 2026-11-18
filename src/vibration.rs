//! Harmonic vibrational analysis of a Cartesian Hessian (theory.typ sec:freq):
//! mass weighting (eq:mass-weight), separation of translations and rotations
//! (eq:tr-vectors), harmonic wavenumbers (eq:freq) and reduced masses (eq:reduced-mass).
//!
//! The Hessian may come from [`crate::Calculation::hessian`] or from elsewhere; the
//! analysis itself does no electronic-structure work.

use crate::error::{QcError, Result};
use crate::linalg::{Matrix, symmetric_eigen};
use crate::molecule::Molecule;

/// Standard atomic weights (u) of H–Ar, IUPAC/CIAAW 2013 (Meija et al. 2016), conventional
/// values for elements given as intervals. theory.typ eq:mass-weight UNCHECKED(Meija2016)
const STANDARD_ATOMIC_MASSES: [f64; 18] = [
    1.008,
    4.002602,
    6.94,
    9.0121831,
    10.81,
    12.011,
    14.007,
    15.999,
    18.998403163,
    20.1797,
    22.98976928,
    24.305,
    26.9815385,
    28.085,
    30.973761998,
    32.06,
    35.45,
    39.948,
];

/// E_h / (h c) in cm⁻¹ (CODATA 2018). theory.typ eq:freq UNCHECKED(CODATA2018)
const HARTREE_IN_WAVENUMBER: f64 = 219474.6313632;

/// m_e / m_u (CODATA 2018). theory.typ eq:freq UNCHECKED(CODATA2018)
const ELECTRON_MASS_IN_U: f64 = 5.48579909065e-4;

/// A rotation a is kept only if I_a > ROTATION_THRESHOLD · I_3 (theory.typ
/// eq:tr-vectors; a numerical convention, not a literature value). DERIVED
const ROTATION_THRESHOLD: f64 = 1e-6;

/// Standard atomic weight (u) of element `z` (theory.typ eq:mass-weight table),
/// `None` outside H–Ar. UNCHECKED(Meija2016)
pub fn standard_atomic_mass(z: u32) -> Option<f64> {
    if (1..=STANDARD_ATOMIC_MASSES.len() as u32).contains(&z) {
        Some(STANDARD_ATOMIC_MASSES[z as usize - 1])
    } else {
        None
    }
}

/// Result of [`harmonic_analysis`].
///
/// Normalization convention (theory.typ sec:freq, "正規化の約束"): `normal_modes`
/// holds the **mass-weighted** unit vectors l_k (|l_k| = 1, dimensionless). The
/// Cartesian displacement of mode k is d_k,i = l_k,i / √m_i (u^-1/2, not normalized),
/// the Cartesian displacement in Bohr per unit change of the mass-weighted normal
/// coordinate Q_k (√u · Bohr); [`HarmonicAnalysis::cartesian_displacements`] returns it.
/// The sign of each l_k, and the basis within a degenerate subspace, are arbitrary.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicAnalysis {
    /// Harmonic wavenumbers in cm⁻¹ (theory.typ eq:freq), ascending; an imaginary
    /// frequency is reported as a negative wavenumber. Length 3 N − `n_trans_rot`.
    pub frequencies_cm1: Vec<f64>,
    /// Mass-weighted normal modes l_k as columns (3 N × n_modes, row index 3·atom + xyz),
    /// orthonormal, in the order of `frequencies_cm1`.
    pub normal_modes: Matrix,
    /// Reduced masses μ_k in u (theory.typ eq:reduced-mass), in the order of `frequencies_cm1`.
    pub reduced_masses: Vec<f64>,
    /// Eigenvalues λ_k of the internal block of the mass-weighted Hessian, E_h / (Bohr² u).
    pub eigenvalues: Vec<f64>,
    /// Number of translations and rotations removed: 6, 5 (linear) or 3 (one atom).
    pub n_trans_rot: usize,
    /// Whether exactly two rotations were kept (theory.typ eq:tr-vectors).
    pub is_linear: bool,
    /// Masses used (u), one per atom.
    pub masses: Vec<f64>,
}

impl HarmonicAnalysis {
    /// Cartesian displacements d_k,i = l_k,i / √m_i as columns (3 N × n_modes,
    /// u^-1/2), not normalized. theory.typ sec:freq DERIVED
    pub fn cartesian_displacements(&self) -> Matrix {
        let n = self.normal_modes.rows();
        Matrix::from_fn(n, self.normal_modes.cols(), |i, k| self.normal_modes[(i, k)] / self.masses[i / 3].sqrt())
    }
}

/// Harmonic analysis of the Cartesian Hessian `hessian` (E_h / Bohr², 3 N × 3 N, index
/// 3·atom + xyz) of `mol` (theory.typ sec:freq). `masses` (u, one per atom) defaults
/// to [`standard_atomic_mass`]. Only the symmetric part of `hessian` is used.
///
/// Steps: mass weighting (eq:mass-weight); translation and principal-axis rotation
/// vectors about the centre of mass (eq:tr-vectors), rotations kept only above
/// 10⁻⁶ I_3; an orthonormal basis D_int of their complement from the eigenvectors of
/// the projector P = 1 − D_tr D_trᵀ with eigenvalue 1; diagonalization of
/// D_intᵀ H̃ D_int; wavenumbers (eq:freq) and reduced masses (eq:reduced-mass).
pub fn harmonic_analysis(mol: &Molecule, hessian: &Matrix, masses: Option<&[f64]>) -> Result<HarmonicAnalysis> {
    let atoms = mol.atoms();
    let n_atom = atoms.len();
    let n = 3 * n_atom;
    if hessian.rows() != n || hessian.cols() != n {
        return Err(QcError::InvalidInput(format!(
            "Hessian is {}×{}, expected {n}×{n} for {n_atom} atoms",
            hessian.rows(),
            hessian.cols()
        )));
    }
    if !hessian.as_slice().iter().all(|x| x.is_finite()) {
        return Err(QcError::InvalidInput("Hessian has non-finite elements".into()));
    }
    let masses: Vec<f64> = match masses {
        Some(m) => {
            if m.len() != n_atom {
                return Err(QcError::InvalidInput(format!("{} masses given for {n_atom} atoms", m.len())));
            }
            if let Some(bad) = m.iter().find(|x| !(x.is_finite() && **x > 0.0)) {
                return Err(QcError::InvalidInput(format!("atomic masses must be finite and positive, got {bad}")));
            }
            m.to_vec()
        }
        None => atoms
            .iter()
            .map(|a| {
                standard_atomic_mass(a.z)
                    .ok_or_else(|| QcError::InvalidInput(format!("no standard atomic mass for Z = {}", a.z)))
            })
            .collect::<Result<_>>()?,
    };
    let sqrt_m: Vec<f64> = (0..n).map(|i| masses[i / 3].sqrt()).collect();

    // eq:mass-weight: H̃_ij = H_ij / √(m_i m_j), on the symmetric part of H.
    let hmw = Matrix::from_fn(n, n, |i, j| 0.5 * (hessian[(i, j)] + hessian[(j, i)]) / (sqrt_m[i] * sqrt_m[j]));

    // eq:tr-vectors: centre of mass, inertia tensor, principal axes.
    let total_mass: f64 = masses.iter().sum();
    let mut com = [0.0; 3];
    for (a, m) in atoms.iter().zip(&masses) {
        for (c, x) in com.iter_mut().zip(a.position) {
            *c += m * x / total_mass;
        }
    }
    let r: Vec<[f64; 3]> = atoms.iter().map(|a| [0, 1, 2].map(|k| a.position[k] - com[k])).collect();
    let mut inertia = Matrix::zeros(3, 3);
    for (ra, m) in r.iter().zip(&masses) {
        let r2 = ra[0] * ra[0] + ra[1] * ra[1] + ra[2] * ra[2];
        for p in 0..3 {
            for q in 0..3 {
                inertia[(p, q)] += m * (if p == q { r2 } else { 0.0 } - ra[p] * ra[q]);
            }
        }
    }
    let principal = symmetric_eigen(&inertia)?;
    let i3 = principal.values[2];

    let mut tr: Vec<Vec<f64>> = Vec::with_capacity(6);
    for alpha in 0..3 {
        // (t_α)_A = √m_A e_α
        tr.push((0..n).map(|i| if i % 3 == alpha { sqrt_m[i] } else { 0.0 }).collect());
    }
    let mut n_rot = 0;
    for a in 0..3 {
        if principal.values[a] > ROTATION_THRESHOLD * i3 {
            // (r_a)_A = √m_A u_a × r_A
            let u = [principal.vectors[(0, a)], principal.vectors[(1, a)], principal.vectors[(2, a)]];
            let mut v = vec![0.0; n];
            for (atom, ra) in r.iter().enumerate() {
                let c = [u[1] * ra[2] - u[2] * ra[1], u[2] * ra[0] - u[0] * ra[2], u[0] * ra[1] - u[1] * ra[0]];
                for k in 0..3 {
                    v[3 * atom + k] = sqrt_m[3 * atom] * c[k];
                }
            }
            tr.push(v);
            n_rot += 1;
        }
    }
    // Normalize and orthogonalize by modified Gram–Schmidt (eq:tr-vectors).
    let mut d_tr: Vec<Vec<f64>> = Vec::with_capacity(tr.len());
    for mut v in tr {
        for q in &d_tr {
            let p: f64 = v.iter().zip(q).map(|(a, b)| a * b).sum();
            v.iter_mut().zip(q).for_each(|(a, b)| *a -= p * b);
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Err(QcError::InvalidInput("degenerate translation/rotation vectors".into()));
        }
        v.iter_mut().for_each(|x| *x /= norm);
        d_tr.push(v);
    }
    let n_trans_rot = d_tr.len();
    let n_modes = n - n_trans_rot;

    // P = 1 − D_tr D_trᵀ; its eigenvectors with eigenvalue 1 span the internal space.
    let projector = Matrix::from_fn(n, n, |i, j| {
        (if i == j { 1.0 } else { 0.0 }) - d_tr.iter().map(|v| v[i] * v[j]).sum::<f64>()
    });
    let p_eig = symmetric_eigen(&projector)?;
    let internal: Vec<usize> = (0..n).filter(|&k| p_eig.values[k] > 0.5).collect();
    if internal.len() != n_modes {
        return Err(QcError::InvalidInput(format!(
            "internal space has dimension {}, expected {n_modes}",
            internal.len()
        )));
    }
    let d_int = Matrix::from_fn(n, n_modes, |i, k| p_eig.vectors[(i, internal[k])]);

    // Internal block D_intᵀ H̃ D_int, its eigenvectors c_k and l_k = D_int c_k.
    let block = &(&d_int.transpose() * &hmw) * &d_int;
    let block_eig = symmetric_eigen(&block)?;
    let normal_modes = &d_int * &block_eig.vectors;

    // eq:freq: c_ν = (E_h / h c) √(m_e / m_u).
    let c_nu = HARTREE_IN_WAVENUMBER * ELECTRON_MASS_IN_U.sqrt();
    let frequencies_cm1 =
        block_eig.values.iter().map(|&l| if l >= 0.0 { c_nu * l.sqrt() } else { -c_nu * (-l).sqrt() }).collect();
    // eq:reduced-mass: μ_k = (Σ_i l_k,i² / m_i)⁻¹.
    let reduced_masses = (0..n_modes)
        .map(|k| 1.0 / (0..n).map(|i| normal_modes[(i, k)].powi(2) / masses[i / 3]).sum::<f64>())
        .collect();

    Ok(HarmonicAnalysis {
        frequencies_cm1,
        normal_modes,
        reduced_masses,
        eigenvalues: block_eig.values,
        n_trans_rot,
        is_linear: n_rot == 2,
        masses,
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

    /// A model Hessian: harmonic springs between all atom pairs (k = 0.3 E_h/Bohr²),
    /// translationally and rotationally invariant by construction.
    fn spring_hessian(mol: &Molecule) -> Matrix {
        let atoms = mol.atoms();
        let n = 3 * atoms.len();
        let mut h = Matrix::zeros(n, n);
        for a in 0..atoms.len() {
            for b in 0..atoms.len() {
                if a == b {
                    continue;
                }
                let d: Vec<f64> = (0..3).map(|k| atoms[a].position[k] - atoms[b].position[k]).collect();
                let len2: f64 = d.iter().map(|x| x * x).sum();
                for p in 0..3 {
                    for q in 0..3 {
                        let kpq = 0.3 * d[p] * d[q] / len2;
                        h[(3 * a + p, 3 * a + q)] += kpq;
                        h[(3 * a + p, 3 * b + q)] -= kpq;
                    }
                }
            }
        }
        h
    }

    #[test]
    fn masses_of_table() {
        assert_eq!(standard_atomic_mass(1), Some(1.008));
        assert_eq!(standard_atomic_mass(17), Some(35.45));
        assert_eq!(standard_atomic_mass(0), None);
        assert_eq!(standard_atomic_mass(19), None);
    }

    #[test]
    fn single_atom_has_no_modes() {
        let mol = Molecule::from_symbols(&["Ne"], &[[0.1, 0.2, 0.3]], Unit::Bohr, 0).unwrap();
        let r = harmonic_analysis(&mol, &Matrix::zeros(3, 3), None).unwrap();
        assert_eq!(r.n_trans_rot, 3);
        assert!(r.frequencies_cm1.is_empty());
        assert_eq!(r.normal_modes.cols(), 0);
        assert!(!r.is_linear);
    }

    #[test]
    fn rejects_bad_input() {
        let mol = water();
        assert!(harmonic_analysis(&mol, &Matrix::zeros(6, 6), None).is_err());
        assert!(harmonic_analysis(&mol, &Matrix::zeros(9, 9), Some(&[1.0, 1.0])).is_err());
        assert!(harmonic_analysis(&mol, &Matrix::zeros(9, 9), Some(&[1.0, -1.0, 1.0])).is_err());
    }

    /// Modes are orthonormal and orthogonal to the translations.
    #[test]
    fn modes_are_orthonormal_and_internal() {
        let mol = water();
        let r = harmonic_analysis(&mol, &spring_hessian(&mol), None).unwrap();
        assert_eq!(r.n_trans_rot, 6);
        let l = &r.normal_modes;
        let overlap = &l.transpose() * l;
        assert!((&overlap - &Matrix::identity(3)).max_abs() < 1e-12);
        for k in 0..3 {
            for alpha in 0..3 {
                let t: f64 = (0..3).map(|a| l[(3 * a + alpha, k)] * r.masses[a].sqrt()).sum();
                assert!(t.abs() < 1e-12);
            }
            let d = r.cartesian_displacements();
            let norm2: f64 = (0..9).map(|i| d[(i, k)].powi(2)).sum();
            assert!((r.reduced_masses[k] - 1.0 / norm2).abs() < 1e-12);
        }
    }

    /// A rigidly translated and rotated molecule (Hessian rotated blockwise) gives the
    /// same wavenumbers.
    #[test]
    fn invariant_under_rigid_motion() {
        let mol = water();
        let base = harmonic_analysis(&mol, &spring_hessian(&mol), None).unwrap();
        let (c, s) = (0.6_f64.cos(), 0.6_f64.sin());
        let rot = Matrix::from_vec(3, 3, vec![c, -s, 0.0, s * 0.8, c * 0.8, -0.6, s * 0.6, c * 0.6, 0.8]).unwrap();
        let mut x = mol.coordinates();
        for a in 0..3 {
            let p = [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
            for i in 0..3 {
                x[3 * a + i] = (0..3).map(|j| rot[(i, j)] * p[j]).sum::<f64>() + [1.5, -2.0, 0.7][i];
            }
        }
        let moved = mol.with_coordinates(&x).unwrap();
        // H' = R H Rᵀ block by block; also equals the spring Hessian of the moved molecule.
        let h = spring_hessian(&mol);
        let h_rot = Matrix::from_fn(9, 9, |i, j| {
            let (a, p, b, q) = (i / 3, i % 3, j / 3, j % 3);
            (0..3).map(|k| (0..3).map(|l| rot[(p, k)] * h[(3 * a + k, 3 * b + l)] * rot[(q, l)]).sum::<f64>()).sum()
        });
        assert!((&h_rot - &spring_hessian(&moved)).max_abs() < 1e-12);
        let r = harmonic_analysis(&moved, &h_rot, None).unwrap();
        for (a, b) in base.frequencies_cm1.iter().zip(&r.frequencies_cm1) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    /// A linear molecule keeps 5 translations/rotations and has 3N − 5 modes.
    #[test]
    fn linear_molecule() {
        let mol = Molecule::from_symbols(&["C", "O", "O"], &[[0.0; 3], [0.0, 0.0, 2.2], [0.0, 0.0, -2.2]], Unit::Bohr, 0)
            .unwrap();
        let r = harmonic_analysis(&mol, &spring_hessian(&mol), None).unwrap();
        assert!(r.is_linear);
        assert_eq!(r.n_trans_rot, 5);
        assert_eq!(r.frequencies_cm1.len(), 4);
    }
}

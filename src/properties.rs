//! One-electron properties.

use crate::basis::Basis;
use crate::integrals::dipole;
use crate::linalg::Matrix;
use crate::molecule::Molecule;

/// Dipole moment (atomic units, origin at 0) for density matrix `p`. theory.typ eq:dipole
pub fn dipole_moment(mol: &Molecule, basis: &Basis, p: &Matrix) -> [f64; 3] {
    let ints = dipole(basis, [0.0; 3]);
    let mut mu = [0.0; 3];
    for k in 0..3 {
        let nuc: f64 = mol.atoms().iter().map(|a| a.z as f64 * a.position[k]).sum();
        mu[k] = nuc - p.dot(&ints[k]);
    }
    mu
}

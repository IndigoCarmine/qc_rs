//! Molecules: atoms, element table (H–Ar), unit conversion and nuclear repulsion.

use crate::error::{QcError, Result};

/// Bohr radius in Angstrom (CODATA 2018). theory.typ eq:bohr
pub const BOHR_IN_ANGSTROM: f64 = 0.529177210903;

const ELEMENTS: [&str; 18] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar",
];

/// Atomic number of an element symbol (case-insensitive), H–Ar only.
pub fn atomic_number(symbol: &str) -> Option<u32> {
    let s = symbol.trim();
    ELEMENTS
        .iter()
        .position(|e| e.eq_ignore_ascii_case(s))
        .map(|i| i as u32 + 1)
}

/// Element symbol of an atomic number, H–Ar only.
pub fn element_symbol(z: u32) -> Option<&'static str> {
    if (1..=ELEMENTS.len() as u32).contains(&z) {
        Some(ELEMENTS[z as usize - 1])
    } else {
        None
    }
}

/// Length unit of input coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// Atomic units (Bohr).
    Bohr,
    /// Angstrom.
    Angstrom,
}

/// A nucleus: atomic number and position in Bohr.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atom {
    /// Atomic number (nuclear charge).
    pub z: u32,
    /// Position in Bohr.
    pub position: [f64; 3],
}

/// A molecule: nuclei plus total charge.
#[derive(Debug, Clone, PartialEq)]
pub struct Molecule {
    atoms: Vec<Atom>,
    charge: i32,
}

impl Molecule {
    /// Creates a molecule from atoms (positions in Bohr) and total charge.
    pub fn new(atoms: Vec<Atom>, charge: i32) -> Result<Self> {
        if atoms.is_empty() {
            return Err(QcError::InvalidInput("molecule has no atoms".into()));
        }
        for a in &atoms {
            if element_symbol(a.z).is_none() {
                return Err(QcError::UnknownElement(format!("Z = {}", a.z)));
            }
            if a.position.iter().any(|x| !x.is_finite()) {
                return Err(QcError::InvalidInput("non-finite coordinate".into()));
            }
        }
        for i in 0..atoms.len() {
            for j in 0..i {
                if distance(&atoms[i].position, &atoms[j].position) < 1e-6 {
                    return Err(QcError::InvalidInput(format!("atoms {j} and {i} coincide")));
                }
            }
        }
        let mol = Molecule { atoms, charge };
        let ztot: i64 = mol.atoms.iter().map(|a| a.z as i64).sum();
        if ztot - (charge as i64) < 0 {
            return Err(QcError::InvalidInput("negative number of electrons".into()));
        }
        Ok(mol)
    }

    /// Creates a molecule from element symbols and coordinates in the given unit.
    pub fn from_symbols(symbols: &[&str], coords: &[[f64; 3]], unit: Unit, charge: i32) -> Result<Self> {
        if symbols.len() != coords.len() {
            return Err(QcError::InvalidInput("symbols and coordinates differ in length".into()));
        }
        let scale = match unit {
            Unit::Bohr => 1.0,
            Unit::Angstrom => 1.0 / BOHR_IN_ANGSTROM, // theory.typ eq:bohr
        };
        let atoms = symbols
            .iter()
            .zip(coords)
            .map(|(s, c)| {
                let z = atomic_number(s).ok_or_else(|| QcError::UnknownElement(s.to_string()))?;
                Ok(Atom { z, position: [c[0] * scale, c[1] * scale, c[2] * scale] })
            })
            .collect::<Result<Vec<_>>>()?;
        Molecule::new(atoms, charge)
    }

    /// Reads a molecule in XYZ format: the atom count, a comment line, then one
    /// `symbol x y z` line per atom, coordinates in Ångström.
    ///
    /// XYZ has no field for the charge, so it is taken from a `charge=N` token in
    /// the comment line when there is one (e.g. `charge=-1 acetate`) and is 0 otherwise.
    /// Extra columns after the coordinates and blank lines after the last atom are ignored.
    pub fn from_xyz(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        let first = lines.next().ok_or_else(|| QcError::Parse("XYZ: empty input".into()))?;
        let n: usize =
            first.trim().parse().map_err(|_| QcError::Parse(format!("XYZ: bad atom count '{}'", first.trim())))?;
        let comment = lines.next().ok_or_else(|| QcError::Parse("XYZ: missing comment line".into()))?;
        let mut charge = 0;
        for token in comment.split_whitespace() {
            if let Some(value) = token.strip_prefix("charge=") {
                charge = value.parse().map_err(|_| QcError::Parse(format!("XYZ: bad charge '{value}'")))?;
            }
        }
        let mut symbols = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n);
        for i in 0..n {
            let line = lines.next().ok_or_else(|| QcError::Parse(format!("XYZ: expected {n} atoms, found {i}")))?;
            let mut fields = line.split_whitespace();
            let symbol = fields.next().ok_or_else(|| QcError::Parse(format!("XYZ: empty line for atom {}", i + 1)))?;
            let mut xyz = [0.0; 3];
            for x in &mut xyz {
                *x = fields
                    .next()
                    .and_then(|f| f.parse::<f64>().ok())
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| QcError::Parse(format!("XYZ: bad coordinates for atom {}: '{line}'", i + 1)))?;
            }
            symbols.push(symbol);
            coords.push(xyz);
        }
        if let Some(extra) = lines.find(|l| !l.trim().is_empty()) {
            return Err(QcError::Parse(format!("XYZ: unexpected line after {n} atoms: '{extra}'")));
        }
        Molecule::from_symbols(&symbols, &coords, Unit::Angstrom, charge)
    }

    /// The atoms.
    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    /// Total molecular charge.
    pub fn charge(&self) -> i32 {
        self.charge
    }

    /// Number of electrons (sum of Z minus charge).
    pub fn n_electrons(&self) -> usize {
        let ztot: i64 = self.atoms.iter().map(|a| a.z as i64).sum();
        (ztot - self.charge as i64) as usize
    }

    /// Flattened coordinates (Bohr), 3 per atom.
    pub fn coordinates(&self) -> Vec<f64> {
        self.atoms.iter().flat_map(|a| a.position).collect()
    }

    /// Returns a copy with new flattened coordinates (Bohr).
    pub fn with_coordinates(&self, coords: &[f64]) -> Result<Self> {
        if coords.len() != 3 * self.atoms.len() {
            return Err(QcError::InvalidInput("coordinate vector has wrong length".into()));
        }
        let atoms = self
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| Atom { z: a.z, position: [coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]] })
            .collect();
        Molecule::new(atoms, self.charge)
    }

    /// Nuclear repulsion energy. theory.typ eq:vnn
    pub fn nuclear_repulsion(&self) -> f64 {
        let mut e = 0.0;
        for (i, a) in self.atoms.iter().enumerate() {
            for b in &self.atoms[..i] {
                e += (a.z * b.z) as f64 / distance(&a.position, &b.position);
            }
        }
        e
    }
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_table() {
        assert_eq!(atomic_number("cl"), Some(17));
        assert_eq!(atomic_number("Ar"), Some(18));
        assert_eq!(atomic_number("K"), None);
        assert_eq!(element_symbol(8), Some("O"));
    }

    #[test]
    fn xyz_reader() {
        let text = "3\nwater charge=0\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692 extra\n\n";
        let m = Molecule::from_xyz(text).unwrap();
        let direct = Molecule::from_symbols(
            &["O", "H", "H"],
            &[[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
            Unit::Angstrom,
            0,
        )
        .unwrap();
        assert_eq!(m, direct);
        // Charge from the comment line; Windows line endings.
        let ion = Molecule::from_xyz("1\r\nfluoride charge=-1\r\nF 0 0 0\r\n").unwrap();
        assert_eq!((ion.charge(), ion.n_electrons()), (-1, 10));
        // No charge token: neutral.
        assert_eq!(Molecule::from_xyz("1\nMOL\nHe 0 0 0").unwrap().charge(), 0);
        for bad in [
            "",
            "x\nc\nH 0 0 0",
            "2\nc\nH 0 0 0",
            "1\nc\nH 0 0",
            "1\nc\nH 0 a 0",
            "1\nc\nXx 0 0 0",
            "1\nc\nH 0 0 0\nH 0 0 1",
            "1\ncharge=one\nH 0 0 0",
        ] {
            assert!(Molecule::from_xyz(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn h2_repulsion() {
        let m = Molecule::from_symbols(&["H", "H"], &[[0.0; 3], [0.0, 0.0, 1.4]], Unit::Bohr, 0).unwrap();
        assert!((m.nuclear_repulsion() - 1.0 / 1.4).abs() < 1e-15);
        assert_eq!(m.n_electrons(), 2);
    }
}

//! Gaussian basis sets: Gaussian94 parser, contracted shells, normalization,
//! spherical d transformation and AO bookkeeping (theory.typ section 2).

use crate::error::{QcError, Result};
use crate::molecule::{Molecule, atomic_number, element_symbol};
use std::collections::HashMap;
use std::f64::consts::PI;

const STO_3G: &str = include_str!("../data/basis/sto-3g.gbs");
const B6_31G: &str = include_str!("../data/basis/6-31g.gbs");
const B6_31GS: &str = include_str!("../data/basis/6-31gs.gbs");
const B6_31GSS: &str = include_str!("../data/basis/6-31gss.gbs");

/// Names of the built-in basis sets.
pub const BUILTIN_BASIS_SETS: [&str; 4] = ["sto-3g", "6-31g", "6-31g*", "6-31g**"];

/// One contracted shell as written in a basis file (coefficients refer to normalized primitives).
#[derive(Debug, Clone, PartialEq)]
pub struct ShellDefinition {
    /// Angular momentum L.
    pub l: usize,
    /// Primitive exponents a_k.
    pub exponents: Vec<f64>,
    /// Contraction coefficients d_k.
    pub coefficients: Vec<f64>,
}

/// A basis-set definition: shells per element.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisSetDefinition {
    /// Name used in error messages.
    pub name: String,
    elements: HashMap<u32, Vec<ShellDefinition>>,
}

impl BasisSetDefinition {
    /// One of the embedded basis sets: "sto-3g", "6-31g", "6-31g*", "6-31g**" (case-insensitive).
    pub fn builtin(name: &str) -> Result<Self> {
        let text = match name.to_ascii_lowercase().as_str() {
            "sto-3g" => STO_3G,
            "6-31g" => B6_31G,
            "6-31g*" | "6-31gs" => B6_31GS,
            "6-31g**" | "6-31gss" => B6_31GSS,
            _ => return Err(QcError::InvalidInput(format!("unknown basis set '{name}'"))),
        };
        Self::from_gbs(name, text)
    }

    /// Parses Gaussian94-format text (theory.typ section 2.4).
    pub fn from_gbs(name: &str, text: &str) -> Result<Self> {
        let mut elements = HashMap::new();
        let lines: Vec<&str> = text.lines().map(|l| l.trim()).collect();
        let mut i = 0;
        let perr = |ln: usize, msg: &str| QcError::Parse(format!("{name}: line {}: {msg}", ln + 1));
        while i < lines.len() {
            let line = lines[i];
            if line.is_empty() || line.starts_with('!') || line == "****" {
                i += 1;
                continue;
            }
            let tok: Vec<&str> = line.split_whitespace().collect();
            if tok.len() != 2 || tok[1] != "0" {
                return Err(perr(i, "expected '<symbol> 0'"));
            }
            let z = atomic_number(tok[0]).ok_or_else(|| QcError::UnknownElement(tok[0].to_string()))?;
            i += 1;
            let mut shells = Vec::new();
            loop {
                let line = *lines.get(i).ok_or_else(|| perr(i, "unexpected end of file"))?;
                if line == "****" {
                    break;
                }
                let tok: Vec<&str> = line.split_whitespace().collect();
                if tok.len() != 3 {
                    return Err(perr(i, "expected '<type> <nprim> <scale>'"));
                }
                let kind = tok[0].to_ascii_uppercase();
                let nprim: usize = tok[1].parse().map_err(|_| perr(i, "bad primitive count"))?;
                let scale = parse_number(tok[2]).ok_or_else(|| perr(i, "bad scale factor"))?;
                let ls: Vec<usize> = match kind.as_str() {
                    "SP" => vec![0, 1],
                    _ => vec![l_from_letter(&kind).ok_or_else(|| perr(i, "unknown shell type"))?],
                };
                if nprim == 0 {
                    return Err(perr(i, "shell without primitives"));
                }
                let mut exps = Vec::with_capacity(nprim);
                let mut coefs = vec![Vec::with_capacity(nprim); ls.len()];
                for k in 0..nprim {
                    let ln = i + 1 + k;
                    let row = lines.get(ln).ok_or_else(|| perr(ln, "unexpected end of file"))?;
                    let vals: Vec<f64> = row
                        .split_whitespace()
                        .map(parse_number)
                        .collect::<Option<_>>()
                        .ok_or_else(|| perr(ln, "bad number"))?;
                    if vals.len() != 1 + ls.len() {
                        return Err(perr(ln, "wrong number of columns"));
                    }
                    if vals[0] <= 0.0 {
                        return Err(perr(ln, "non-positive exponent"));
                    }
                    // Gaussian scale factor: exponents are multiplied by scale^2.
                    exps.push(vals[0] * scale * scale);
                    for (c, v) in coefs.iter_mut().zip(&vals[1..]) {
                        c.push(*v);
                    }
                }
                for (l, c) in ls.into_iter().zip(coefs) {
                    shells.push(ShellDefinition { l, exponents: exps.clone(), coefficients: c });
                }
                i += 1 + nprim;
            }
            elements.insert(z, shells);
            i += 1;
        }
        Ok(BasisSetDefinition { name: name.to_string(), elements })
    }

    /// Shells for atomic number `z`, if present.
    pub fn shells_for(&self, z: u32) -> Option<&[ShellDefinition]> {
        self.elements.get(&z).map(|v| v.as_slice())
    }
}

fn l_from_letter(s: &str) -> Option<usize> {
    ["S", "P", "D", "F", "G", "H"].iter().position(|x| *x == s)
}

fn parse_number(s: &str) -> Option<f64> {
    s.replace(['D', 'd'], "E").parse().ok()
}

/// (2k-1)!! for k >= 0, with (-1)!! = 1.
pub fn odd_double_factorial(k: usize) -> f64 {
    (1..=k).fold(1.0, |acc, i| acc * (2 * i - 1) as f64)
}

/// Normalization constant N(a; l, m, n) of a primitive. theory.typ eq:prim-norm
pub fn primitive_norm(a: f64, lmn: [usize; 3]) -> f64 {
    let l = lmn[0] + lmn[1] + lmn[2];
    let den = odd_double_factorial(lmn[0]) * odd_double_factorial(lmn[1]) * odd_double_factorial(lmn[2]);
    (2.0 * a / PI).powf(0.75) * ((4.0 * a).powi(l as i32) / den).sqrt()
}

/// Relative Cartesian normalization c_lmn. theory.typ eq:cart-norm
pub fn cartesian_norm(lmn: [usize; 3]) -> f64 {
    let l = lmn[0] + lmn[1] + lmn[2];
    let den = odd_double_factorial(lmn[0]) * odd_double_factorial(lmn[1]) * odd_double_factorial(lmn[2]);
    (odd_double_factorial(l) / den).sqrt()
}

/// Cartesian components of angular momentum `l` in lexicographic order (theory.typ section 2.2).
pub fn cartesian_components(l: usize) -> Vec<[usize; 3]> {
    let mut v = Vec::with_capacity((l + 1) * (l + 2) / 2);
    for lx in (0..=l).rev() {
        for ly in (0..=l - lx).rev() {
            v.push([lx, ly, l - lx - ly]);
        }
    }
    v
}

const SQRT3_2: f64 = 0.866_025_403_784_438_6;

/// 5×6 Cartesian→spherical d transformation, rows m = -2..2, columns xx,xy,xz,yy,yz,zz.
/// theory.typ eq:sph-d
pub const CART_TO_SPH_D: [[f64; 6]; 5] = [
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],      // d(-2) = xy
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      // d(-1) = yz
    [-0.5, 0.0, 0.0, -0.5, 0.0, 1.0],    // d(0) = zz - xx/2 - yy/2
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],      // d(+1) = xz
    [SQRT3_2, 0.0, 0.0, -SQRT3_2, 0.0, 0.0], // d(+2) = sqrt(3)/2 (xx - yy)
];

/// A contracted shell placed on an atom.
#[derive(Debug, Clone, PartialEq)]
pub struct Shell {
    /// Angular momentum L.
    pub l: usize,
    /// Centre (Bohr).
    pub center: [f64; 3],
    /// Index of the atom carrying the shell.
    pub atom: usize,
    /// Primitive exponents a_k.
    pub exponents: Vec<f64>,
    /// Normalized contraction coefficients d̃_k (theory.typ eq:contr-norm); multiply
    /// by c_lmn (eq:cart-norm) for each Cartesian component.
    pub coefficients: Vec<f64>,
    /// Whether the shell uses spherical (5d) instead of Cartesian components.
    pub spherical: bool,
    /// Index of the first AO of this shell.
    pub ao_offset: usize,
}

impl Shell {
    /// Builds a shell from file data, applying theory.typ eq:contr-norm.
    pub fn new(def: &ShellDefinition, center: [f64; 3], atom: usize, spherical: bool, ao_offset: usize) -> Self {
        let l = def.l;
        let a = &def.exponents;
        let d = &def.coefficients;
        // theory.typ eq:contr-norm
        let mut s = 0.0;
        for k in 0..a.len() {
            for kk in 0..a.len() {
                s += d[k] * d[kk] * (2.0 * (a[k] * a[kk]).sqrt() / (a[k] + a[kk])).powf(l as f64 + 1.5);
            }
        }
        let n_tilde = 1.0 / s.sqrt();
        let coefficients = a
            .iter()
            .zip(d)
            .map(|(&ak, &dk)| n_tilde * dk * primitive_norm(ak, [l, 0, 0]))
            .collect();
        Shell { l, center, atom, exponents: a.clone(), coefficients, spherical, ao_offset }
    }

    /// Number of Cartesian components.
    pub fn n_cartesian(&self) -> usize {
        (self.l + 1) * (self.l + 2) / 2
    }

    /// Number of basis functions (5 for a spherical d shell).
    pub fn n_functions(&self) -> usize {
        if self.spherical { 2 * self.l + 1 } else { self.n_cartesian() }
    }

    /// Cartesian→output transformation, if the shell is spherical (only d shells).
    pub fn spherical_transform(&self) -> Option<&'static [[f64; 6]; 5]> {
        if self.spherical { Some(&CART_TO_SPH_D) } else { None }
    }
}

/// Options for building a [`Basis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BasisOptions {
    /// Use 6 Cartesian d functions instead of 5 spherical ones.
    pub cartesian_d: bool,
}

/// The AO basis of a molecule.
#[derive(Debug, Clone, PartialEq)]
pub struct Basis {
    shells: Vec<Shell>,
    nao: usize,
}

impl Basis {
    /// Places the basis-set shells on every atom of `mol`.
    pub fn new(mol: &Molecule, def: &BasisSetDefinition, options: BasisOptions) -> Result<Self> {
        let mut shells = Vec::new();
        let mut offset = 0;
        for (ia, atom) in mol.atoms().iter().enumerate() {
            let defs = def.shells_for(atom.z).ok_or_else(|| QcError::MissingBasis {
                basis: def.name.clone(),
                element: element_symbol(atom.z).unwrap_or("?").to_string(),
            })?;
            for sd in defs {
                if sd.l > 2 {
                    return Err(QcError::UnsupportedAngularMomentum(sd.l));
                }
                let spherical = sd.l == 2 && !options.cartesian_d;
                let sh = Shell::new(sd, atom.position, ia, spherical, offset);
                offset += sh.n_functions();
                shells.push(sh);
            }
        }
        Ok(Basis { shells, nao: offset })
    }

    /// Convenience: built-in basis by name with default options (spherical d).
    pub fn from_name(mol: &Molecule, name: &str) -> Result<Self> {
        Basis::new(mol, &BasisSetDefinition::builtin(name)?, BasisOptions::default())
    }

    /// The basis of a single atom of the molecule: its shells, renumbered from AO 0
    /// and attributed to atom 0, at their original centre. Used for the atomic
    /// calculations of the initial guess.
    pub fn atom_basis(&self, atom: usize) -> Basis {
        let mut shells = Vec::new();
        let mut offset = 0;
        for sh in self.shells.iter().filter(|sh| sh.atom == atom) {
            let mut sh = sh.clone();
            sh.atom = 0;
            sh.ao_offset = offset;
            offset += sh.n_functions();
            shells.push(sh);
        }
        Basis { shells, nao: offset }
    }

    /// The shells, in AO order.
    pub fn shells(&self) -> &[Shell] {
        &self.shells
    }

    /// Number of basis functions (AOs).
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// Atom index of every AO.
    pub fn ao_atom(&self) -> Vec<usize> {
        let mut v = Vec::with_capacity(self.nao);
        for s in &self.shells {
            v.extend(std::iter::repeat_n(s.atom, s.n_functions()));
        }
        v
    }
}

/// Applies a Cartesian→spherical transform `t` (rows = output) along `axis` of a
/// row-major tensor with shape `dims`, updating `dims`.
pub(crate) fn transform_axis(data: &[f64], dims: &mut [usize], axis: usize, t: &[[f64; 6]; 5]) -> Vec<f64> {
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();
    let nin = dims[axis];
    debug_assert_eq!(nin, 6);
    let nout = t.len();
    let mut out = vec![0.0; outer * nout * inner];
    for o in 0..outer {
        for (r, row) in t.iter().enumerate() {
            let dst = &mut out[(o * nout + r) * inner..(o * nout + r + 1) * inner];
            for (c, &w) in row.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let src = &data[(o * nin + c) * inner..(o * nin + c + 1) * inner];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += w * s;
                }
            }
        }
    }
    dims[axis] = nout;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_sets_parse() {
        for name in BUILTIN_BASIS_SETS {
            let b = BasisSetDefinition::builtin(name).unwrap();
            for z in 1..=18 {
                assert!(b.shells_for(z).is_some(), "{name} lacks Z={z}");
            }
        }
        let b = BasisSetDefinition::builtin("6-31g*").unwrap();
        // O: S6, SP3 -> S3 + P3, SP1 -> S1 + P1, D1
        let o: Vec<usize> = b.shells_for(8).unwrap().iter().map(|s| s.l).collect();
        assert_eq!(o, vec![0, 0, 1, 0, 1, 2]);
    }

    #[test]
    fn cartesian_order() {
        assert_eq!(cartesian_components(1), vec![[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        assert_eq!(
            cartesian_components(2),
            vec![[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
        );
    }

    #[test]
    fn bad_input() {
        assert!(BasisSetDefinition::from_gbs("x", "H 0\nS 1 1.0\n abc 1.0\n****\n").is_err());
        assert!(BasisSetDefinition::from_gbs("x", "Xx 0\n****\n").is_err());
        assert!(BasisSetDefinition::from_gbs("x", "H 0\nS 2 1.0\n 1.0 1.0\n").is_err());
    }
}

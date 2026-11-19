//! Gaussian basis sets: Gaussian94 parser, contracted shells, normalization,
//! spherical d and f transformations and AO bookkeeping (theory.typ sec:basis).

use crate::error::{QcError, Result};
use crate::molecule::{Molecule, atomic_number, element_symbol};
use std::collections::HashMap;
use std::f64::consts::PI;

// Basis-set data as downloaded from Basis Set Exchange; the exponents and coefficients
// have not been compared with the original papers: UNCHECKED(Hehre1969)
// UNCHECKED(Hehre1972) UNCHECKED(Hariharan1973) UNCHECKED(Francl1982)
const STO_3G: &str = include_str!("../data/basis/sto-3g.gbs");
const B6_31G: &str = include_str!("../data/basis/6-31g.gbs");
const B6_31GS: &str = include_str!("../data/basis/6-31gs.gbs");
const B6_31GSS: &str = include_str!("../data/basis/6-31gss.gbs");
// 6-311G family and the sets with f functions, likewise unchecked (full reference list in
// data/basis/README.md): UNCHECKED(Krishnan1980) UNCHECKED(McLean1980) UNCHECKED(Frisch1984)
// UNCHECKED(Clark1983)
const B6_311G: &str = include_str!("../data/basis/6-311g.gbs");
const B6_311GS: &str = include_str!("../data/basis/6-311gs.gbs");
const B6_311GSS: &str = include_str!("../data/basis/6-311gss.gbs");
const B6_31G_2DF_P: &str = include_str!("../data/basis/6-31g_2df_p.gbs");
const B6_311G_2DF_2PD: &str = include_str!("../data/basis/6-311g_2df_2pd.gbs");
const B6_311PPG_3DF_3PD: &str = include_str!("../data/basis/6-311ppg_3df_3pd.gbs");

/// Highest shell angular momentum accepted by [`Basis::new`] (f functions).
const MAX_SHELL_L: usize = 3;

/// Names of the built-in basis sets.
pub const BUILTIN_BASIS_SETS: [&str; 10] = [
    "sto-3g",
    "6-31g",
    "6-31g*",
    "6-31g**",
    "6-311g",
    "6-311g*",
    "6-311g**",
    "6-31g(2df,p)",
    "6-311g(2df,2pd)",
    "6-311++g(3df,3pd)",
];

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
    /// One of the embedded basis sets (case-insensitive): "sto-3g", "6-31g", "6-31g*",
    /// "6-31g**", "6-311g", "6-311g*", "6-311g**" (H–Ar); "6-31g(2df,p)" (H–Ar);
    /// "6-311g(2df,2pd)" (H–Ne); "6-311++g(3df,3pd)" (H, Li–Ar). "s" may replace "*".
    pub fn builtin(name: &str) -> Result<Self> {
        let text = match name.to_ascii_lowercase().as_str() {
            "sto-3g" => STO_3G,
            "6-31g" => B6_31G,
            "6-31g*" | "6-31gs" => B6_31GS,
            "6-31g**" | "6-31gss" => B6_31GSS,
            "6-311g" => B6_311G,
            "6-311g*" | "6-311gs" => B6_311GS,
            "6-311g**" | "6-311gss" => B6_311GSS,
            "6-31g(2df,p)" => B6_31G_2DF_P,
            "6-311g(2df,2pd)" => B6_311G_2DF_2PD,
            "6-311++g(3df,3pd)" => B6_311PPG_3DF_3PD,
            _ => return Err(QcError::InvalidInput(format!("unknown basis set '{name}'"))),
        };
        Self::from_gbs(name, text)
    }

    /// Parses Gaussian94-format text (theory.typ sec:basis-file).
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

/// Normalization constant N(a; l, m, n) of a primitive. theory.typ eq:prim-norm UNCHECKED(Helgaker2000)
pub fn primitive_norm(a: f64, lmn: [usize; 3]) -> f64 {
    let l = lmn[0] + lmn[1] + lmn[2];
    let den = odd_double_factorial(lmn[0]) * odd_double_factorial(lmn[1]) * odd_double_factorial(lmn[2]);
    (2.0 * a / PI).powf(0.75) * ((4.0 * a).powi(l as i32) / den).sqrt()
}

/// Relative Cartesian normalization c_lmn. theory.typ eq:cart-norm UNCHECKED(Helgaker2000)
pub fn cartesian_norm(lmn: [usize; 3]) -> f64 {
    let l = lmn[0] + lmn[1] + lmn[2];
    let den = odd_double_factorial(lmn[0]) * odd_double_factorial(lmn[1]) * odd_double_factorial(lmn[2]);
    (odd_double_factorial(l) / den).sqrt()
}

/// Cartesian components of angular momentum `l` in lexicographic order (theory.typ sec:contracted).
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

/// A Cartesian→spherical transformation: one row per spherical function (m = -L..L),
/// each row with one weight per normalized Cartesian component in lexicographic order.
pub type SphericalTransform = [&'static [f64]];

/// 5×6 Cartesian→spherical d transformation, rows m = -2..2, columns xx,xy,xz,yy,yz,zz.
/// theory.typ eq:sph-d UNCHECKED(Schlegel1995)
pub const CART_TO_SPH_D: &SphericalTransform = &[
    &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],      // d(-2) = xy
    &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      // d(-1) = yz
    &[-0.5, 0.0, 0.0, -0.5, 0.0, 1.0],    // d(0) = zz - xx/2 - yy/2
    &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],      // d(+1) = xz
    &[SQRT3_2, 0.0, 0.0, -SQRT3_2, 0.0, 0.0], // d(+2) = sqrt(3)/2 (xx - yy)
];

/// a = sqrt(5/8) of eq:sph-f.
const F_A: f64 = 0.790_569_415_042_094_9;
/// b = sqrt(3/8) of eq:sph-f.
const F_B: f64 = 0.612_372_435_695_794_5;
/// 3a/sqrt(5) = 3/sqrt(8).
const F_3A_R5: f64 = 1.060_660_171_779_821_2;
/// b/sqrt(5) = sqrt(3/40).
const F_B_R5: f64 = 0.273_861_278_752_583_04;
/// 4b/sqrt(5).
const F_4B_R5: f64 = 1.095_445_115_010_332_1;
/// 3/(2 sqrt(5)).
const F_3_2R5: f64 = 0.670_820_393_249_936_9;

/// 7×10 Cartesian→spherical f transformation, rows m = -3..3, columns
/// xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz. theory.typ eq:sph-f UNCHECKED(Schlegel1995)
pub const CART_TO_SPH_F: &SphericalTransform = &[
    &[0.0, F_3A_R5, 0.0, 0.0, 0.0, 0.0, -F_A, 0.0, 0.0, 0.0], // f(-3) = 3a/√5 xxy - a yyy
    &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],      // f(-2) = xyz
    &[0.0, -F_B_R5, 0.0, 0.0, 0.0, 0.0, -F_B, 0.0, F_4B_R5, 0.0], // f(-1) = 4b/√5 yzz - b/√5 xxy - b yyy
    &[0.0, 0.0, -F_3_2R5, 0.0, 0.0, 0.0, 0.0, -F_3_2R5, 0.0, 1.0], // f(0) = zzz - 3/(2√5) (xxz + yyz)
    &[-F_B, 0.0, 0.0, -F_B_R5, 0.0, F_4B_R5, 0.0, 0.0, 0.0, 0.0], // f(+1) = 4b/√5 xzz - b xxx - b/√5 xyy
    &[0.0, 0.0, SQRT3_2, 0.0, 0.0, 0.0, 0.0, -SQRT3_2, 0.0, 0.0], // f(+2) = sqrt(3)/2 (xxz - yyz)
    &[F_A, 0.0, 0.0, -F_3A_R5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // f(+3) = a xxx - 3a/√5 xyy
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
    /// Normalized contraction coefficients d̃_k (theory.typ eq:contr-norm); multiply UNCHECKED(Helgaker2000)
    /// by c_lmn (eq:cart-norm) for each Cartesian component.
    pub coefficients: Vec<f64>,
    /// Whether the shell uses spherical (5d, 7f) instead of Cartesian components.
    pub spherical: bool,
    /// Index of the first AO of this shell.
    pub ao_offset: usize,
}

impl Shell {
    /// Builds a shell from file data, applying theory.typ eq:contr-norm.
    ///
    /// # Panics
    /// If `spherical` is set for an angular momentum without a spherical
    /// transformation (L other than 2 or 3; s and p shells are always Cartesian).
    pub fn new(def: &ShellDefinition, center: [f64; 3], atom: usize, spherical: bool, ao_offset: usize) -> Self {
        let l = def.l;
        assert!(!spherical || matches!(l, 2 | 3), "no spherical transformation for L = {l}");
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

    /// Number of basis functions (5 for a spherical d shell, 7 for a spherical f shell).
    pub fn n_functions(&self) -> usize {
        if self.spherical { 2 * self.l + 1 } else { self.n_cartesian() }
    }

    /// Cartesian→output transformation, if the shell is spherical
    /// (eq:sph-d for d shells, eq:sph-f for f shells).
    ///
    /// # Panics
    /// If the shell is marked spherical but has no transformation (L other than 2 or 3).
    pub fn spherical_transform(&self) -> Option<&'static SphericalTransform> {
        if !self.spherical {
            return None;
        }
        match self.l {
            2 => Some(CART_TO_SPH_D),
            3 => Some(CART_TO_SPH_F),
            l => panic!("no spherical transformation for L = {l}"),
        }
    }
}

/// Options for building a [`Basis`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BasisOptions {
    /// Use Cartesian d and f functions (6d, 10f) instead of spherical ones (5d, 7f),
    /// like PySCF's `cart=True`. The name is kept from when shells stopped at d.
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
                if sd.l > MAX_SHELL_L {
                    return Err(QcError::UnsupportedAngularMomentum(sd.l));
                }
                // eq:sph-d, eq:sph-f; s and p shells stay Cartesian.
                let spherical = sd.l >= 2 && !options.cartesian_d;
                let sh = Shell::new(sd, atom.position, ia, spherical, offset);
                offset += sh.n_functions();
                shells.push(sh);
            }
        }
        Ok(Basis { shells, nao: offset })
    }

    /// Convenience: built-in basis by name with default options (spherical d and f).
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
pub(crate) fn transform_axis(data: &[f64], dims: &mut [usize], axis: usize, t: &SphericalTransform) -> Vec<f64> {
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();
    let nin = dims[axis];
    debug_assert!(t.iter().all(|row| row.len() == nin));
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

    /// One-centre overlap of the normalized Cartesian components of angular momentum `l`
    /// (common radial factor dropped): ⟨x^p⟩ over the sphere ∝ Π (p_i - 1)!! for even p_i.
    fn cartesian_angular_overlap(l: usize) -> Vec<Vec<f64>> {
        let comps = cartesian_components(l);
        let raw = |p: [usize; 3], q: [usize; 3]| -> f64 {
            let s = [p[0] + q[0], p[1] + q[1], p[2] + q[2]];
            if s.iter().any(|x| x % 2 == 1) {
                return 0.0;
            }
            s.iter().map(|&x| odd_double_factorial(x / 2)).product()
        };
        comps
            .iter()
            .map(|&p| comps.iter().map(|&q| raw(p, q) / (raw(p, p) * raw(q, q)).sqrt()).collect())
            .collect()
    }

    /// T S Tᵀ = I for eq:sph-d and eq:sph-f, and the f rows are orthogonal to r²p.
    #[test]
    fn spherical_transforms_orthonormal() {
        for (l, t) in [(2, CART_TO_SPH_D), (3, CART_TO_SPH_F)] {
            let s = cartesian_angular_overlap(l);
            let ncart = (l + 1) * (l + 2) / 2;
            assert_eq!(t.len(), 2 * l + 1);
            for row in t {
                assert_eq!(row.len(), ncart);
            }
            for (i, ri) in t.iter().enumerate() {
                for (j, rj) in t.iter().enumerate() {
                    let mut v = 0.0;
                    for a in 0..ncart {
                        for b in 0..ncart {
                            v += ri[a] * s[a][b] * rj[b];
                        }
                    }
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((v - expect).abs() < 1e-14, "L={l} (T S Tᵀ)[{i}][{j}] = {v}");
                }
            }
        }
        // r² x_k = Σ_i x_k x_i² over normalized components: monomial = χ / c_lmn (eq:cart-norm).
        let comps = cartesian_components(3);
        let s = cartesian_angular_overlap(3);
        for k in 0..3 {
            let v: Vec<f64> = comps
                .iter()
                .map(|&p| {
                    let odd_k = p[k] % 2 == 1 && (0..3).all(|j| j == k || p[j] % 2 == 0);
                    if odd_k { 1.0 / cartesian_norm(p) } else { 0.0 }
                })
                .collect();
            for (m, row) in CART_TO_SPH_F.iter().enumerate() {
                let mut ov = 0.0;
                for a in 0..10 {
                    for b in 0..10 {
                        ov += row[a] * s[a][b] * v[b];
                    }
                }
                assert!(ov.abs() < 1e-14, "f({}) not orthogonal to r² p_{k}: {ov}", m as i64 - 3);
            }
        }
    }

    #[test]
    fn spherical_transform_by_l() {
        let def = ShellDefinition { l: 3, exponents: vec![1.0], coefficients: vec![1.0] };
        let f = Shell::new(&def, [0.0; 3], 0, true, 0);
        assert_eq!(f.n_functions(), 7);
        assert_eq!(f.spherical_transform().unwrap(), CART_TO_SPH_F);
        let fc = Shell::new(&def, [0.0; 3], 0, false, 0);
        assert!(fc.spherical_transform().is_none());
        let d = Shell::new(&ShellDefinition { l: 2, ..def }, [0.0; 3], 0, true, 0);
        assert_eq!(d.spherical_transform().unwrap(), CART_TO_SPH_D);
    }

    #[test]
    fn builtin_sets_parse() {
        let h_ar: Vec<u32> = (1..=18).collect();
        let h_ne: Vec<u32> = (1..=10).collect();
        let h_li_ar: Vec<u32> = std::iter::once(1).chain(3..=18).collect();
        let elements = |name: &str| -> &[u32] {
            match name {
                "6-311g(2df,2pd)" => &h_ne,
                "6-311++g(3df,3pd)" => &h_li_ar,
                _ => &h_ar,
            }
        };
        for name in BUILTIN_BASIS_SETS {
            let b = BasisSetDefinition::builtin(name).unwrap();
            let expect = elements(name);
            for z in 1..=18 {
                assert_eq!(b.shells_for(z).is_some(), expect.contains(&z), "{name}, Z={z}");
            }
            // f shells only in the sets named with "f", and nothing above f.
            let max_l = expect.iter().flat_map(|&z| b.shells_for(z).unwrap()).map(|s| s.l).max().unwrap();
            assert_eq!(max_l, if name.contains('f') { 3 } else if name.contains('*') { 2 } else { 1 }, "{name}");
        }
        let l_of = |name: &str, z: u32| -> Vec<usize> {
            BasisSetDefinition::builtin(name).unwrap().shells_for(z).unwrap().iter().map(|s| s.l).collect()
        };
        // O: S6, SP3 -> S3 + P3, SP1 -> S1 + P1, D1
        assert_eq!(l_of("6-31g*", 8), vec![0, 0, 1, 0, 1, 2]);
        // Shells are kept in file order, whatever it is.
        // 6-31G(2df,p) H: S3, S1, P1, P1 (two P shells).
        assert_eq!(l_of("6-31g(2df,p)", 1), vec![0, 0, 1, 1]);
        // 6-311++G(3df,3pd) Al: S×6, P×4, SP, P, D×3, F.
        assert_eq!(l_of("6-311++g(3df,3pd)", 13), vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 2, 2, 2, 3]);
        // 6-311G(2df,2pd) O: S, SP×3, D×2, F.
        assert_eq!(l_of("6-311g(2df,2pd)", 8), vec![0, 0, 1, 0, 1, 0, 1, 2, 2, 3]);
        // Aliases and case.
        for (alias, name) in [("6-311gs", "6-311g*"), ("6-311GSS", "6-311g**"), ("6-31G(2DF,P)", "6-31g(2df,p)")] {
            assert_eq!(
                BasisSetDefinition::builtin(alias).unwrap().shells_for(8),
                BasisSetDefinition::builtin(name).unwrap().shells_for(8)
            );
        }
    }

    #[test]
    fn angular_momentum_limit() {
        let mol = Molecule::from_symbols(&["H"], &[[0.0; 3]], crate::molecule::Unit::Bohr, 0).unwrap();
        let gbs = |kind: &str| format!("H 0
{kind} 1 1.00
 1.0 1.0
****
");
        for cart in [false, true] {
            let opts = BasisOptions { cartesian_d: cart };
            let f = Basis::new(&mol, &BasisSetDefinition::from_gbs("f", &gbs("F")).unwrap(), opts).unwrap();
            assert_eq!(f.nao(), if cart { 10 } else { 7 });
            assert_eq!(f.shells()[0].spherical, !cart);
            let g = Basis::new(&mol, &BasisSetDefinition::from_gbs("g", &gbs("G")).unwrap(), opts);
            assert!(matches!(g, Err(QcError::UnsupportedAngularMomentum(4))));
        }
    }

    #[test]
    #[should_panic(expected = "no spherical transformation for L = 1")]
    fn spherical_p_shell_rejected() {
        let def = ShellDefinition { l: 1, exponents: vec![1.0], coefficients: vec![1.0] };
        Shell::new(&def, [0.0; 3], 0, true, 0);
    }

    #[test]
    #[should_panic(expected = "no spherical transformation for L = 4")]
    fn spherical_g_shell_rejected() {
        let def = ShellDefinition { l: 4, exponents: vec![1.0], coefficients: vec![1.0] };
        Shell::new(&def, [0.0; 3], 0, true, 0);
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

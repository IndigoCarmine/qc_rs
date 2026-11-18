//! DFT-D3 building blocks (theory.typ section sec:d3): the reference data, the
//! coordination number with its nuclear derivatives (eq:d3-cn) and the Gaussian UNCHECKED(Grimme2010) UNCHECKED(Pyykko2009)
//! interpolation of C6 with its derivatives with respect to the coordination
//! numbers (eq:d3-c6), the damping parameters of D3(BJ) and D3(0) and the UNCHECKED(Grimme2010)
//! two-body dispersion energy (eq:d3-e, eq:d3-c8, eq:d3-bj, eq:d3-zero) with its UNCHECKED(Grimme2010) UNCHECKED(Grimme2011) UNCHECKED(Johnson2005)
//! nuclear gradient (eq:d3-grad).

use crate::error::{QcError, Result};
use crate::molecule::{Atom, Molecule, element_symbol};
use std::sync::OnceLock;

// DFT-D3 reference data for H–Ar (covalent radii, r4r2, reference coordination
// numbers, reference C6, pairwise R0). Not original work of qc_rs: extracted from
// tad-dftd3 0.7.0 and tad-mctc 0.8.0, distributed under the Apache License 2.0
// (data/dispersion/LICENSE-APACHE-2.0.txt). The values go back to Grimme et al.
// (2010); source, modifications, units and the meaning of every column are in
// data/dispersion/README.md. A compiled program embeds this text and must carry
// that attribution and license. UNCHECKED(Grimme2010)
const D3_DATA: &str = include_str!("../data/dispersion/d3_h_ar.dat");

/// Number of elements covered by the D3 data (H–Ar).
pub const D3_MAX_Z: u32 = 18;

/// Steepness k1 of the counting function, theory.typ eq:d3-cn.
pub const K1: f64 = 16.0;
/// Width k3 of the Gaussian weights, theory.typ eq:d3-c6.
pub const K3: f64 = 4.0;

const N_ELEM: usize = D3_MAX_Z as usize;

/// Per-element D3 data.
#[derive(Debug, Clone, PartialEq)]
struct ElementData {
    /// k2·R_cov in Bohr (k2 = 4/3 is already applied in the data file).
    rcov: f64,
    /// sqrt(Q) in Bohr, as used in C8 = 3 C6 r4r2_A r4r2_B.
    r4r2: f64,
    /// Reference coordination numbers, in the order of the data file.
    cnref: Vec<f64>,
}

/// The DFT-D3 reference data for H–Ar (see `data/dispersion/README.md`).
#[derive(Debug, Clone, PartialEq)]
pub struct D3Data {
    elements: Vec<ElementData>,
    /// r0ab[(za-1)*18 + (zb-1)], symmetric, Bohr.
    r0ab: Vec<f64>,
    /// c6[(za-1)*18 + (zb-1)][i*nref(zb) + j]: reference C6 for reference i of za
    /// and reference j of zb, stored for both orders of the pair.
    c6: Vec<Vec<f64>>,
}

/// C6 of an atom pair at given coordination numbers and its partial derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C6Value {
    /// C6^AB in E_h a0^6.
    pub c6: f64,
    /// ∂C6^AB/∂CN_A.
    pub dc6_dcn_a: f64,
    /// ∂C6^AB/∂CN_B.
    pub dc6_dcn_b: f64,
}

/// Coordination numbers of all atoms and their derivatives with respect to the
/// nuclear positions.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinationNumbers {
    /// CN_A for every atom A.
    pub cn: Vec<f64>,
    /// dcn_dr[a * n + c] = dCN_a / dR_c (Bohr⁻¹), n = number of atoms.
    pub dcn_dr: Vec<[f64; 3]>,
}

/// The embedded D3 data, parsed once.
pub fn d3_data() -> Result<&'static D3Data> {
    static DATA: OnceLock<Result<D3Data>> = OnceLock::new();
    DATA.get_or_init(|| D3Data::parse(D3_DATA)).as_ref().map_err(Clone::clone)
}

fn element_index(z: u32) -> Result<usize> {
    if (1..=D3_MAX_Z).contains(&z) {
        Ok(z as usize - 1)
    } else {
        Err(QcError::UnknownElement(format!("Z = {z} (DFT-D3 data cover H–Ar only)")))
    }
}

impl D3Data {
    /// Parses the text format of `data/dispersion/d3_h_ar.dat` (LF or CRLF line
    /// endings). Every element H–Ar, every R0 pair and every reference C6 must be
    /// given exactly once.
    pub fn parse(text: &str) -> Result<Self> {
        let mut elements: Vec<Option<ElementData>> = vec![None; N_ELEM];
        let mut r0ab = vec![f64::NAN; N_ELEM * N_ELEM];
        // Raw C6 records, keyed by (za, zb) with za <= zb, before the reference
        // counts are known for certain.
        let mut c6_records: Vec<(usize, usize, usize, usize, f64, usize)> = Vec::new();
        let perr = |ln: usize, msg: &str| QcError::Parse(format!("D3 data: line {}: {msg}", ln + 1));
        let num = |ln: usize, s: &str| -> Result<f64> {
            s.parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| perr(ln, &format!("invalid number '{s}'")))
        };
        let int = |ln: usize, s: &str| -> Result<usize> {
            s.parse::<usize>().map_err(|_| perr(ln, &format!("invalid integer '{s}'")))
        };
        let elem = |ln: usize, s: &str| -> Result<usize> {
            let z = int(ln, s)?;
            if (1..=N_ELEM).contains(&z) { Ok(z - 1) } else { Err(perr(ln, &format!("atomic number {z} outside 1..18"))) }
        };

        // str::lines strips "\n" and "\r\n"; trim also removes a stray "\r".
        for (ln, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            match f[0] {
                "element" => {
                    if f.len() < 6 {
                        return Err(perr(ln, "element record too short"));
                    }
                    let a = elem(ln, f[1])?;
                    if element_symbol(a as u32 + 1) != Some(f[2]) {
                        return Err(perr(ln, &format!("symbol '{}' does not match Z = {}", f[2], a + 1)));
                    }
                    let nref = int(ln, f[5])?;
                    if nref == 0 || f.len() != 6 + nref {
                        return Err(perr(ln, "number of reference CNs does not match nref"));
                    }
                    let cnref = f[6..].iter().map(|s| num(ln, s)).collect::<Result<Vec<_>>>()?;
                    if elements[a].is_some() {
                        return Err(perr(ln, "duplicate element record"));
                    }
                    elements[a] = Some(ElementData { rcov: num(ln, f[3])?, r4r2: num(ln, f[4])?, cnref });
                }
                "r0ab" => {
                    if f.len() != 4 {
                        return Err(perr(ln, "r0ab record must have 4 fields"));
                    }
                    let (a, b) = (elem(ln, f[1])?, elem(ln, f[2])?);
                    if a > b {
                        return Err(perr(ln, "r0ab record must have Za <= Zb"));
                    }
                    if !r0ab[a * N_ELEM + b].is_nan() {
                        return Err(perr(ln, "duplicate r0ab record"));
                    }
                    let v = num(ln, f[3])?;
                    r0ab[a * N_ELEM + b] = v;
                    r0ab[b * N_ELEM + a] = v;
                }
                "c6" => {
                    if f.len() != 6 {
                        return Err(perr(ln, "c6 record must have 6 fields"));
                    }
                    let (a, b) = (elem(ln, f[1])?, elem(ln, f[2])?);
                    if a > b {
                        return Err(perr(ln, "c6 record must have Za <= Zb"));
                    }
                    let (i, j) = (int(ln, f[3])?, int(ln, f[4])?);
                    if i == 0 || j == 0 {
                        return Err(perr(ln, "reference indices are 1-based"));
                    }
                    c6_records.push((a, b, i - 1, j - 1, num(ln, f[5])?, ln));
                }
                other => return Err(perr(ln, &format!("unknown record '{other}'"))),
            }
        }

        let elements = elements
            .into_iter()
            .enumerate()
            .map(|(a, e)| e.ok_or_else(|| QcError::Parse(format!("D3 data: no element record for Z = {}", a + 1))))
            .collect::<Result<Vec<_>>>()?;
        if let Some(k) = r0ab.iter().position(|v| v.is_nan()) {
            return Err(QcError::Parse(format!("D3 data: no r0ab for Z = {} / {}", k / N_ELEM + 1, k % N_ELEM + 1)));
        }

        let mut c6: Vec<Vec<f64>> = (0..N_ELEM * N_ELEM)
            .map(|k| vec![f64::NAN; elements[k / N_ELEM].cnref.len() * elements[k % N_ELEM].cnref.len()])
            .collect();
        for &(a, b, i, j, v, ln) in &c6_records {
            let (na, nb) = (elements[a].cnref.len(), elements[b].cnref.len());
            if i >= na || j >= nb {
                return Err(perr(ln, "reference index out of range"));
            }
            if !c6[a * N_ELEM + b][i * nb + j].is_nan() {
                return Err(perr(ln, "duplicate c6 record"));
            }
            c6[a * N_ELEM + b][i * nb + j] = v;
            if a != b {
                // C6(Zb, Za, j, i) = C6(Za, Zb, i, j) (data/dispersion/README.md).
                c6[b * N_ELEM + a][j * na + i] = v;
            }
        }
        if let Some(k) = c6.iter().position(|t| t.iter().any(|v| v.is_nan())) {
            return Err(QcError::Parse(format!(
                "D3 data: incomplete c6 records for Z = {} / {}",
                k / N_ELEM + 1,
                k % N_ELEM + 1
            )));
        }
        Ok(D3Data { elements, r0ab, c6 })
    }

    /// k2·R_cov of element `z` in Bohr (k2 = 4/3 included), theory.typ eq:d3-cn.
    pub fn scaled_covalent_radius(&self, z: u32) -> Result<f64> {
        Ok(self.elements[element_index(z)?].rcov)
    }

    /// The r4r2 factor of element `z` in Bohr (sqrt(Q) of theory.typ eq:d3-c8). UNCHECKED(Grimme2010)
    pub fn r4r2(&self, z: u32) -> Result<f64> {
        Ok(self.elements[element_index(z)?].r4r2)
    }

    /// Pairwise cutoff radius R0^AB in Bohr (zero damping, theory.typ eq:d3-zero). UNCHECKED(Grimme2010)
    pub fn r0ab(&self, za: u32, zb: u32) -> Result<f64> {
        Ok(self.r0ab[element_index(za)? * N_ELEM + element_index(zb)?])
    }

    /// Reference coordination numbers CN^i of element `z`.
    pub fn reference_cn(&self, z: u32) -> Result<&[f64]> {
        Ok(&self.elements[element_index(z)?].cnref)
    }

    /// Reference C6 for reference `i` of `za` and reference `j` of `zb` (0-based).
    pub fn reference_c6(&self, za: u32, zb: u32, i: usize, j: usize) -> Result<f64> {
        let (a, b) = (element_index(za)?, element_index(zb)?);
        let (na, nb) = (self.elements[a].cnref.len(), self.elements[b].cnref.len());
        if i >= na || j >= nb {
            return Err(QcError::InvalidInput(format!("reference index ({i}, {j}) out of range for Z = {za} / {zb}")));
        }
        Ok(self.c6[a * N_ELEM + b][i * nb + j])
    }

    /// Coordination numbers of all atoms (positions in Bohr) and dCN_a/dR_c.
    /// theory.typ eq:d3-cn and the derivative dg_AB/dr of the gradient section.
    pub fn coordination_numbers(&self, atoms: &[Atom]) -> Result<CoordinationNumbers> {
        let n = atoms.len();
        let rcov = atoms
            .iter()
            .map(|a| self.scaled_covalent_radius(a.z))
            .collect::<Result<Vec<_>>>()?;
        let mut cn = vec![0.0; n];
        let mut dcn_dr = vec![[0.0; 3]; n * n];
        for a in 0..n {
            for b in 0..a {
                let d: [f64; 3] = std::array::from_fn(|k| atoms[a].position[k] - atoms[b].position[k]);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if r == 0.0 {
                    return Err(QcError::InvalidInput(format!("atoms {a} and {b} coincide")));
                }
                // g = 1 / (1 + e^u), u = -k1 (R^c / r - 1), R^c = k2 (R_A + R_B).
                let rc = rcov[a] + rcov[b];
                let g = 1.0 / (1.0 + (-K1 * (rc / r - 1.0)).exp());
                let dg = -g * (1.0 - g) * K1 * rc / (r * r);
                cn[a] += g;
                cn[b] += g;
                // dr/dR_a = d / r, dr/dR_b = -d / r.
                for k in 0..3 {
                    let v = dg * d[k] / r;
                    dcn_dr[a * n + a][k] += v;
                    dcn_dr[a * n + b][k] -= v;
                    dcn_dr[b * n + a][k] += v;
                    dcn_dr[b * n + b][k] -= v;
                }
            }
        }
        Ok(CoordinationNumbers { cn, dcn_dr })
    }

    /// C6^AB of elements `za`, `zb` at coordination numbers `cn_a`, `cn_b`, with
    /// ∂C6/∂CN_A and ∂C6/∂CN_B. theory.typ eq:d3-c6 (with the shift by d_min given
    /// there, so that the weights cannot all underflow).
    pub fn c6(&self, za: u32, zb: u32, cn_a: f64, cn_b: f64) -> Result<C6Value> {
        let (a, b) = (element_index(za)?, element_index(zb)?);
        let (ra, rb) = (&self.elements[a].cnref, &self.elements[b].cnref);
        let table = &self.c6[a * N_ELEM + b];
        let nb = rb.len();
        let dist = |i: usize, j: usize| (cn_a - ra[i]).powi(2) + (cn_b - rb[j]).powi(2);
        let mut dmin = f64::INFINITY;
        for i in 0..ra.len() {
            for j in 0..nb {
                dmin = dmin.min(dist(i, j));
            }
        }
        // Z = Σ C L, W = Σ L, and their derivatives; ∂L/∂CN_A = -2 k3 (CN_A - CN_A^i) L.
        let (mut z, mut w) = (0.0, 0.0);
        let (mut za_d, mut wa_d, mut zb_d, mut wb_d) = (0.0, 0.0, 0.0, 0.0);
        for (i, &cri) in ra.iter().enumerate() {
            for (j, &crj) in rb.iter().enumerate() {
                let l = (-K3 * (dist(i, j) - dmin)).exp();
                let c = table[i * nb + j];
                let dla = -2.0 * K3 * (cn_a - cri) * l;
                let dlb = -2.0 * K3 * (cn_b - crj) * l;
                z += c * l;
                w += l;
                za_d += c * dla;
                wa_d += dla;
                zb_d += c * dlb;
                wb_d += dlb;
            }
        }
        Ok(C6Value {
            c6: z / w,
            dc6_dcn_a: (za_d * w - z * wa_d) / (w * w),
            dc6_dcn_b: (zb_d * w - z * wb_d) / (w * w),
        })
    }
}

/// Damping function of the D3 two-body term and its parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum D3Damping {
    /// Becke–Johnson (rational) damping, theory.typ eq:d3-bj. `a2` is in Bohr.
    Bj { s6: f64, a1: f64, s8: f64, a2: f64 },
    /// Zero damping, theory.typ eq:d3-zero, with α8 = α6 + 2 (sec:d3).
    Zero { s6: f64, sr6: f64, s8: f64, sr8: f64, alpha6: f64 },
}

/// Which damping to take from the published parameter tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum D3Variant {
    /// D3(BJ).
    Bj,
    /// D3(0).
    Zero,
}

/// Parameters of a DFT-D3 correction. Only the two-body term is implemented.
/// Construct with [`D3Parameters::for_functional`] or [`D3Parameters::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct D3Parameters {
    /// The damping and its parameters.
    pub damping: D3Damping,
}

impl D3Parameters {
    /// Parameters given explicitly. Every parameter must be finite. They set the
    /// damping radius and steepness, so `sr6`, `sr8` and `alpha6` must be positive,
    /// and `a1`, `a2` non-negative and not both zero (the BJ radius a1 R0 + a2 must
    /// be positive).
    pub fn new(damping: D3Damping) -> Result<Self> {
        let (all, ok): (&[f64], bool) = match &damping {
            D3Damping::Bj { s6, a1, s8, a2 } => (&[*s6, *a1, *s8, *a2], *a1 >= 0.0 && *a2 >= 0.0 && *a1 + *a2 > 0.0),
            D3Damping::Zero { s6, sr6, s8, sr8, alpha6 } => {
                (&[*s6, *sr6, *s8, *sr8, *alpha6], *sr6 > 0.0 && *sr8 > 0.0 && *alpha6 > 0.0)
            }
        };
        if all.iter().any(|v| !v.is_finite()) || !ok {
            return Err(QcError::InvalidInput(format!("invalid D3 damping parameters {damping:?}")));
        }
        Ok(D3Parameters { damping })
    }

    /// The published parameters of a functional (table in theory.typ sec:d3).
    /// Names are matched like [`crate::Functional::from_name`] (case and white
    /// space ignored): "b3lyp", "b3lypg", "b3lyp5" and "cam-b3lyp" ("camb3lyp",
    /// "cam_b3lyp"). B3LYP with VWN-RPA and with VWN5 get the same values: which of
    /// the two the parameters were fitted to is unconfirmed (theory.typ sec:d3).
    /// Other functionals are an error.
    pub fn for_functional(name: &str, variant: D3Variant) -> Result<Self> {
        let key: String = name.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_ascii_lowercase();
        let damping = match (key.as_str(), variant) {
            ("b3lyp" | "b3lypg" | "b3lyp5", D3Variant::Bj) => D3Damping::Bj { s6: 1.0, a1: 0.3981, s8: 1.9889, a2: 4.4211 },
            ("b3lyp" | "b3lypg" | "b3lyp5", D3Variant::Zero) => {
                D3Damping::Zero { s6: 1.0, sr6: 1.261, s8: 1.703, sr8: 1.0, alpha6: 14.0 }
            }
            ("cam-b3lyp" | "camb3lyp" | "cam_b3lyp", D3Variant::Bj) => {
                D3Damping::Bj { s6: 1.0, a1: 0.3708, s8: 2.0674, a2: 5.4743 }
            }
            ("cam-b3lyp" | "camb3lyp" | "cam_b3lyp", D3Variant::Zero) => {
                D3Damping::Zero { s6: 1.0, sr6: 1.378, s8: 1.217, sr8: 1.0, alpha6: 14.0 }
            }
            _ => {
                return Err(QcError::InvalidInput(format!(
                    "no DFT-D3 {variant:?} parameters for functional '{name}' (available: b3lyp, b3lyp5, cam-b3lyp)"
                )));
            }
        };
        Self::new(damping)
    }
}

/// Two-body energy e_AB of one atom pair at distance `r` (theory.typ eq:d3-e) and
/// its derivative ∂e_AB/∂r at fixed C6 (gradient section, the equation before
/// eq:d3-grad), with C8 from eq:d3-c8 and the damping of eq:d3-bj or eq:d3-zero.
/// The tabulated R0^AB is looked up for zero damping only.
fn pair_energy(damping: &D3Damping, data: &D3Data, za: u32, zb: u32, c6: f64, r: f64) -> Result<(f64, f64)> {
    // eq:d3-c8: C8 = 3 C6 sqrt(Q_A Q_B), with r4r2 = sqrt(Q).
    let q = 3.0 * data.r4r2(za)? * data.r4r2(zb)?;
    let c8 = c6 * q;
    let (r2, r6) = (r * r, r.powi(6));
    let r8 = r6 * r2;
    Ok(match *damping {
        D3Damping::Bj { s6, a1, s8, a2 } => {
            // eq:d3-bj: R0 = sqrt(C8 / C6) = sqrt(q), independent of the CN.
            let rd = a1 * q.sqrt() + a2;
            let rd6 = rd.powi(6);
            let (den6, den8) = (r6 + rd6, r8 + rd6 * rd * rd);
            let e = -(s6 * c6 / den6 + s8 * c8 / den8);
            // ∂e/∂r = Σ_n s_n C_n n r^(n-1) / (r^n + R^n)^2.
            let de = s6 * c6 * 6.0 * r6 / r / (den6 * den6) + s8 * c8 * 8.0 * r8 / r / (den8 * den8);
            (e, de)
        }
        D3Damping::Zero { s6, sr6, s8, sr8, alpha6 } => {
            // eq:d3-zero: f_n = 1 / (1 + 6 t_n^(-α_n)), t_n = r / (s_r,n R0), α8 = α6 + 2.
            let r0 = data.r0ab(za, zb)?;
            let alpha8 = alpha6 + 2.0;
            let (p6, p8) = ((r / (sr6 * r0)).powf(-alpha6), (r / (sr8 * r0)).powf(-alpha8));
            let f6 = 1.0 / (1.0 + 6.0 * p6);
            let f8 = 1.0 / (1.0 + 6.0 * p8);
            let e = -(s6 * c6 / r6 * f6 + s8 * c8 / r8 * f8);
            // ∂e/∂r = Σ_n s_n C_n f_n / r^(n+1) [n - 6 α_n t_n^(-α_n) f_n].
            let de = s6 * c6 * f6 / (r6 * r) * (6.0 - 6.0 * alpha6 * p6 * f6)
                + s8 * c8 * f8 / (r8 * r) * (8.0 - 6.0 * alpha8 * p8 * f8);
            (e, de)
        }
    })
}

/// DFT-D3 dispersion energy in Hartree (theory.typ eq:d3-e), summed over all atom
/// pairs without a distance cutoff. Elements beyond Ar are an error.
pub fn dispersion_energy(mol: &Molecule, params: &D3Parameters) -> Result<f64> {
    let data = d3_data()?;
    let atoms = mol.atoms();
    let cn = data.coordination_numbers(atoms)?.cn;
    let mut energy = 0.0;
    for a in 0..atoms.len() {
        let za = atoms[a].z;
        for b in 0..a {
            let zb = atoms[b].z;
            let d: [f64; 3] = std::array::from_fn(|k| atoms[a].position[k] - atoms[b].position[k]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let c6 = data.c6(za, zb, cn[a], cn[b])?.c6;
            energy += pair_energy(&params.damping, data, za, zb, c6, r)?.0;
        }
    }
    Ok(energy)
}

/// DFT-D3 dispersion energy in Hartree and its gradient dE/dR_A in Hartree/Bohr
/// (one entry per atom), theory.typ eq:d3-grad. The energy is the same number as
/// [`dispersion_energy`] returns (same pairs, same order, same arithmetic).
pub fn dispersion_energy_and_gradient(mol: &Molecule, params: &D3Parameters) -> Result<(f64, Vec<[f64; 3]>)> {
    let data = d3_data()?;
    let atoms = mol.atoms();
    let n = atoms.len();
    let cn = data.coordination_numbers(atoms)?;
    let mut energy = 0.0;
    let mut gradient = vec![[0.0; 3]; n];
    // D_A = ∂E/∂CN_A = Σ_B (e_AB / C6^AB) ∂C6^AB/∂CN_A (gradient section of sec:d3).
    let mut de_dcn = vec![0.0; n];
    for a in 0..n {
        let za = atoms[a].z;
        for b in 0..a {
            let zb = atoms[b].z;
            let d: [f64; 3] = std::array::from_fn(|k| atoms[a].position[k] - atoms[b].position[k]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let c6 = data.c6(za, zb, cn.cn[a], cn.cn[b])?;
            let (e, de_dr) = pair_energy(&params.damping, data, za, zb, c6.c6, r)?;
            energy += e;
            // e_AB is linear in C6 (R0 does not depend on the CN): ∂e/∂C6 = e / C6.
            let de_dc6 = e / c6.c6;
            de_dcn[a] += de_dc6 * c6.dc6_dcn_a;
            de_dcn[b] += de_dc6 * c6.dc6_dcn_b;
            // Direct term of eq:d3-grad: ∂e/∂r (R_A - R_B) / r.
            for k in 0..3 {
                let v = de_dr * d[k] / r;
                gradient[a][k] += v;
                gradient[b][k] -= v;
            }
        }
    }
    // CN term of eq:d3-grad: Σ_A D_A dCN_A/dR_C. dCN_A/dR_C is built from the same
    // dg_AB/dr, so this equals Σ_B (D_A + D_B) dg_AB/dr (R_A - R_B) / r.
    for (a, &da) in de_dcn.iter().enumerate() {
        for (c, g) in gradient.iter_mut().enumerate() {
            let dcn = &cn.dcn_dr[a * n + c];
            for k in 0..3 {
                g[k] += da * dcn[k];
            }
        }
    }
    Ok((energy, gradient))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> &'static D3Data {
        d3_data().expect("embedded D3 data must parse")
    }

    fn atoms(z: &[u32], pos: &[[f64; 3]]) -> Vec<Atom> {
        z.iter().zip(pos).map(|(&z, &position)| Atom { z, position }).collect()
    }

    #[test]
    fn embedded_data_contents() {
        let d = data();
        assert_eq!(d.reference_cn(1).unwrap(), &[0.9118, 0.0]);
        assert_eq!(d.reference_cn(6).unwrap().len(), 5);
        assert_eq!(d.reference_cn(18).unwrap(), &[0.0]);
        assert_eq!(d.reference_c6(1, 1, 0, 0).unwrap(), 3.0267);
        assert_eq!(d.reference_c6(1, 6, 0, 2).unwrap(), 9.4203);
        assert_eq!(d.reference_c6(1, 6, 1, 0).unwrap(), 19.2653);
        // The other half of the pair table is the transpose.
        assert_eq!(d.reference_c6(6, 1, 2, 0).unwrap(), 9.4203);
        assert_eq!(d.reference_c6(6, 1, 0, 1).unwrap(), 19.2653);
        assert_eq!(d.r0ab(1, 6).unwrap(), 4.6283174);
        assert_eq!(d.r0ab(6, 1).unwrap(), 4.6283174);
        assert_eq!(d.scaled_covalent_radius(1).unwrap(), 0.8062831468620066);
        assert_eq!(d.r4r2(1).unwrap(), 2.007349);
    }

    #[test]
    fn parse_accepts_lf_and_crlf_alike() {
        let lf = D3_DATA.replace("\r\n", "\n");
        let crlf = lf.replace('\n', "\r\n");
        assert_eq!(D3Data::parse(&lf).unwrap(), D3Data::parse(&crlf).unwrap());
        assert_eq!(&D3Data::parse(&lf).unwrap(), data());
    }

    #[test]
    fn parse_rejects_bad_input() {
        let lf = D3_DATA.replace("\r\n", "\n");
        let without = |prefix: &str| -> String {
            lf.lines().filter(|l| !l.starts_with(prefix)).collect::<Vec<_>>().join("\n")
        };
        assert!(D3Data::parse(&without("element 7 ")).is_err());
        assert!(D3Data::parse(&without("r0ab 3 9 ")).is_err());
        assert!(D3Data::parse(&without("c6 6 8 2 3 ")).is_err());
        assert!(D3Data::parse(&format!("{lf}\nc6 1 1 1 1 3.0")).is_err()); // duplicate
        assert!(D3Data::parse(&format!("{lf}\nc6 2 1 1 1 3.0")).is_err()); // Za > Zb
        assert!(D3Data::parse(&format!("{lf}\nc6 1 1 3 1 3.0")).is_err()); // index out of range
        assert!(D3Data::parse(&format!("{lf}\nelement 19 K 1 1 1 0")).is_err());
        assert!(D3Data::parse(&lf.replace("element 1 H ", "element 1 He ")).is_err());
        assert!(D3Data::parse(&lf.replace("r0ab 1 1 4.1239495", "r0ab 1 1 nan")).is_err());
    }

    #[test]
    fn unsupported_elements_are_errors() {
        let d = data();
        assert!(matches!(d.c6(19, 1, 0.0, 0.0), Err(QcError::UnknownElement(_))));
        assert!(matches!(d.c6(1, 0, 0.0, 0.0), Err(QcError::UnknownElement(_))));
        let mol = atoms(&[1, 19], &[[0.0; 3], [0.0, 0.0, 2.0]]);
        assert!(matches!(d.coordination_numbers(&mol), Err(QcError::UnknownElement(_))));
        let same = atoms(&[1, 1], &[[0.0; 3], [0.0; 3]]);
        assert!(matches!(d.coordination_numbers(&same), Err(QcError::InvalidInput(_))));
    }

    #[test]
    fn coordination_number_of_h2_and_isolated_atom() {
        let d = data();
        let r = 1.4;
        let cn = d.coordination_numbers(&atoms(&[1, 1], &[[0.0; 3], [0.0, 0.0, r]])).unwrap();
        let rc = 2.0 * 0.8062831468620066;
        let g = 1.0 / (1.0 + (-16.0 * (rc / r - 1.0)).exp());
        assert!((cn.cn[0] - g).abs() < 1e-15 && (cn.cn[1] - g).abs() < 1e-15);
        // Near-equilibrium H2 is close to the H reference with CN = 0.9118.
        assert!((cn.cn[0] - 0.9118).abs() < 0.1, "CN(H2) = {}", cn.cn[0]);
        let single = d.coordination_numbers(&atoms(&[6], &[[1.0, 2.0, 3.0]])).unwrap();
        assert_eq!(single.cn, vec![0.0]);
        assert_eq!(single.dcn_dr, vec![[0.0; 3]]);
    }

    #[test]
    fn c6_single_reference_elements_equal_the_table() {
        let d = data();
        for (za, zb) in [(2, 2), (2, 10), (10, 18), (18, 18)] {
            let t = d.reference_c6(za, zb, 0, 0).unwrap();
            for (ca, cb) in [(0.0, 0.0), (0.7, 2.3), (12.0, 0.1)] {
                let v = d.c6(za, zb, ca, cb).unwrap();
                assert!((v.c6 - t).abs() <= 1e-14 * t, "{za}-{zb}: {} vs {t}", v.c6);
                assert_eq!((v.dc6_dcn_a, v.dc6_dcn_b), (0.0, 0.0));
            }
        }
    }

    #[test]
    fn c6_weights_by_hand() {
        // H (two references) with He (one): C6 = Σ_i W_i C6(H, He, i, 1).
        let d = data();
        let cn_h = 0.5;
        let refs = d.reference_cn(1).unwrap();
        let l: Vec<f64> = refs.iter().map(|&r| (-4.0 * (cn_h - r) * (cn_h - r)).exp()).collect();
        let expect = (l[0] * d.reference_c6(1, 2, 0, 0).unwrap() + l[1] * d.reference_c6(1, 2, 1, 0).unwrap())
            / (l[0] + l[1]);
        let v = d.c6(1, 2, cn_h, 1.7).unwrap();
        assert!((v.c6 - expect).abs() < 1e-13 * expect);
        // CN equal to the reference CN of an element with a single reference: the
        // table value itself.
        let a = d.c6(18, 18, 0.0, 0.0).unwrap().c6;
        assert_eq!(a, d.reference_c6(18, 18, 0, 0).unwrap());
    }

    #[test]
    fn c6_is_symmetric_in_the_pair() {
        let d = data();
        for (za, zb, ca, cb) in [(1, 6, 0.9, 3.1), (6, 8, 2.2, 1.1), (7, 14, 0.3, 3.9), (6, 6, 1.4, 3.6)] {
            let ab = d.c6(za, zb, ca, cb).unwrap();
            let ba = d.c6(zb, za, cb, ca).unwrap();
            assert!((ab.c6 - ba.c6).abs() <= 1e-13 * ab.c6);
            assert!((ab.dc6_dcn_a - ba.dc6_dcn_b).abs() <= 1e-12 * ab.c6);
            assert!((ab.dc6_dcn_b - ba.dc6_dcn_a).abs() <= 1e-12 * ab.c6);
        }
    }

    #[test]
    fn c6_far_from_all_references_does_not_underflow() {
        // At CN = 60 every unshifted weight exp(-4 d) underflows to 0. The shifted
        // form of eq:d3-c6 gives the limit: only the C reference with the largest CN
        // (index 4) contributes.
        let d = data();
        let cn_h = 0.3;
        let v = d.c6(6, 1, 60.0, cn_h).unwrap();
        let refs = d.reference_cn(1).unwrap();
        let l: Vec<f64> = refs.iter().map(|&r| (-4.0 * (cn_h - r) * (cn_h - r)).exp()).collect();
        let expect = (l[0] * d.reference_c6(6, 1, 4, 0).unwrap() + l[1] * d.reference_c6(6, 1, 4, 1).unwrap())
            / (l[0] + l[1]);
        assert!(v.c6.is_finite() && v.dc6_dcn_a.is_finite() && v.dc6_dcn_b.is_finite());
        assert!((v.c6 - expect).abs() < 1e-12 * expect, "{} vs {expect}", v.c6);
        assert!(v.dc6_dcn_a.abs() < 1e-12);
    }

    #[test]
    fn c6_derivatives_match_central_differences() {
        let d = data();
        let h = 1e-5;
        for (za, zb, ca, cb) in
            [(1, 1, 0.4, 0.95), (1, 6, 0.9, 3.1), (6, 8, 2.2, 1.1), (7, 14, 0.3, 3.9), (5, 13, 4.2, 2.5), (8, 16, 1.5, 0.6)]
        {
            let v = d.c6(za, zb, ca, cb).unwrap();
            let num_a = (d.c6(za, zb, ca + h, cb).unwrap().c6 - d.c6(za, zb, ca - h, cb).unwrap().c6) / (2.0 * h);
            let num_b = (d.c6(za, zb, ca, cb + h).unwrap().c6 - d.c6(za, zb, ca, cb - h).unwrap().c6) / (2.0 * h);
            let tol = 1e-8 * v.c6.max(1.0);
            assert!((num_a - v.dc6_dcn_a).abs() < tol, "{za}-{zb} dA: {num_a} vs {}", v.dc6_dcn_a);
            assert!((num_b - v.dc6_dcn_b).abs() < tol, "{za}-{zb} dB: {num_b} vs {}", v.dc6_dcn_b);
        }
    }

    #[test]
    fn dimer_energy_by_hand() {
        // He2 and Ar2: one reference each, so C6 is the table value at any CN.
        for (z, r) in [(2u32, 5.7), (18, 7.1)] {
            let mol = Molecule::new(atoms(&[z, z], &[[0.0; 3], [0.0, 0.0, r]]), 0).unwrap();
            let d = data();
            let c6 = d.reference_c6(z, z, 0, 0).unwrap();
            let q = d.r4r2(z).unwrap();
            let c8 = 3.0 * c6 * q * q; // eq:d3-c8
            let r0 = d.r0ab(z, z).unwrap();

            let (s6, a1, s8, a2) = (1.0, 0.3981, 1.9889, 4.4211);
            let rd = a1 * (c8 / c6).sqrt() + a2; // eq:d3-bj
            let e_bj = -(s6 * c6 / (r.powi(6) + rd.powi(6)) + s8 * c8 / (r.powi(8) + rd.powi(8)));
            let bj = D3Parameters::new(D3Damping::Bj { s6, a1, s8, a2 }).unwrap();
            let e = dispersion_energy(&mol, &bj).unwrap();
            assert!((e - e_bj).abs() <= 1e-14 * e_bj.abs(), "Z={z} BJ: {e} vs {e_bj}");

            let (s6, sr6, s8) = (1.0, 1.261, 1.703);
            let f6 = 1.0 / (1.0 + 6.0 * (r / (sr6 * r0)).powf(-14.0)); // eq:d3-zero
            let f8 = 1.0 / (1.0 + 6.0 * (r / r0).powf(-16.0));
            let e_zero = -(s6 * c6 / r.powi(6) * f6 + s8 * c8 / r.powi(8) * f8);
            let zero = D3Parameters::for_functional("b3lyp", D3Variant::Zero).unwrap();
            let e = dispersion_energy(&mol, &zero).unwrap();
            assert!((e - e_zero).abs() <= 1e-14 * e_zero.abs(), "Z={z} zero: {e} vs {e_zero}");
        }
    }

    #[test]
    fn energy_of_single_atom_is_zero_and_translation_invariant() {
        let p = D3Parameters::for_functional("cam-b3lyp", D3Variant::Bj).unwrap();
        let one = Molecule::new(atoms(&[6], &[[0.3, 0.2, 0.1]]), 0).unwrap();
        assert_eq!(dispersion_energy(&one, &p).unwrap(), 0.0);
        let pos = [[0.0, 0.0, 0.0], [1.8, 0.1, 0.0], [-0.5, 1.7, 0.2], [0.3, -0.4, 2.9]];
        let shifted: Vec<[f64; 3]> = pos.iter().map(|p| [p[0] + 3.0, p[1] - 1.0, p[2] + 0.5]).collect();
        let a = dispersion_energy(&Molecule::new(atoms(&[8, 1, 1, 6], &pos), 0).unwrap(), &p).unwrap();
        let b = dispersion_energy(&Molecule::new(atoms(&[8, 1, 1, 6], &shifted), 0).unwrap(), &p).unwrap();
        assert!(a < 0.0 && (a - b).abs() <= 1e-15 * a.abs(), "{a} vs {b}");
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let bad = [
            D3Damping::Bj { s6: 1.0, a1: 0.4, s8: f64::NAN, a2: 4.4 },
            D3Damping::Bj { s6: 1.0, a1: 0.0, s8: 1.9, a2: 0.0 },
            D3Damping::Bj { s6: 1.0, a1: -0.1, s8: 1.9, a2: 4.4 },
            D3Damping::Bj { s6: 1.0, a1: 0.4, s8: 1.9, a2: -1.0 },
            D3Damping::Zero { s6: 1.0, sr6: 0.0, s8: 1.7, sr8: 1.0, alpha6: 14.0 },
            D3Damping::Zero { s6: 1.0, sr6: 1.2, s8: 1.7, sr8: 1.0, alpha6: f64::INFINITY },
        ];
        for d in bad {
            assert!(matches!(D3Parameters::new(d), Err(QcError::InvalidInput(_))), "{d:?}");
        }
        assert!(D3Parameters::new(D3Damping::Bj { s6: 1.0, a1: 0.4, s8: 0.0, a2: 4.4 }).is_ok());
        assert!(D3Parameters::new(D3Damping::Bj { s6: 1.0, a1: 0.0, s8: 1.9, a2: 4.4 }).is_ok());
    }

    #[test]
    fn coordination_number_derivatives_match_central_differences() {
        let d = data();
        let z = [8, 1, 1, 6, 1, 17];
        let pos = [
            [0.0, 0.0, 0.0],
            [1.8, 0.1, 0.0],
            [-0.5, 1.7, 0.2],
            [0.3, -0.4, 2.9],
            [1.2, -1.9, 3.6],
            [-2.6, -0.8, 3.3],
        ];
        let mol = atoms(&z, &pos);
        let n = mol.len();
        let cn = d.coordination_numbers(&mol).unwrap();
        let h = 1e-5;
        for c in 0..n {
            for k in 0..3 {
                let shifted = |s: f64| {
                    let mut m = mol.clone();
                    m[c].position[k] += s;
                    d.coordination_numbers(&m).unwrap().cn
                };
                let (p, m) = (shifted(h), shifted(-h));
                for a in 0..n {
                    let num = (p[a] - m[a]) / (2.0 * h);
                    let ana = cn.dcn_dr[a * n + c][k];
                    assert!((num - ana).abs() < 1e-8, "dCN_{a}/dR_{c},{k}: {num} vs {ana}");
                }
            }
        }
        // Translational invariance: Σ_c dCN_a/dR_c = 0.
        for a in 0..n {
            for k in 0..3 {
                let s: f64 = (0..n).map(|c| cn.dcn_dr[a * n + c][k]).sum();
                assert!(s.abs() < 1e-14);
            }
        }
    }
}

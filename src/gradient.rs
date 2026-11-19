//! Analytic nuclear gradients dE/dR of closed-shell RHF and RKS energies
//! (theory.typ section 10, eq:grad). UNCHECKED(Pulay1969)
//!
//! The gradient is assembled term by term as in theory.typ eq:grad:
//!
//! * one-electron: Σ P h^(X) — kinetic energy (eq:dS form) and nuclear attraction, UNCHECKED(Pulay1969)
//!   including the operator ("Hellmann–Feynman") part via translational invariance
//!   (eq:hf-force); UNCHECKED(Pulay1969)
//! * two-electron: Σ Γ (μν|λσ)^(X) with Γ of eq:gamma; for a range-separated exchange UNCHECKED(Pulay1969)
//!   kernel also the attenuated integrals of eq:cam-grad; DERIVED (from eq:cam-fock; Yanai2004 has no gradients)
//! * overlap: -Σ W S^(X) with the energy-weighted density W of eq:W; UNCHECKED(Pulay1969)
//! * exchange–correlation: eq:xc-grad (grid-weight derivatives neglected); UNCHECKED(Johnson1993)
//! * nuclear repulsion: eq:vnn-grad. UNCHECKED(Szabo1996)
//!
//! All derivative integrals use the centre-derivative rule eq:center-deriv on UNCHECKED(Pulay1969)
//! primitives with angular momentum ±1. Instead of transforming derivative integrals
//! to the spherical d basis, the density-like matrices are transformed to the
//! normalized Cartesian basis once (Σ_μν M_μν I_μν = Σ_cd (Tᵀ M T)_cd I^cart_cd with T
//! the eq:sph-d matrix), which is equivalent because eq:sph-d commutes with the UNCHECKED(Schlegel1995)
//! derivative (likewise eq:sph-f for spherical f shells). UNCHECKED(Schlegel1995)
//!
//! Derivative ERIs are never stored: each unique shell quartet (8-fold symmetry,
//! eq:eri-sym) is contracted with the symmetrized Γ on the fly, using the UNCHECKED(Szabo1996)
//! McMurchie–Davidson form of eq:eri. Per primitive quartet only the derivatives with VERIFIED(McMurchie1978, derived from Eqs. (3.30)-(3.34), p. 14 (preprint), with (2.27))
//! respect to the centres A, B, C are evaluated; the D derivative follows from
//! translational invariance (the four centre derivatives sum to zero). Quartets are
//! skipped when the density-weighted Schwarz bound Q_ab Q_cd max|Γ| is below
//! [`GRADIENT_SCREENING_THRESHOLD`].
//!
//! **SCF convergence.** The gradient formula assumes a stationary SCF solution; its
//! error is first order in the density error. Feed it an SCF converged on the DIIS
//! commutator to ~1e-9–1e-10 (see `calculation::GRADIENT_SCF_ERROR_TOLERANCE`), not the
//! default 1e-7.

use crate::ao_eval::{D_X, eval_ao_shells, hessian_component};
use crate::basis::{Basis, cartesian_components, cartesian_norm};
use crate::boys::boys;
use crate::dft::{XcIntegrator, density_on_points, xc_on_points};
use crate::error::{QcError, Result};
use crate::hermite::{HermiteR, PrimitivePair, erf_attenuation};
use crate::integrals::{ExchangeKernel, cartesian_schwarz_factors, pair_index};
use crate::linalg::Matrix;
use crate::molecule::Molecule;
use crate::scf::ScfResult;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Shell quartets with Q_ab Q_cd max|Γ_quartet| below this are skipped in the
/// two-electron gradient. Derivative integrals exceed the plain ones by at most a
/// factor ~2√a_max (~150 for the tightest exponents here), so the neglected terms
/// stay far below 1e-10 Hartree/Bohr.
pub const GRADIENT_SCREENING_THRESHOLD: f64 = 1e-14;

/// Contributions of eq:grad, each an `N_atom` list of (∂/∂x, ∂/∂y, ∂/∂z) in Hartree/Bohr.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientComponents {
    /// Σ P h^(X): kinetic energy and nuclear attraction (basis-centre and operator parts).
    pub one_electron: Vec<[f64; 3]>,
    /// Σ Γ (μν|λσ)^(X): Coulomb and exact exchange (a_HF-scaled, or with the kernel
    /// [α + β erf(ωr)]/r of eq:cam-grad).
    pub two_electron: Vec<[f64; 3]>,
    /// -Σ W S^(X).
    pub overlap: Vec<[f64; 3]>,
    /// E_xc^(X) (zero for HF).
    pub xc: Vec<[f64; 3]>,
    /// ∂V_nn/∂R.
    pub nuclear_repulsion: Vec<[f64; 3]>,
}

impl GradientComponents {
    /// The total gradient dE/dR (sum of all contributions).
    pub fn total(&self) -> Vec<[f64; 3]> {
        let mut t = self.one_electron.clone();
        for part in [&self.two_electron, &self.overlap, &self.xc, &self.nuclear_repulsion] {
            add_into(&mut t, part);
        }
        t
    }
}

/// Per-atom gradient contributions (one [x, y, z] per atom).
type AtomVectors = Vec<[f64; 3]>;

fn add_into(acc: &mut [[f64; 3]], x: &[[f64; 3]]) {
    for (a, b) in acc.iter_mut().zip(x) {
        for k in 0..3 {
            a[k] += b[k];
        }
    }
}

/// ∂V_nn/∂R_A = -Σ_{B≠A} Z_A Z_B (R_A - R_B)/R_AB³. theory.typ eq:vnn-grad
pub fn nuclear_repulsion_gradient(mol: &Molecule) -> Vec<[f64; 3]> {
    let atoms = mol.atoms();
    let mut g = vec![[0.0; 3]; atoms.len()];
    for (a, ga) in g.iter_mut().enumerate() {
        for (b, atb) in atoms.iter().enumerate() {
            if a == b {
                continue;
            }
            let d = [0, 1, 2].map(|k| atoms[a].position[k] - atb.position[k]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let f = (atoms[a].z * atb.z) as f64 / (r * r * r);
            for k in 0..3 {
                ga[k] -= f * d[k];
            }
        }
    }
    g
}

/// W_μν = 2 Σ_i^occ ε_i C_μi C_νi. theory.typ eq:W
pub fn energy_weighted_density(scf: &ScfResult) -> Matrix {
    let c = &scf.coefficients;
    let n = c.rows();
    Matrix::from_fn(n, n, |mu, nu| {
        2.0 * (0..scf.n_occupied).map(|i| scf.orbital_energies[i] * c[(mu, i)] * c[(nu, i)]).sum::<f64>()
    })
}

/// Offsets of every shell in the normalized-Cartesian AO list and its length.
fn cartesian_offsets(basis: &Basis) -> (Vec<usize>, usize) {
    let mut off = Vec::with_capacity(basis.shells().len());
    let mut n = 0;
    for s in basis.shells() {
        off.push(n);
        n += s.n_cartesian();
    }
    (off, n)
}

/// Tᵀ M T: a matrix over the basis AOs expressed over normalized Cartesian components
/// (identity for Cartesian shells, eq:sph-d / eq:sph-f for spherical d / f shells).
fn to_cartesian(basis: &Basis, m: &Matrix) -> Matrix {
    let (coff, nc) = cartesian_offsets(basis);
    let n = basis.nao();
    let mut t = Matrix::zeros(n, nc);
    for (s, &co) in basis.shells().iter().zip(&coff) {
        match s.spherical_transform() {
            None => {
                for k in 0..s.n_cartesian() {
                    t[(s.ao_offset + k, co + k)] = 1.0;
                }
            }
            Some(tr) => {
                for (mrow, row) in tr.iter().enumerate() {
                    for (c, &w) in row.iter().enumerate() {
                        t[(s.ao_offset + mrow, co + c)] = w;
                    }
                }
            }
        }
    }
    t.transpose().matmul(m).matmul(&t)
}

fn check_square(m: &Matrix, n: usize, what: &str) -> Result<()> {
    if m.rows() != n || m.cols() != n {
        return Err(QcError::InvalidInput(format!("{what} is {}x{}, basis has {n} functions", m.rows(), m.cols())));
    }
    Ok(())
}

fn shift(l: [usize; 3], k: usize, up: bool) -> [usize; 3] {
    let mut r = l;
    if up {
        r[k] += 1;
    } else {
        r[k] -= 1;
    }
    r
}

// ---------------------------------------------------------------------------
// One-electron terms
// ---------------------------------------------------------------------------

/// Which one-electron derivative terms to evaluate.
struct OneElectronRequest<'a> {
    /// Cartesian density for Σ P h^(X).
    p: Option<&'a Matrix>,
    /// Cartesian energy-weighted density for -Σ W S^(X).
    w: Option<&'a Matrix>,
}

/// Returns (Σ P h^(X), -Σ W S^(X)) for Cartesian-basis matrices.
///
/// Looping over *ordered* shell pairs and differentiating only the bra function uses
/// the symmetry of P, W and the integrals: Σ_μν M_μν I^(X_A)_μν = 2 Σ_{μ∈A} Σ_ν M_μν
/// ⟨∂χ_μ/∂A|..|χ_ν⟩ (theory.typ eq:dS). For the nuclear attraction the operator part
/// is, by eq:hf-force, +2 Z_C Σ_μν P_μν d_μ on atom C.
fn one_electron_terms(mol: &Molecule, basis: &Basis, req: OneElectronRequest) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let shells = basis.shells();
    let (coff, _) = cartesian_offsets(basis);
    let natom = mol.atoms().len();
    let charges: Vec<(f64, [f64; 3])> = mol.atoms().iter().map(|a| (a.z as f64, a.position)).collect();
    let parts: Vec<(AtomVectors, AtomVectors)> = (0..shells.len())
        .into_par_iter()
        .map(|ia| {
            let mut gh = vec![[0.0; 3]; natom];
            let mut gs = vec![[0.0; 3]; natom];
            let sa = &shells[ia];
            let ca = cartesian_components(sa.l);
            let na: Vec<f64> = ca.iter().map(|&c| cartesian_norm(c)).collect();
            let mut r = HermiteR::new();
            for (ib, sb) in shells.iter().enumerate() {
                let cb = cartesian_components(sb.l);
                let nb: Vec<f64> = cb.iter().map(|&c| cartesian_norm(c)).collect();
                let (oa, ob) = (coff[ia], coff[ib]);
                for (&ea, &da) in sa.exponents.iter().zip(&sa.coefficients) {
                    for (&eb, &db) in sb.exponents.iter().zip(&sb.coefficients) {
                        let pair = PrimitivePair::new(ea, sa.center, sa.l + 1, eb, sb.center, sb.l);
                        // Overlap and kinetic: ∂/∂A of the bra (theory.typ eq:center-deriv, eq:dS).
                        for (x, &la) in ca.iter().enumerate() {
                            for (y, &lb) in cb.iter().enumerate() {
                                let c = da * db * na[x] * nb[y];
                                let pv = req.p.map_or(0.0, |p| p[(oa + x, ob + y)]);
                                let wv = req.w.map_or(0.0, |w| w[(oa + x, ob + y)]);
                                for k in 0..3 {
                                    let up = shift(la, k, true);
                                    let mut ds = 2.0 * ea * pair.overlap(up, lb);
                                    let mut dt = 2.0 * ea * pair.kinetic(up, lb);
                                    if la[k] > 0 {
                                        let dn = shift(la, k, false);
                                        ds -= la[k] as f64 * pair.overlap(dn, lb);
                                        dt -= la[k] as f64 * pair.kinetic(dn, lb);
                                    }
                                    gh[sa.atom][k] += 2.0 * c * pv * dt;
                                    gs[sa.atom][k] -= 2.0 * c * wv * ds;
                                }
                            }
                        }
                        // Nuclear attraction, theory.typ eq:nuc, eq:hf-force. UNCHECKED(Szabo1996)
                        if req.p.is_some() {
                            let lmax = sa.l + 1 + sb.l;
                            for (ic, &(zc, rc)) in charges.iter().enumerate() {
                                let pc = [0, 1, 2].map(|k| pair.center[k] - rc[k]);
                                r.compute(lmax, pair.p, pc);
                                for (x, &la) in ca.iter().enumerate() {
                                    for (y, &lb) in cb.iter().enumerate() {
                                        let pv = req.p.map_or(0.0, |p| p[(oa + x, ob + y)]);
                                        if pv == 0.0 {
                                            continue;
                                        }
                                        let c = da * db * na[x] * nb[y] * pv * 2.0 * zc;
                                        for k in 0..3 {
                                            let mut dv = 2.0 * ea * pair.nuclear(shift(la, k, true), lb, &r);
                                            if la[k] > 0 {
                                                dv -= la[k] as f64 * pair.nuclear(shift(la, k, false), lb, &r);
                                            }
                                            // V = -Z_C I^C: basis-centre part on the bra atom,
                                            // operator part (eq:hf-force) on atom C.
                                            gh[sa.atom][k] -= c * dv;
                                            gh[ic][k] += c * dv;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (gh, gs)
        })
        .collect();
    let mut gh = vec![[0.0; 3]; natom];
    let mut gs = vec![[0.0; 3]; natom];
    for (h, s) in parts {
        add_into(&mut gh, &h);
        add_into(&mut gs, &s);
    }
    (gh, gs)
}

/// Σ_μν P_μν h_μν^(X) (kinetic + nuclear attraction) for a symmetric AO density `p`.
/// theory.typ eq:grad (first term), eq:dS, eq:hf-force
pub fn one_electron_gradient(mol: &Molecule, basis: &Basis, p: &Matrix) -> Result<Vec<[f64; 3]>> {
    check_square(p, basis.nao(), "density matrix")?;
    let pc = to_cartesian(basis, p);
    Ok(one_electron_terms(mol, basis, OneElectronRequest { p: Some(&pc), w: None }).0)
}

/// -Σ_μν W_μν S_μν^(X) for a symmetric AO matrix `w`. theory.typ eq:grad, eq:dS
pub fn overlap_gradient(mol: &Molecule, basis: &Basis, w: &Matrix) -> Result<Vec<[f64; 3]>> {
    check_square(w, basis.nao(), "energy-weighted density")?;
    let wc = to_cartesian(basis, w);
    Ok(one_electron_terms(mol, basis, OneElectronRequest { p: None, w: Some(&wc) }).1)
}

// ---------------------------------------------------------------------------
// Two-electron term
// ---------------------------------------------------------------------------

/// Hermite indices (t, u, v) with t+u+v <= l, ordered by total degree so that the
/// first `n_hermite(l')` entries are those of degree <= l'.
fn hermite_list(l: usize) -> Vec<[usize; 3]> {
    let mut v = Vec::new();
    for n in 0..=l {
        for t in (0..=n).rev() {
            for u in (0..=n - t).rev() {
                v.push([t, u, n - t - u]);
            }
        }
    }
    v
}

/// Number of (t, u, v) with t+u+v <= l.
const fn n_hermite(l: usize) -> usize {
    (l + 1) * (l + 2) * (l + 3) / 6
}

/// Number of E-coefficient sets per primitive pair: value, ∂/∂A_{x,y,z}, ∂/∂B_{x,y,z}.
const NSETS: usize = 7;

/// One primitive pair of a shell pair, with Hermite coefficients of the product
/// distribution and of its derivatives with respect to both centres.
struct GradPrim {
    p: f64,
    center: [f64; 3],
    /// `e[(set * n1 + i) * ncomp + comp]`, n1 = n_hermite(lab + 1); contraction and
    /// normalization factors d̃_a d̃_b c_a c_b included.
    e: Vec<f64>,
    /// For a pair with lab <= `GRAD_SMALL_LAB` (empty otherwise), the entries of `e`
    /// that [`quartet_gradient_small`] reads when the pair is the ket, transposed so
    /// that one component's coefficients are contiguous:
    /// `et[comp * w + j] = e[j * ncomp + comp]` (value set, j < n_hermite(lab)) and
    /// `et[comp * w + nj + k * n1 + j] = e[((1 + k) * n1 + j) * ncomp + comp]` (∂/∂A_k,
    /// j < n1), with nj = n_hermite(lab) and w = nj + 3 n1.
    et: Vec<f64>,
}

/// A shell pair (a >= b) prepared for the derivative-ERI contraction.
struct GradPair {
    sa: usize,
    sb: usize,
    lab: usize,
    ncomp: usize,
    tuv: Vec<[usize; 3]>,
    prims: Vec<GradPrim>,
}

impl GradPair {
    fn new(basis: &Basis, sa: usize, sb: usize) -> Self {
        let shells = basis.shells();
        let (a, b) = (&shells[sa], &shells[sb]);
        let ca = cartesian_components(a.l);
        let cb = cartesian_components(b.l);
        let na: Vec<f64> = ca.iter().map(|&c| cartesian_norm(c)).collect();
        let nb: Vec<f64> = cb.iter().map(|&c| cartesian_norm(c)).collect();
        let lab = a.l + b.l;
        let tuv = hermite_list(lab + 1);
        let n1 = tuv.len();
        let ncomp = ca.len() * cb.len();
        let mut prims = Vec::new();
        for (&ea, &da) in a.exponents.iter().zip(&a.coefficients) {
            for (&eb, &db) in b.exponents.iter().zip(&b.coefficients) {
                let pair = PrimitivePair::new(ea, a.center, a.l + 1, eb, b.center, b.l + 1);
                let mut e = vec![0.0; NSETS * n1 * ncomp];
                let mut emax: f64 = 0.0;
                for (i, h) in tuv.iter().enumerate() {
                    for (x, &la) in ca.iter().enumerate() {
                        for (y, &lb) in cb.iter().enumerate() {
                            let c = da * db * na[x] * nb[y];
                            let comp = x * cb.len() + y;
                            let f = [0, 1, 2].map(|k| pair.e(k, la[k], lb[k], h[k]));
                            // theory.typ eq:center-deriv applied to the Hermite expansion:
                            // ∂/∂A_k → 2a E^{i+1,j} - i E^{i-1,j}; ∂/∂B_k → 2b E^{i,j+1} - j E^{i,j-1}.
                            let fa = [0, 1, 2].map(|k| {
                                let mut v = 2.0 * ea * pair.e(k, la[k] + 1, lb[k], h[k]);
                                if la[k] > 0 {
                                    v -= la[k] as f64 * pair.e(k, la[k] - 1, lb[k], h[k]);
                                }
                                v
                            });
                            let fb = [0, 1, 2].map(|k| {
                                let mut v = 2.0 * eb * pair.e(k, la[k], lb[k] + 1, h[k]);
                                if lb[k] > 0 {
                                    v -= lb[k] as f64 * pair.e(k, la[k], lb[k] - 1, h[k]);
                                }
                                v
                            });
                            let mut vals = [0.0; NSETS];
                            vals[0] = f[0] * f[1] * f[2];
                            vals[1] = fa[0] * f[1] * f[2];
                            vals[2] = f[0] * fa[1] * f[2];
                            vals[3] = f[0] * f[1] * fa[2];
                            vals[4] = fb[0] * f[1] * f[2];
                            vals[5] = f[0] * fb[1] * f[2];
                            vals[6] = f[0] * f[1] * fb[2];
                            for (s, v) in vals.iter().enumerate() {
                                let val = c * v;
                                e[(s * n1 + i) * ncomp + comp] = val;
                                emax = emax.max(val.abs());
                            }
                        }
                    }
                }
                if emax > 1e-30 {
                    let mut et = Vec::new();
                    if lab <= GRAD_SMALL_LAB {
                        let nj = n_hermite(lab);
                        et.reserve(ncomp * (nj + 3 * n1));
                        for comp in 0..ncomp {
                            et.extend((0..nj).map(|j| e[j * ncomp + comp]));
                            for k in 0..3 {
                                et.extend((0..n1).map(|j| e[((1 + k) * n1 + j) * ncomp + comp]));
                            }
                        }
                    }
                    prims.push(GradPrim { p: pair.p, center: pair.center, e, et });
                }
            }
        }
        GradPair { sa, sb, lab, ncomp, tuv, prims }
    }
}

/// Per-thread scratch buffers of the quartet kernels.
#[derive(Default)]
struct QuartetScratch {
    r: HermiteR,
    ridx: Vec<usize>,
    rmat: Vec<f64>,
    zv: Vec<f64>,
    zc: Vec<f64>,
    hv: Vec<f64>,
    q: Vec<f64>,
    gamma: Vec<f64>,
    /// Buffers of [`quartet_gradient_small`].
    small: GradSmallBuffers,
}

/// Largest l_a + l_b of a shell pair served by [`quartet_gradient_small`] (s and p
/// shells, and s with d), so that its R tables have ltot = l_ab + l_cd + 1 <= 5.
const GRAD_SMALL_LAB: usize = 2;
/// Largest ltot of [`hermite_r_grad_small`].
const GRAD_SMALL_LTOT: usize = 2 * GRAD_SMALL_LAB + 1;
/// Entries of the largest compact R table of [`hermite_r_grad_small`]: n_hermite(ltot).
const GRAD_SMALL_NR: usize = n_hermite(GRAD_SMALL_LTOT);
/// Most Cartesian components of a pair with l_ab <= GRAD_SMALL_LAB (p × p).
const GRAD_SMALL_NCOMP: usize = 9;
/// n_hermite(GRAD_SMALL_LAB) and n_hermite(GRAD_SMALL_LAB + 1).
const GRAD_SMALL_NH: usize = n_hermite(GRAD_SMALL_LAB);
const GRAD_SMALL_NH1: usize = n_hermite(GRAD_SMALL_LAB + 1);

/// Fixed-size buffers of [`quartet_gradient_small`]: the R layers of
/// [`hermite_r_grad_small`] and the Z_v, Z_C, H, Q arrays of [`quartet_gradient_general`]
/// at their largest size (used with the same runtime layout).
struct GradSmallBuffers {
    /// Two compact R layers (layer n in `r[n % 2]`); entry `GRAD_SMALL_NR` is never
    /// written and stays 0.
    r: [[f64; GRAD_SMALL_NR + 1]; 2],
    f: [f64; GRAD_SMALL_LTOT + 1],
    zv: [f64; GRAD_SMALL_NCOMP * GRAD_SMALL_NH],
    zc: [f64; 3 * GRAD_SMALL_NCOMP * GRAD_SMALL_NH1],
    hv: [f64; GRAD_SMALL_NH1 * GRAD_SMALL_NCOMP],
    q: [f64; GRAD_SMALL_NCOMP * GRAD_SMALL_NH1],
}

impl Default for GradSmallBuffers {
    fn default() -> Self {
        GradSmallBuffers {
            r: [[0.0; GRAD_SMALL_NR + 1]; 2],
            f: [0.0; GRAD_SMALL_LTOT + 1],
            zv: [0.0; GRAD_SMALL_NCOMP * GRAD_SMALL_NH],
            zc: [0.0; 3 * GRAD_SMALL_NCOMP * GRAD_SMALL_NH1],
            hv: [0.0; GRAD_SMALL_NH1 * GRAD_SMALL_NCOMP],
            q: [0.0; GRAD_SMALL_NCOMP * GRAD_SMALL_NH1],
        }
    }
}

/// One step of theory.typ eq:R-rec for the compact Hermite index e (position in
/// [`hermite_list`]) of degree >= 1, lowering the first nonzero of (t, u, v), say t:
/// R^n_e = r[axis] R^{n+1}_{prev} + coef R^{n+1}_{prev2} with `prev` = (t−1, u, v),
/// `prev2` = (t−2, u, v) and `coef` = t − 1 (0, and `prev2` unused, when t = 1).
#[derive(Clone, Copy)]
struct RStep {
    axis: usize,
    prev: usize,
    prev2: usize,
    coef: f64,
}

/// Position of (t, u, v) in `list`.
const fn hermite_index(list: &[[usize; 3]], t: usize, u: usize, v: usize) -> usize {
    let mut e = 0;
    while e < list.len() {
        if list[e][0] == t && list[e][1] == u && list[e][2] == v {
            return e;
        }
        e += 1;
    }
    panic!("Hermite index not in the list")
}

/// The [`RStep`] of every compact Hermite index of degree <= `GRAD_SMALL_LTOT`
/// (entry 0, the start of each layer, is unused). Because [`hermite_list`] is ordered
/// by degree, the steps of a table with ltot < `GRAD_SMALL_LTOT` are a prefix.
const fn grad_r_steps() -> [RStep; GRAD_SMALL_NR] {
    let list = hermite_array::<GRAD_SMALL_NR>(GRAD_SMALL_LTOT);
    let mut out = [RStep { axis: 0, prev: 0, prev2: 0, coef: 0.0 }; GRAD_SMALL_NR];
    let mut e = 1;
    while e < GRAD_SMALL_NR {
        let [t, u, v] = list[e];
        // The first nonzero of (t, u, v) is lowered, as in HermiteR::compute.
        let (axis, k) = if t > 0 {
            (0, t)
        } else if u > 0 {
            (1, u)
        } else {
            (2, v)
        };
        let mut h = [t, u, v];
        h[axis] -= 1;
        out[e].axis = axis;
        out[e].prev = hermite_index(&list, h[0], h[1], h[2]);
        if k > 1 {
            h[axis] -= 1;
            out[e].prev2 = hermite_index(&list, h[0], h[1], h[2]);
            out[e].coef = (k - 1) as f64;
        }
        e += 1;
    }
    out
}

/// [`HermiteR::compute`] for `lmax <= GRAD_SMALL_LTOT` into the fixed buffers of `s`,
/// stored compactly: entry e is R_{t,u,v} for (t, u, v) = [`hermite_list`]`(lmax)[e]`,
/// and entry `GRAD_SMALL_NR` is 0. Each layer n evaluates only its n_hermite(lmax − n)
/// entries with the compile-time steps of [`grad_r_steps`]. The Boys call, the
/// (−2p)^n F_n via `powi`, and the operations of the recurrence (theory.typ eq:R-rec)
/// and their order are those of [`HermiteR::compute`], so the entries are identical
/// bit for bit. Inlined into [`quartet_gradient_small`], where `lmax` is a constant.
#[inline(always)]
fn hermite_r_grad_small(lmax: usize, p: f64, r: [f64; 3], s: &mut GradSmallBuffers) -> &[f64; GRAD_SMALL_NR + 1] {
    debug_assert!(lmax <= GRAD_SMALL_LTOT);
    let steps: [RStep; GRAD_SMALL_NR] = const { grad_r_steps() };
    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let f = &mut s.f;
    boys(lmax, p * r2, f);
    let [even, odd] = &mut s.r;
    let m2p = -2.0 * p;
    for n in (0..=lmax).rev() {
        // Layer n is built from layer n+1; layer 0 ends in `r[0]`.
        let (cur, prev) = if n % 2 == 0 { (&mut *even, &*odd) } else { (&mut *odd, &*even) };
        // `n` is hidden from the optimizer: once this loop is unrolled, a constant
        // exponent lets LLVM expand powi into a different product, which changes the
        // bits. As an opaque value it goes to the same runtime powi as HermiteR::compute.
        cur[0] = m2p.powi(std::hint::black_box(n as i32)) * f[n];
        for (e, step) in steps.iter().enumerate().take(n_hermite(lmax - n)).skip(1) {
            let mut val = r[step.axis] * prev[step.prev];
            if step.coef != 0.0 {
                val += step.coef * prev[step.prev2];
            }
            cur[e] = val;
        }
    }
    &s.r[0]
}

/// [`hermite_list`]`(l)` as a fixed array (`N` = n_hermite(l)), for compile-time tables.
const fn hermite_array<const N: usize>(l: usize) -> [[usize; 3]; N] {
    assert!(N == n_hermite(l));
    let mut out = [[0; 3]; N];
    let mut k = 0;
    let mut n = 0;
    while n <= l {
        let mut t = n + 1;
        while t > 0 {
            t -= 1;
            let mut u = n - t + 1;
            while u > 0 {
                u -= 1;
                out[k] = [t, u, n - t - u];
                k += 1;
            }
        }
        n += 1;
    }
    out
}

/// The table `QuartetScratch::ridx` of [`quartet_gradient_general`] for l_ab = `lab`,
/// l_cd = `lcd` (bra-major, `[i][j]`), as indices into the compact table of
/// [`hermite_r_grad_small`], with the index `GRAD_SMALL_NR` (an R entry that is 0)
/// where the general kernel stores 0.
const fn grad_ridx_table<const NI1: usize, const NJ1: usize>(lab: usize, lcd: usize) -> [[usize; NJ1]; NI1] {
    let (hs, ks) = (hermite_array::<NI1>(lab + 1), hermite_array::<NJ1>(lcd + 1));
    let list = hermite_array::<GRAD_SMALL_NR>(GRAD_SMALL_LTOT);
    let ltot = lab + lcd + 1;
    assert!(ltot <= GRAD_SMALL_LTOT);
    let mut out = [[0; NJ1]; NI1];
    let mut i = 0;
    while i < NI1 {
        let mut j = 0;
        while j < NJ1 {
            let (h, k) = (hs[i], ks[j]);
            let deg = h[0] + h[1] + h[2] + k[0] + k[1] + k[2];
            out[i][j] = if deg <= ltot {
                hermite_index(&list, h[0] + k[0], h[1] + k[1], h[2] + k[2])
            } else {
                GRAD_SMALL_NR
            };
            j += 1;
        }
        i += 1;
    }
    out
}

/// (−1)^{τ+ν+φ} per ket Hermite index of degree <= lcd + 1 (`signs` of
/// [`quartet_gradient_general`]).
const fn grad_sign_table<const NJ1: usize>(lcd: usize) -> [f64; NJ1] {
    let ks = hermite_array::<NJ1>(lcd + 1);
    let mut out = [0.0; NJ1];
    let mut j = 0;
    while j < NJ1 {
        out[j] = if (ks[j][0] + ks[j][1] + ks[j][2]).is_multiple_of(2) { 1.0 } else { -1.0 };
        j += 1;
    }
    out
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Σ_{abcd} G_{ab,cd} ∂(ab|cd)/∂{A, B, C, D} for one shell quartet, where `g`
/// (row-major `ncomp_ab × ncomp_cd`) already holds the symmetrized Γ times the
/// permutational multiplicity. theory.typ eq:eri, eq:center-deriv, eq:gamma
///
/// With X_{ij} = R_{t_i+τ_j, u_i+ν_j, v_i+φ_j}(α, P−Q) and s_j = (−1)^{τ+ν+φ}:
/// * Z_v[ab][j] = s_j Σ_cd G[ab][cd] E^{cd}_j[cd] (independent of the bra primitive),
///   H[i][ab] = Σ_j Z_v[ab][j] X_ij, and ∂/∂A, ∂/∂B = pref Σ_{i,ab} Ẽ^{A,B}_i[ab] H[i][ab];
/// * Z_C[ab][j] likewise with the ket's C-derivative coefficients,
///   Q[ab][j] = Σ_i E^{ab}_i[ab] X_ij, and ∂/∂C = pref Σ_{ab,j} Z_C[ab][j] Q[ab][j];
/// * ∂/∂D = −(∂/∂A + ∂/∂B + ∂/∂C) (translational invariance).
///
/// `omega = Some(ω)` uses the operator erf(ωr)/r instead of 1/r (theory.typ eq:eri-erf; VERIFIED(Adamson1999, derived from Eqs. (1), (11)-(13), p. 922; the paper prints the erfc integral)
/// the centre-derivative rule is unchanged, because the kernel depends only on r₁ − r₂
/// and the centres enter only through the Hermite coefficients and P − Q).
///
/// Pairs with l_ab, l_cd <= GRAD_SMALL_LAB go to [`quartet_gradient_small`], the others
/// to [`quartet_gradient_general`]; both give bit-identical results.
fn quartet_gradient(
    ab: &GradPair,
    cd: &GradPair,
    g: &[f64],
    omega: Option<f64>,
    s: &mut QuartetScratch,
) -> [[f64; 3]; 4] {
    match (ab.lab, cd.lab) {
        (0, 0) => quartet_gradient_small::<0, 0, 1, 4, 1, 4>(ab, cd, g, omega, s),
        (0, 1) => quartet_gradient_small::<0, 1, 1, 4, 4, 10>(ab, cd, g, omega, s),
        (0, 2) => quartet_gradient_small::<0, 2, 1, 4, 10, 20>(ab, cd, g, omega, s),
        (1, 0) => quartet_gradient_small::<1, 0, 4, 10, 1, 4>(ab, cd, g, omega, s),
        (1, 1) => quartet_gradient_small::<1, 1, 4, 10, 4, 10>(ab, cd, g, omega, s),
        (1, 2) => quartet_gradient_small::<1, 2, 4, 10, 10, 20>(ab, cd, g, omega, s),
        (2, 0) => quartet_gradient_small::<2, 0, 10, 20, 1, 4>(ab, cd, g, omega, s),
        (2, 1) => quartet_gradient_small::<2, 1, 10, 20, 4, 10>(ab, cd, g, omega, s),
        (2, 2) => quartet_gradient_small::<2, 2, 10, 20, 10, 20>(ab, cd, g, omega, s),
        _ => quartet_gradient_general(ab, cd, g, omega, s),
    }
}

/// [`quartet_gradient_general`] for l_ab = `LAB`, l_cd = `LCD` <= GRAD_SMALL_LAB
/// (`NI`, `NI1`, `NJ`, `NJ1` = n_hermite of LAB, LAB+1, LCD, LCD+1): the same sums in
/// the same order, with the compact R table of [`hermite_r_grad_small`] (the same
/// operations as [`HermiteR::compute`], with ltot a constant), the R index and sign
/// tables built at compile time, and fixed-size buffers. The results are bit-identical.
fn quartet_gradient_small<
    const LAB: usize,
    const LCD: usize,
    const NI: usize,
    const NI1: usize,
    const NJ: usize,
    const NJ1: usize,
>(
    ab: &GradPair,
    cd: &GradPair,
    g: &[f64],
    omega: Option<f64>,
    s: &mut QuartetScratch,
) -> [[f64; 3]; 4] {
    const { assert!(NI == n_hermite(LAB) && NI1 == n_hermite(LAB + 1) && NJ == n_hermite(LCD) && NJ1 == n_hermite(LCD + 1)) };
    const { assert!(LAB <= GRAD_SMALL_LAB && LCD <= GRAD_SMALL_LAB) };
    let ridx: [[usize; NJ1]; NI1] = const { grad_ridx_table::<NI1, NJ1>(LAB, LCD) };
    let signs: [f64; NJ1] = const { grad_sign_table::<NJ1>(LCD) };
    debug_assert!(ab.lab == LAB && cd.lab == LCD);
    let (nab, ncd) = (ab.ncomp, cd.ncomp);
    debug_assert!(nab <= GRAD_SMALL_NCOMP && ncd <= GRAD_SMALL_NCOMP);
    let b = &mut s.small;
    let mut rmat = [[0.0; NJ1]; NI1];
    let mut gab = [[0.0; 3]; 2];
    let mut gc = [0.0; 3];
    for pc in &cd.prims {
        // Z_v and Z_C for this ket primitive pair. For each ab, all NJ + 3 NJ1 sums
        // Σ_cd G[ab][cd] E[cd] are accumulated together over cd from the transposed
        // coefficients `et` (contiguous per cd), as independent add chains. Each is
        // the same fold as `dot(grow, e)` (start -0.0, add g·e in cd order), so the
        // values are bit-identical to separate dots.
        let w = NJ + 3 * NJ1;
        debug_assert!(pc.et.len() == ncd * w);
        let zv = &mut b.zv[..nab * NJ];
        let zc = &mut b.zc[..3 * nab * NJ1];
        for a in 0..nab {
            let grow = &g[a * ncd..(a + 1) * ncd];
            let mut accv = [-0.0f64; NJ];
            let mut accc = [[-0.0f64; NJ1]; 3];
            for (&gv, et) in grow.iter().zip(pc.et.chunks_exact(w)) {
                let (ev, ec) = et.split_at(NJ);
                for (sum, &e) in accv.iter_mut().zip(ev) {
                    *sum += gv * e;
                }
                // The C derivative of (cd) is the "A" (first-centre) set of the pair.
                for (acck, eck) in accc.iter_mut().zip(ec.as_chunks::<NJ1>().0) {
                    for (sum, &e) in acck.iter_mut().zip(eck) {
                        *sum += gv * e;
                    }
                }
            }
            for ((z, &sum), &sign) in zv[a * NJ..(a + 1) * NJ].iter_mut().zip(&accv).zip(&signs) {
                *z = sign * sum;
            }
            for (k, acck) in accc.iter().enumerate() {
                for ((z, &sum), &sign) in zc[(k * nab + a) * NJ1..(k * nab + a + 1) * NJ1].iter_mut().zip(acck).zip(&signs) {
                    *z = sign * sum;
                }
            }
        }
        for pa in &ab.prims {
            let (p, q) = (pa.p, pc.p);
            let mut alpha = p * q / (p + q);
            let pq = [0, 1, 2].map(|k| pa.center[k] - pc.center[k]);
            let mut pref = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt());
            if let Some(w) = omega {
                // theory.typ eq:eri-erf
                let (a_eff, scale) = erf_attenuation(alpha, w);
                alpha = a_eff;
                pref *= scale;
            }
            let rdata = hermite_r_grad_small(LAB + LCD + 1, alpha, pq, b);
            for (row, ix) in rmat.iter_mut().zip(&ridx) {
                for (x, &i) in row.iter_mut().zip(ix) {
                    *x = rdata[i];
                }
            }
            // H[i][ab], i < ni1.
            let hv = &mut b.hv[..NI1 * nab];
            for (i, xr) in rmat.iter().enumerate() {
                let xrow = &xr[..NJ];
                for a in 0..nab {
                    hv[i * nab + a] = dot(&b.zv[a * NJ..(a + 1) * NJ], xrow);
                }
            }
            // The six dot(Ẽ_set, H) (sets 1..=6 → A_x..B_z) in one pass, so that the six
            // independent add chains overlap. Each accumulator is the same fold as `dot`
            // (start -0.0, add e·h in index order), so the result is bit-identical.
            let es: [&[f64]; 6] = std::array::from_fn(|k| &pa.e[(k + 1) * NI1 * nab..(k + 2) * NI1 * nab]);
            let mut acc = [-0.0f64; 6];
            for (x, &h) in hv.iter().enumerate() {
                for (sum, e) in acc.iter_mut().zip(&es) {
                    *sum += e[x] * h;
                }
            }
            for (k, &sum) in acc.iter().enumerate() {
                gab[k / 3][k % 3] += pref * sum;
            }
            // Q[ab][j], j < nj1, from the value coefficients (i < ni).
            let qv = &mut b.q[..nab * NJ1];
            qv.fill(0.0);
            for (i, xrow) in rmat.iter().enumerate().take(NI) {
                let ev = &pa.e[i * nab..(i + 1) * nab];
                for (a, &e) in ev.iter().enumerate() {
                    if e == 0.0 {
                        continue;
                    }
                    for (o, x) in qv[a * NJ1..(a + 1) * NJ1].iter_mut().zip(xrow) {
                        *o += e * x;
                    }
                }
            }
            // The three dot(Z_C[k], Q) in one pass, the same fold as `dot` (see above).
            let zs: [&[f64]; 3] = std::array::from_fn(|k| &b.zc[k * nab * NJ1..(k + 1) * nab * NJ1]);
            let mut acc = [-0.0f64; 3];
            for (x, &qx) in qv.iter().enumerate() {
                for (sum, z) in acc.iter_mut().zip(&zs) {
                    *sum += z[x] * qx;
                }
            }
            for (gck, &sum) in gc.iter_mut().zip(&acc) {
                *gck += pref * sum;
            }
        }
    }
    let gd = [0, 1, 2].map(|k| -(gab[0][k] + gab[1][k] + gc[k]));
    [gab[0], gab[1], gc, gd]
}

/// [`quartet_gradient`] for any angular momenta.
fn quartet_gradient_general(
    ab: &GradPair,
    cd: &GradPair,
    g: &[f64],
    omega: Option<f64>,
    s: &mut QuartetScratch,
) -> [[f64; 3]; 4] {
    let (nab, ncd) = (ab.ncomp, cd.ncomp);
    let (ni, ni1) = (n_hermite(ab.lab), n_hermite(ab.lab + 1));
    let (nj, nj1) = (n_hermite(cd.lab), n_hermite(cd.lab + 1));
    let ltot = ab.lab + cd.lab + 1;
    let dim = ltot + 1;
    s.ridx.clear();
    for h in &ab.tuv[..ni1] {
        for k in &cd.tuv[..nj1] {
            let deg = h[0] + h[1] + h[2] + k[0] + k[1] + k[2];
            s.ridx.push(if deg <= ltot { ((h[0] + k[0]) * dim + h[1] + k[1]) * dim + h[2] + k[2] } else { usize::MAX });
        }
    }
    let signs: Vec<f64> = cd.tuv[..nj1].iter().map(|k| if (k[0] + k[1] + k[2]) % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let mut gab = [[0.0; 3]; 2];
    let mut gc = [0.0; 3];
    for pc in &cd.prims {
        // Z_v and Z_C for this ket primitive pair.
        s.zv.clear();
        s.zv.resize(nab * nj, 0.0);
        s.zc.clear();
        s.zc.resize(3 * nab * nj1, 0.0);
        for (j, &sign) in signs.iter().enumerate() {
            let ev = &pc.e[j * ncd..(j + 1) * ncd];
            for a in 0..nab {
                let grow = &g[a * ncd..(a + 1) * ncd];
                if j < nj {
                    s.zv[a * nj + j] = sign * dot(grow, ev);
                }
                for k in 0..3 {
                    // The C derivative of (cd) is the "A" (first-centre) set of the pair.
                    let ec = &pc.e[((1 + k) * nj1 + j) * ncd..((1 + k) * nj1 + j + 1) * ncd];
                    s.zc[(k * nab + a) * nj1 + j] = sign * dot(grow, ec);
                }
            }
        }
        for pa in &ab.prims {
            let (p, q) = (pa.p, pc.p);
            let mut alpha = p * q / (p + q);
            let pq = [0, 1, 2].map(|k| pa.center[k] - pc.center[k]);
            let mut pref = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt());
            if let Some(w) = omega {
                // theory.typ eq:eri-erf
                let (a_eff, scale) = erf_attenuation(alpha, w);
                alpha = a_eff;
                pref *= scale;
            }
            s.r.compute(ltot, alpha, pq);
            let rdata = s.r.as_slice();
            s.rmat.clear();
            s.rmat.extend(s.ridx.iter().map(|&ix| if ix == usize::MAX { 0.0 } else { rdata[ix] }));
            // H[i][ab], i < ni1.
            s.hv.clear();
            s.hv.resize(ni1 * nab, 0.0);
            for i in 0..ni1 {
                let xrow = &s.rmat[i * nj1..i * nj1 + nj];
                for a in 0..nab {
                    s.hv[i * nab + a] = dot(&s.zv[a * nj..(a + 1) * nj], xrow);
                }
            }
            let n1 = ni1;
            for (set, out) in [(1, (0, 0)), (2, (0, 1)), (3, (0, 2)), (4, (1, 0)), (5, (1, 1)), (6, (1, 2))] {
                let e = &pa.e[set * n1 * nab..(set + 1) * n1 * nab];
                gab[out.0][out.1] += pref * dot(e, &s.hv);
            }
            // Q[ab][j], j < nj1, from the value coefficients (i < ni).
            s.q.clear();
            s.q.resize(nab * nj1, 0.0);
            for i in 0..ni {
                let xrow = &s.rmat[i * nj1..(i + 1) * nj1];
                let ev = &pa.e[i * nab..(i + 1) * nab];
                for (a, &e) in ev.iter().enumerate() {
                    if e == 0.0 {
                        continue;
                    }
                    for (o, x) in s.q[a * nj1..(a + 1) * nj1].iter_mut().zip(xrow) {
                        *o += e * x;
                    }
                }
            }
            for (k, gck) in gc.iter_mut().enumerate() {
                *gck += pref * dot(&s.zc[k * nab * nj1..(k + 1) * nab * nj1], &s.q);
            }
        }
    }
    let gd = [0, 1, 2].map(|k| -(gab[0][k] + gab[1][k] + gc[k]));
    [gab[0], gab[1], gc, gd]
}

/// Σ_μνλσ Γ_μνλσ (μν|λσ)^(X) with Γ = ½[P_μν P_λσ − (a_HF/2) P_μλ P_νσ] for a
/// symmetric AO density `p`. theory.typ eq:grad (second term), eq:gamma
///
/// `exchange` is a_HF (an `f64`) or an [`ExchangeKernel`]; for a range-separated
/// kernel the exchange part becomes −¼ Σ P_μλ P_νσ [α (μν|λσ)^(X) + β (μν|λσ)_ω^(X)]
/// (theory.typ eq:cam-grad).
pub fn two_electron_gradient(
    mol: &Molecule,
    basis: &Basis,
    p: &Matrix,
    exchange: impl Into<ExchangeKernel>,
) -> Result<Vec<[f64; 3]>> {
    check_square(p, basis.nao(), "density matrix")?;
    let kernel: ExchangeKernel = exchange.into();
    let a_hf = kernel.alpha;
    // Attenuated exchange part (theory.typ eq:cam-grad), if any.
    let erf_part = kernel.is_range_separated().then_some((kernel.beta, kernel.omega));
    let natom = mol.atoms().len();
    let shells = basis.shells();
    let pc = to_cartesian(basis, p);
    let (coff, _) = cartesian_offsets(basis);
    let mut pairs_idx = Vec::new();
    for a in 0..shells.len() {
        for b in 0..=a {
            pairs_idx.push((a, b));
        }
    }
    let pairs: Vec<GradPair> = pairs_idx.par_iter().map(|&(a, b)| GradPair::new(basis, a, b)).collect();
    let q = cartesian_schwarz_factors(basis);
    // max |P| per shell-pair block (for the density-weighted screening).
    let pmax: Vec<f64> = pairs_idx
        .iter()
        .map(|&(a, b)| {
            let mut m: f64 = 0.0;
            for x in 0..shells[a].n_cartesian() {
                for y in 0..shells[b].n_cartesian() {
                    m = m.max(pc[(coff[a] + x, coff[b] + y)].abs());
                }
            }
            m
        })
        .collect();
    let pm = |a: usize, b: usize| pmax[pair_index(a, b)];
    let parts: Vec<Vec<[f64; 3]>> = (0..pairs.len())
        .into_par_iter()
        .map_init(QuartetScratch::default, |s, ij| {
            let mut grad = vec![[0.0; 3]; natom];
            let ab = &pairs[ij];
            let (a, b) = (ab.sa, ab.sb);
            for (kl, cd) in pairs[..=ij].iter().enumerate() {
                let (c, d) = (cd.sa, cd.sb);
                let pexch = (pm(a, c) * pm(b, d)).max(pm(a, d) * pm(b, c));
                let gmax = 0.5 * (pm(a, b) * pm(c, d) + 0.5 * a_hf.abs() * pexch);
                // The attenuated integrals are bounded by the Coulomb ones
                // ((μν|μν)_ω <= (μν|μν)), so Q screens them too.
                let erf_screened =
                    erf_part.is_none_or(|(beta, _)| q[ij] * q[kl] * 0.25 * beta.abs() * pexch < GRADIENT_SCREENING_THRESHOLD);
                if q[ij] * q[kl] * gmax < GRADIENT_SCREENING_THRESHOLD && erf_screened {
                    continue;
                }
                // Multiplicity of the shell quartet among all ordered quartets (eq:eri-sym).
                let mut deg = 1.0;
                if a != b {
                    deg *= 2.0;
                }
                if c != d {
                    deg *= 2.0;
                }
                if ij != kl {
                    deg *= 2.0;
                }
                // Symmetrized Γ (eq:gamma) over Cartesian components.
                let (na, nb, nc, nd) =
                    (shells[a].n_cartesian(), shells[b].n_cartesian(), shells[c].n_cartesian(), shells[d].n_cartesian());
                s.gamma.clear();
                for x in 0..na {
                    let mu = coff[a] + x;
                    for y in 0..nb {
                        let nu = coff[b] + y;
                        for z in 0..nc {
                            let la = coff[c] + z;
                            for w in 0..nd {
                                let si = coff[d] + w;
                                let coul = pc[(mu, nu)] * pc[(la, si)];
                                let exch = pc[(mu, la)] * pc[(nu, si)] + pc[(mu, si)] * pc[(nu, la)];
                                s.gamma.push(deg * 0.5 * (coul - 0.25 * a_hf * exch));
                            }
                        }
                    }
                }
                let gamma = std::mem::take(&mut s.gamma);
                let f = quartet_gradient(ab, cd, &gamma, None, s);
                s.gamma = gamma;
                let atoms = [shells[a].atom, shells[b].atom, shells[c].atom, shells[d].atom];
                for (&atom, fk) in atoms.iter().zip(f) {
                    for k in 0..3 {
                        grad[atom][k] += fk[k];
                    }
                }
                if let Some((beta, omega)) = erf_part.filter(|_| !erf_screened) {
                    // Γ_ω = −(β/4) P_μλ P_νσ, symmetrized (theory.typ eq:cam-grad).
                    s.gamma.clear();
                    for x in 0..na {
                        let mu = coff[a] + x;
                        for y in 0..nb {
                            let nu = coff[b] + y;
                            for z in 0..nc {
                                let la = coff[c] + z;
                                for w in 0..nd {
                                    let si = coff[d] + w;
                                    let exch = pc[(mu, la)] * pc[(nu, si)] + pc[(mu, si)] * pc[(nu, la)];
                                    s.gamma.push(deg * 0.5 * (-0.25 * beta * exch));
                                }
                            }
                        }
                    }
                    let gamma = std::mem::take(&mut s.gamma);
                    let f = quartet_gradient(ab, cd, &gamma, Some(omega), s);
                    s.gamma = gamma;
                    for (&atom, fk) in atoms.iter().zip(f) {
                        for k in 0..3 {
                            grad[atom][k] += fk[k];
                        }
                    }
                }
            }
            grad
        })
        .collect();
    let mut g = vec![[0.0; 3]; natom];
    for part in parts {
        add_into(&mut g, &part);
    }
    Ok(g)
}

// ---------------------------------------------------------------------------
// Exchange-correlation term
// ---------------------------------------------------------------------------

/// E_xc^(X) of theory.typ eq:xc-grad for density `p` on the grid of `xc` (whose
/// basis must be `basis`). Grid-weight derivatives are neglected (theory.typ
/// section 10.5), so the result is not translationally invariant to grid accuracy.
///
/// Per grid block, with a_g = w_g v_ρ, b_g = 2 w_g v_σ ∇ρ, t_μ = Σ_ν P_μν χ_ν and
/// u_μ = Σ_ν P_μν (b·∇χ_ν), the contribution of AO μ (on atom A) is
/// −2 Σ_g [∂_kχ_μ (a_g t_μ + u_μ) + (b·∇∂_kχ_μ) t_μ], which is eq:xc-grad regrouped.
pub fn xc_gradient(mol: &Molecule, basis: &Basis, xc: &XcIntegrator, p: &Matrix) -> Result<Vec<[f64; 3]>> {
    check_square(p, basis.nao(), "density matrix")?;
    let natom = mol.atoms().len();
    let functional = xc.functional();
    if !functional.has_density_part() {
        return Ok(vec![[0.0; 3]; natom]);
    }
    let gga = functional.is_gga();
    let deriv = if gga { 2 } else { 1 };
    let ao_atom = basis.ao_atom();
    let grid = xc.grid();
    let parts: Vec<Vec<[f64; 3]>> = (0..grid.blocks.len())
        .into_par_iter()
        .map(|ib| {
            let mut g = vec![[0.0; 3]; natom];
            let aos = xc.block_aos(ib);
            let na = aos.len();
            if na == 0 {
                return g;
            }
            let range = grid.blocks[ib].range.clone();
            let w = &grid.weights[range.clone()];
            let np = w.len();
            let ao = eval_ao_shells(basis, xc.block_shells(ib), &grid.points[range], deriv);
            let psub: Vec<f64> = aos.iter().flat_map(|&a| aos.iter().map(move |&b| p[(a, b)])).collect();
            let (rho, grad_rho, t) = density_on_points(&ao, &psub, gga);
            let pd = xc_on_points(functional, rho, grad_rho);
            let a_g: Vec<f64> = (0..np).map(|k| w[k] * pd.v_rho[k]).collect();
            let b_g: Vec<[f64; 3]> = if gga {
                (0..np).map(|k| pd.grad_rho[k].map(|x| 2.0 * w[k] * pd.v_sigma[k] * x)).collect()
            } else {
                Vec::new()
            };
            // u_μ(g) = Σ_ν P_μν b_g·∇χ_ν(g)
            let mut u = vec![0.0; na * np];
            if gga {
                let mut phi = vec![0.0; na * np];
                for nu in 0..na {
                    for i in 0..3 {
                        let d = ao.get(D_X + i, nu);
                        for ((ph, b), dk) in phi[nu * np..(nu + 1) * np].iter_mut().zip(&b_g).zip(d) {
                            *ph += b[i] * dk;
                        }
                    }
                }
                for mu in 0..na {
                    let um = &mut u[mu * np..(mu + 1) * np];
                    for nu in 0..na {
                        let pv = psub[mu * na + nu];
                        if pv == 0.0 {
                            continue;
                        }
                        for (o, f) in um.iter_mut().zip(&phi[nu * np..(nu + 1) * np]) {
                            *o += pv * f;
                        }
                    }
                }
            }
            for mu in 0..na {
                let tm = &t[mu * np..(mu + 1) * np];
                let um = &u[mu * np..(mu + 1) * np];
                let atom = ao_atom[aos[mu]];
                for (kd, gk) in g[atom].iter_mut().enumerate() {
                    let dk = ao.get(D_X + kd, mu);
                    let mut s = 0.0;
                    for k in 0..np {
                        s += dk[k] * (a_g[k] * tm[k] + um[k]);
                    }
                    if gga {
                        let h = [0, 1, 2].map(|i| ao.get(hessian_component(i, kd), mu));
                        for k in 0..np {
                            let bh = b_g[k][0] * h[0][k] + b_g[k][1] * h[1][k] + b_g[k][2] * h[2][k];
                            s += bh * tm[k];
                        }
                    }
                    *gk -= 2.0 * s;
                }
            }
            g
        })
        .collect();
    let mut g = vec![[0.0; 3]; natom];
    for part in parts {
        add_into(&mut g, &part);
    }
    Ok(g)
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

fn scf_gradient(
    mol: &Molecule,
    basis: &Basis,
    scf: &ScfResult,
    exchange: ExchangeKernel,
    xc: Option<&XcIntegrator>,
) -> Result<GradientComponents> {
    let n = basis.nao();
    check_square(&scf.density, n, "SCF density")?;
    if scf.coefficients.rows() != n || scf.orbital_energies.len() < scf.n_occupied {
        return Err(QcError::InvalidInput("SCF result does not match the basis".into()));
    }
    let w = energy_weighted_density(scf);
    let pc = to_cartesian(basis, &scf.density);
    let wc = to_cartesian(basis, &w);
    let (one_electron, overlap) = one_electron_terms(mol, basis, OneElectronRequest { p: Some(&pc), w: Some(&wc) });
    let two_electron = two_electron_gradient(mol, basis, &scf.density, exchange)?;
    let xc_part = match xc {
        Some(x) => xc_gradient(mol, basis, x, &scf.density)?,
        None => vec![[0.0; 3]; mol.atoms().len()],
    };
    Ok(GradientComponents {
        one_electron,
        two_electron,
        overlap,
        xc: xc_part,
        nuclear_repulsion: nuclear_repulsion_gradient(mol),
    })
}

/// Analytic RHF gradient (theory.typ eq:grad with a_HF = 1, no E_xc) for a converged
/// SCF result of `mol` in `basis`.
pub fn rhf_gradient(mol: &Molecule, basis: &Basis, scf: &ScfResult) -> Result<GradientComponents> {
    scf_gradient(mol, basis, scf, ExchangeKernel::global(1.0), None)
}

/// Analytic RKS gradient (theory.typ eq:grad, eq:cam-grad) for a converged SCF result
/// obtained with the integrator `xc` (functional, grid and basis of the SCF). Works for
/// any functional, including pure HF (`Functional::hf()`, no grid contribution) and
/// range-separated hybrids.
pub fn rks_gradient(mol: &Molecule, basis: &Basis, scf: &ScfResult, xc: &XcIntegrator) -> Result<GradientComponents> {
    scf_gradient(mol, basis, scf, xc.functional().exchange_kernel(), Some(xc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{BasisOptions, BasisSetDefinition};
    use crate::functional::Functional;
    use crate::grid::{GridSettings, MolecularGrid};
    use crate::integrals::{EriTensor, core_hamiltonian, overlap};
    use crate::molecule::Unit;

    fn mol_at(c: &[f64]) -> Molecule {
        let m = Molecule::from_symbols(
            &["O", "H", "N"],
            &[[0.0, 0.1, 0.2], [1.5, -0.3, 1.4], [-1.9, 1.1, -0.8]],
            Unit::Bohr,
            0,
        )
        .unwrap();
        if c.is_empty() { m } else { m.with_coordinates(c).unwrap() }
    }

    fn basis(m: &Molecule, cart: bool) -> Basis {
        Basis::new(m, &BasisSetDefinition::builtin("6-31g*").unwrap(), BasisOptions { cartesian_d: cart }).unwrap()
    }

    /// A fixed symmetric "density" with structure but no physical meaning.
    fn test_matrix(n: usize, seed: usize) -> Matrix {
        Matrix::from_fn(n, n, |i, j| {
            let (a, b) = (i.min(j), i.max(j));
            ((a * 13 + b * 7 + seed * 3 + a * b) % 17) as f64 * 0.02 - 0.15 + if i == j { 0.4 } else { 0.0 }
        })
    }

    /// Derivative integrals contracted with fixed matrices vs central finite
    /// differences of the contracted integrals (fixed matrices, moving basis).
    #[test]
    fn derivative_integrals_match_finite_differences() {
        for cart in [false, true] {
            let m0 = mol_at(&[]);
            let b0 = basis(&m0, cart);
            let n = b0.nao();
            let p = test_matrix(n, 1);
            let w = test_matrix(n, 2);
            let a_hf = 0.7;
            let kernel = ExchangeKernel { alpha: 0.19, beta: 0.46, omega: 0.33 };
            let g1 = one_electron_gradient(&m0, &b0, &p).unwrap();
            let gs = overlap_gradient(&m0, &b0, &w).unwrap();
            let g2 = two_electron_gradient(&m0, &b0, &p, a_hf).unwrap();
            let g3 = two_electron_gradient(&m0, &b0, &p, kernel).unwrap();
            let x0 = m0.coordinates();
            let f1 = |x: &[f64]| {
                let m = mol_at(x);
                p.dot(&core_hamiltonian(&basis(&m, cart), &m))
            };
            let fs = |x: &[f64]| -w.dot(&overlap(&basis(&mol_at(x), cart)));
            let f2 = |x: &[f64]| {
                let eri = EriTensor::compute(&basis(&mol_at(x), cart), 0.0);
                let (j, k) = eri.coulomb_exchange(&p, true);
                0.5 * p.dot(&j) - 0.25 * a_hf * p.dot(&k.unwrap())
            };
            // theory.typ eq:cam-k, eq:cam-fock: ½ P·J − ¼ P·K^x. VERIFIED(Yanai2004, derived from Eqs. (6), (7), p. 52)
            let f3 = |x: &[f64]| {
                let b = basis(&mol_at(x), cart);
                let eri = EriTensor::compute(&b, 0.0);
                let (j, _) = eri.coulomb_exchange(&p, false);
                let kx = eri.exchange_tensor(&b, kernel, 0.0).unwrap().exchange(&p);
                0.5 * p.dot(&j) - 0.25 * p.dot(&kx)
            };
            let h = 1e-4;
            let mut worst = [0.0f64; 4];
            type EnergyFn<'a> = &'a dyn Fn(&[f64]) -> f64;
            let fns: [EnergyFn; 4] = [&f1, &fs, &f2, &f3];
            let grads = [&g1, &gs, &g2, &g3];
            for i in 0..x0.len() {
                let (mut xp, mut xm) = (x0.clone(), x0.clone());
                xp[i] += h;
                xm[i] -= h;
                let (a, k) = (i / 3, i % 3);
                for (t, (f, g)) in fns.iter().zip(grads).enumerate() {
                    let fd = (f(&xp) - f(&xm)) / (2.0 * h);
                    let dev = (fd - g[a][k]).abs();
                    worst[t] = worst[t].max(dev);
                    assert!(dev < 2e-7 * fd.abs().max(1.0), "cart={cart} term {t} coord {i}: {} vs fd {fd}", g[a][k]);
                }
            }
            println!(
                "cart={cart}: worst |analytic - FD| one-electron {:.1e}, overlap {:.1e}, two-electron {:.1e}, \
                 range-separated two-electron {:.1e}",
                worst[0], worst[1], worst[2], worst[3]
            );
            // A kernel with β = 0 is the a_HF form, bit for bit.
            assert_eq!(two_electron_gradient(&m0, &b0, &p, ExchangeKernel::global(a_hf)).unwrap(), g2);
        }
    }

    /// eq:xc-grad equals the derivative of E_xc[P] for a fixed density matrix and a
    /// *frozen* grid (only the basis functions move).
    #[test]
    fn xc_gradient_matches_frozen_grid_finite_differences() {
        let m0 = mol_at(&[]);
        let b0 = basis(&m0, false);
        let p = crate::scf::run_rhf(&m0, &b0, &crate::scf::ScfOptions::default()).unwrap().density;
        let grid = MolecularGrid::new(&m0, GridSettings { n_radial: 60, n_theta: 14 }).unwrap();
        for fun in [Functional::svwn5(), Functional::b3lyp(), Functional::cam_b3lyp()] {
            let xc0 = XcIntegrator::with_grid(&b0, &fun, grid.clone(), Default::default()).unwrap();
            let g = xc_gradient(&m0, &b0, &xc0, &p).unwrap();
            let x0 = m0.coordinates();
            let e = |x: &[f64]| {
                let b = basis(&mol_at(x), false);
                XcIntegrator::with_grid(&b, &fun, grid.clone(), Default::default()).unwrap().evaluate_density(&p).energy
            };
            let h = 1e-4;
            let mut worst: f64 = 0.0;
            for i in 0..x0.len() {
                let (mut xp, mut xm) = (x0.clone(), x0.clone());
                xp[i] += h;
                xm[i] -= h;
                let fd = (e(&xp) - e(&xm)) / (2.0 * h);
                let dev = (fd - g[i / 3][i % 3]).abs();
                worst = worst.max(dev);
                assert!(dev < 1e-7 * fd.abs().max(1.0), "{} coord {i}: {} vs fd {fd}", fun.name(), g[i / 3][i % 3]);
            }
            println!("{}: worst |analytic - frozen-grid FD| {worst:.1e}", fun.name());
        }
    }

    /// The compact R table equals `HermiteR::compute` bit for bit (same environment,
    /// so the libm `exp` of the Boys function is the same).
    #[test]
    fn small_r_table_matches_hermite_r() {
        let mut general = HermiteR::new();
        let mut small = GradSmallBuffers::default();
        for lmax in 0..=2 * GRAD_SMALL_LAB + 1 {
            for (p, r) in [
                (0.37, [0.0, 0.0, 0.0]),
                (1.9, [0.3, -1.2, 0.8]),
                (25.0, [-2.5, 0.1, 3.7]),
                (0.012, [4.0, -6.0, 1.5]),
                (310.0, [1e-3, 2e-3, -5e-4]),
            ] {
                general.compute(lmax, p, r);
                let rs = hermite_r_grad_small(lmax, p, r, &mut small);
                let list = hermite_list(lmax);
                assert_eq!(list.len(), n_hermite(lmax));
                for t in 0..=lmax {
                    for u in 0..=lmax - t {
                        for v in 0..=lmax - t - u {
                            let e = hermite_index(&list, t, u, v);
                            assert_eq!(rs[e].to_bits(), general.get(t, u, v).to_bits(), "lmax {lmax} ({t},{u},{v})");
                        }
                    }
                }
                assert_eq!(rs[GRAD_SMALL_NR].to_bits(), 0.0f64.to_bits());
            }
        }
    }

    /// The compile-time Hermite list is `hermite_list`.
    #[test]
    fn hermite_array_matches_hermite_list() {
        assert_eq!(hermite_array::<1>(0).to_vec(), hermite_list(0));
        assert_eq!(hermite_array::<4>(1).to_vec(), hermite_list(1));
        assert_eq!(hermite_array::<10>(2).to_vec(), hermite_list(2));
        assert_eq!(hermite_array::<20>(3).to_vec(), hermite_list(3));
        assert_eq!(hermite_array::<GRAD_SMALL_NR>(GRAD_SMALL_LTOT).to_vec(), hermite_list(GRAD_SMALL_LTOT));
    }

    /// `quartet_gradient` (the specialized kernels for l_ab, l_cd <= 2) equals
    /// `quartet_gradient_general` bit for bit for every shell quartet of a 6-31G*
    /// molecule (s, p and Cartesian d shells), random Γ, without and with ω.
    #[test]
    fn small_quartet_gradient_matches_general() {
        let m = mol_at(&[]);
        let b = basis(&m, true);
        let nsh = b.shells().len();
        let mut pairs = Vec::new();
        for a in 0..nsh {
            for c in 0..=a {
                pairs.push(GradPair::new(&b, a, c));
            }
        }
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut seen = [[false; 3]; 3];
        let (mut s1, mut s2) = (QuartetScratch::default(), QuartetScratch::default());
        for (ij, ab) in pairs.iter().enumerate() {
            for cd in &pairs[..=ij] {
                let g: Vec<f64> = (0..ab.ncomp * cd.ncomp).map(|_| rand()).collect();
                for omega in [None, Some(0.33)] {
                    let fast = quartet_gradient(ab, cd, &g, omega, &mut s1);
                    let slow = quartet_gradient_general(ab, cd, &g, omega, &mut s2);
                    let bits = |f: [[f64; 3]; 4]| f.map(|r| r.map(f64::to_bits));
                    assert_eq!(bits(fast), bits(slow), "lab {} lcd {} omega {omega:?}", ab.lab, cd.lab);
                }
                if ab.lab <= GRAD_SMALL_LAB && cd.lab <= GRAD_SMALL_LAB {
                    seen[ab.lab][cd.lab] = true;
                }
            }
        }
        assert_eq!(seen, [[true; 3]; 3], "not every (l_ab, l_cd) <= 2 was exercised");
    }

    #[test]
    fn nuclear_repulsion_gradient_matches_fd() {
        let m0 = mol_at(&[]);
        let g = nuclear_repulsion_gradient(&m0);
        let x0 = m0.coordinates();
        let h = 1e-5;
        for i in 0..x0.len() {
            let (mut xp, mut xm) = (x0.clone(), x0.clone());
            xp[i] += h;
            xm[i] -= h;
            let fd = (mol_at(&xp).nuclear_repulsion() - mol_at(&xm).nuclear_repulsion()) / (2.0 * h);
            assert!((fd - g[i / 3][i % 3]).abs() < 1e-8);
        }
    }
}

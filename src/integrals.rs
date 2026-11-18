//! Contracted AO integrals: overlap, kinetic, nuclear attraction, dipole and
//! two-electron repulsion integrals (theory.typ sections 4–5).
//!
//! Primitive integrals come from [`crate::hermite`]; this module applies the
//! contraction coefficients d̃_k, the Cartesian factors c_lmn (eq:contracted) and
//! the spherical d transformation (eq:sph-d).

use crate::basis::{Basis, Shell, cartesian_components, cartesian_norm, transform_axis};
use crate::hermite::{HermiteR, PrimitivePair};
use crate::linalg::Matrix;
use crate::molecule::Molecule;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Normalized Cartesian components and their factors c_lmn for a shell.
fn components(sh: &Shell) -> (Vec<[usize; 3]>, Vec<f64>) {
    let c = cartesian_components(sh.l);
    let n = c.iter().map(|&lmn| cartesian_norm(lmn)).collect();
    (c, n)
}

/// Generic one-electron matrix builder. `kernel(pair, ca, cb, block)` must add the
/// primitive integrals for all Cartesian component pairs into `block` (row-major,
/// `ca.len() × cb.len()`); contraction, normalization and transformation happen here.
fn one_electron<F>(basis: &Basis, kernel: F) -> Matrix
where
    F: Fn(&PrimitivePair, &[[usize; 3]], &[[usize; 3]], f64, &mut [f64]),
{
    let n = basis.nao();
    let mut m = Matrix::zeros(n, n);
    let shells = basis.shells();
    for (ia, sa) in shells.iter().enumerate() {
        let (ca, na) = components(sa);
        for sb in &shells[..=ia] {
            let (cb, nb) = components(sb);
            let mut block = vec![0.0; ca.len() * cb.len()];
            for (&ea, &da) in sa.exponents.iter().zip(&sa.coefficients) {
                for (&eb, &db) in sb.exponents.iter().zip(&sb.coefficients) {
                    let pair = PrimitivePair::new(ea, sa.center, sa.l, eb, sb.center, sb.l);
                    kernel(&pair, &ca, &cb, da * db, &mut block);
                }
            }
            for x in 0..ca.len() {
                for y in 0..cb.len() {
                    block[x * cb.len() + y] *= na[x] * nb[y];
                }
            }
            let mut dims = [ca.len(), cb.len()];
            if let Some(t) = sa.spherical_transform() {
                block = transform_axis(&block, &mut dims, 0, t);
            }
            if let Some(t) = sb.spherical_transform() {
                block = transform_axis(&block, &mut dims, 1, t);
            }
            for x in 0..dims[0] {
                for y in 0..dims[1] {
                    let v = block[x * dims[1] + y];
                    m[(sa.ao_offset + x, sb.ao_offset + y)] = v;
                    m[(sb.ao_offset + y, sa.ao_offset + x)] = v;
                }
            }
        }
    }
    m
}

/// Overlap matrix S. theory.typ eq:ovlp
pub fn overlap(basis: &Basis) -> Matrix {
    one_electron(basis, |pair, ca, cb, c, block| {
        for (x, &la) in ca.iter().enumerate() {
            for (y, &lb) in cb.iter().enumerate() {
                block[x * cb.len() + y] += c * pair.overlap(la, lb);
            }
        }
    })
}

/// Kinetic-energy matrix T. theory.typ eq:kin
pub fn kinetic(basis: &Basis) -> Matrix {
    one_electron(basis, |pair, ca, cb, c, block| {
        for (x, &la) in ca.iter().enumerate() {
            for (y, &lb) in cb.iter().enumerate() {
                block[x * cb.len() + y] += c * pair.kinetic(la, lb);
            }
        }
    })
}

/// Nuclear-attraction matrix V (including -Z_C). theory.typ eq:nuc
pub fn nuclear_attraction(basis: &Basis, mol: &Molecule) -> Matrix {
    point_charge_attraction(basis, &mol.atoms().iter().map(|a| (a.z as f64, a.position)).collect::<Vec<_>>())
}

/// -Σ_C q_C ⟨χ_μ|1/|r - R_C||χ_ν⟩ for arbitrary point charges (q, position).
pub fn point_charge_attraction(basis: &Basis, charges: &[(f64, [f64; 3])]) -> Matrix {
    one_electron(basis, |pair, ca, cb, c, block| {
        let lmax = basis_l(ca) + basis_l(cb);
        let mut r = HermiteR::new();
        for &(q, pos) in charges {
            let pc = [0, 1, 2].map(|k| pair.center[k] - pos[k]);
            r.compute(lmax, pair.p, pc);
            for (x, &la) in ca.iter().enumerate() {
                for (y, &lb) in cb.iter().enumerate() {
                    block[x * cb.len() + y] -= q * c * pair.nuclear(la, lb, &r);
                }
            }
        }
    })
}

fn basis_l(c: &[[usize; 3]]) -> usize {
    c[0][0] + c[0][1] + c[0][2]
}

/// Core Hamiltonian h = T + V. theory.typ eq:nuc
pub fn core_hamiltonian(basis: &Basis, mol: &Molecule) -> Matrix {
    &kinetic(basis) + &nuclear_attraction(basis, mol)
}

/// Dipole integrals ⟨χ_μ|r - origin|χ_ν⟩ for x, y, z. theory.typ eq:dip-1d
pub fn dipole(basis: &Basis, origin: [f64; 3]) -> [Matrix; 3] {
    [0, 1, 2].map(|k| {
        one_electron(basis, |pair, ca, cb, c, block| {
            for (x, &la) in ca.iter().enumerate() {
                for (y, &lb) in cb.iter().enumerate() {
                    block[x * cb.len() + y] += c * pair.dipole(la, lb, origin)[k];
                }
            }
        })
    })
}

// ---------------------------------------------------------------------------
// Two-electron integrals
// ---------------------------------------------------------------------------

/// Index of the unordered pair (i, j) in packed lower-triangular storage.
#[inline]
pub fn pair_index(i: usize, j: usize) -> usize {
    if i >= j { i * (i + 1) / 2 + j } else { j * (j + 1) / 2 + i }
}

/// Hermite data of one primitive pair inside a shell pair: E_tuv for every
/// Cartesian component pair, already multiplied by d̃_a d̃_b c_a c_b.
struct PrimPairData {
    p: f64,
    center: [f64; 3],
    /// Row-major `[tuv][component pair]`.
    e: Vec<f64>,
}

/// A shell pair (a >= b) prepared for ERI evaluation.
struct ShellPairData {
    sa: usize,
    sb: usize,
    lab: usize,
    ncomp: usize,
    tuv: Vec<[usize; 3]>,
    prims: Vec<PrimPairData>,
}

impl ShellPairData {
    fn new(shells: &[Shell], sa: usize, sb: usize) -> Self {
        let (a, b) = (&shells[sa], &shells[sb]);
        let (ca, na) = components(a);
        let (cb, nb) = components(b);
        let lab = a.l + b.l;
        let mut tuv = Vec::new();
        for t in 0..=lab {
            for u in 0..=lab - t {
                for v in 0..=lab - t - u {
                    tuv.push([t, u, v]);
                }
            }
        }
        let ncomp = ca.len() * cb.len();
        let mut prims = Vec::new();
        for (&ea, &da) in a.exponents.iter().zip(&a.coefficients) {
            for (&eb, &db) in b.exponents.iter().zip(&b.coefficients) {
                let pair = PrimitivePair::new(ea, a.center, a.l, eb, b.center, b.l);
                let mut e = vec![0.0; tuv.len() * ncomp];
                for (it, &h) in tuv.iter().enumerate() {
                    for (x, &la) in ca.iter().enumerate() {
                        for (y, &lb) in cb.iter().enumerate() {
                            if h[0] > la[0] + lb[0] || h[1] > la[1] + lb[1] || h[2] > la[2] + lb[2] {
                                continue;
                            }
                            e[it * ncomp + x * cb.len() + y] =
                                da * db * na[x] * nb[y] * pair.hermite_coefficient(la, lb, h);
                        }
                    }
                }
                prims.push(PrimPairData { p: pair.p, center: pair.center, e });
            }
        }
        ShellPairData { sa, sb, lab, ncomp, tuv, prims }
    }
}

/// Scratch buffers for one thread.
#[derive(Default)]
struct EriScratch {
    r: HermiteR,
    x: Vec<f64>,
}

/// Contracted Cartesian shell quartet (ab|cd), row-major `[comp_ab][comp_cd]`. theory.typ eq:eri
fn shell_quartet_cart(ab: &ShellPairData, cd: &ShellPairData, s: &mut EriScratch) -> Vec<f64> {
    let ltot = ab.lab + cd.lab;
    let (nab, ncd) = (ab.ncomp, cd.ncomp);
    let mut out = vec![0.0; nab * ncd];
    let signs: Vec<f64> = cd.tuv.iter().map(|h| if (h[0] + h[1] + h[2]) % 2 == 0 { 1.0 } else { -1.0 }).collect();
    for pa in &ab.prims {
        for pc in &cd.prims {
            let (p, q) = (pa.p, pc.p);
            let alpha = p * q / (p + q);
            let pq = [0, 1, 2].map(|k| pa.center[k] - pc.center[k]);
            s.r.compute(ltot, alpha, pq);
            let pref = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt());
            // X[tuv][cd] = Σ_{τνφ} (-1)^{τ+ν+φ} E^{cd}_{τνφ} R_{t+τ,u+ν,v+φ}
            s.x.clear();
            s.x.resize(ab.tuv.len() * ncd, 0.0);
            for (it, h) in ab.tuv.iter().enumerate() {
                let row = &mut s.x[it * ncd..(it + 1) * ncd];
                for (jt, g) in cd.tuv.iter().enumerate() {
                    let r = signs[jt] * s.r.get(h[0] + g[0], h[1] + g[1], h[2] + g[2]);
                    let ecd = &pc.e[jt * ncd..(jt + 1) * ncd];
                    for (xv, &e) in row.iter_mut().zip(ecd) {
                        *xv += r * e;
                    }
                }
            }
            // (ab|cd) += pref Σ_tuv E^{ab}_tuv X[tuv][cd]
            for it in 0..ab.tuv.len() {
                let eab = &pa.e[it * nab..(it + 1) * nab];
                let xrow = &s.x[it * ncd..(it + 1) * ncd];
                for (ca, &e) in eab.iter().enumerate() {
                    if e == 0.0 {
                        continue;
                    }
                    let f = pref * e;
                    let orow = &mut out[ca * ncd..(ca + 1) * ncd];
                    for (o, &xv) in orow.iter_mut().zip(xrow) {
                        *o += f * xv;
                    }
                }
            }
        }
    }
    out
}

/// Shell quartet in the final (spherical or Cartesian) AO representation, shape
/// `[na, nb, nc, nd]`.
fn shell_quartet(shells: &[Shell], ab: &ShellPairData, cd: &ShellPairData, s: &mut EriScratch) -> (Vec<f64>, [usize; 4]) {
    let sh = [&shells[ab.sa], &shells[ab.sb], &shells[cd.sa], &shells[cd.sb]];
    let mut data = shell_quartet_cart(ab, cd, s);
    let mut dims = sh.map(|x| x.n_cartesian());
    for (axis, shell) in sh.iter().enumerate() {
        if let Some(t) = shell.spherical_transform() {
            data = transform_axis(&data, &mut dims, axis, t);
        }
    }
    (data, dims)
}

/// Schwarz factors Q_ab = sqrt(max |(μν|μν)|) over the normalized *Cartesian*
/// components μ ∈ a, ν ∈ b of every shell pair a >= b, indexed by `pair_index(a, b)`
/// over shells. theory.typ eq:schwarz
pub fn cartesian_schwarz_factors(basis: &Basis) -> Vec<f64> {
    let shells = basis.shells();
    let mut pairs = Vec::new();
    for a in 0..shells.len() {
        for b in 0..=a {
            pairs.push((a, b));
        }
    }
    pairs
        .par_iter()
        .map_init(EriScratch::default, |s, &(a, b)| {
            let ab = ShellPairData::new(shells, a, b);
            let data = shell_quartet_cart(&ab, &ab, s);
            let (na, nb) = (shells[a].n_cartesian(), shells[b].n_cartesian());
            let n = na * nb;
            let mut m: f64 = 0.0;
            for i in 0..n {
                m = m.max(data[i * n + i].abs());
            }
            m.sqrt()
        })
        .collect()
}

/// Two-electron integrals (μν|λσ) stored once per 8-fold permutation class.
/// theory.typ eq:eri-sym
#[derive(Debug, Clone)]
pub struct EriTensor {
    nao: usize,
    data: Vec<f64>,
}

impl EriTensor {
    /// Computes all unique ERIs, skipping shell quartets whose Schwarz bound
    /// (theory.typ eq:schwarz) is below `schwarz_threshold`. Parallelized with rayon.
    pub fn compute(basis: &Basis, schwarz_threshold: f64) -> Self {
        let shells = basis.shells();
        let nao = basis.nao();
        let npair = nao * (nao + 1) / 2;
        let mut pairs = Vec::new();
        for a in 0..shells.len() {
            for b in 0..=a {
                pairs.push((a, b));
            }
        }
        let pdata: Vec<ShellPairData> =
            pairs.par_iter().map(|&(a, b)| ShellPairData::new(shells, a, b)).collect();
        // Schwarz factors Q_ab = sqrt(max |(μν|μν)|) per shell pair.
        let q: Vec<f64> = pdata
            .par_iter()
            .map_init(EriScratch::default, |s, ab| {
                let (data, d) = shell_quartet(shells, ab, ab, s);
                let mut m: f64 = 0.0;
                for i in 0..d[0] {
                    for j in 0..d[1] {
                        let idx = ((i * d[1] + j) * d[2] + i) * d[3] + j;
                        m = m.max(data[idx].abs());
                    }
                }
                m.sqrt()
            })
            .collect();
        let chunks: Vec<Vec<(usize, f64)>> = (0..pdata.len())
            .into_par_iter()
            .map_init(EriScratch::default, |s, ij| {
                let ab = &pdata[ij];
                let mut out = Vec::new();
                for kl in 0..=ij {
                    if q[ij] * q[kl] < schwarz_threshold {
                        continue;
                    }
                    let cd = &pdata[kl];
                    let (data, d) = shell_quartet(shells, ab, cd, s);
                    let off = [ab.sa, ab.sb, cd.sa, cd.sb].map(|x| shells[x].ao_offset);
                    for i in 0..d[0] {
                        let mu = off[0] + i;
                        for j in 0..d[1] {
                            let nu = off[1] + j;
                            if nu > mu {
                                continue;
                            }
                            let mn = pair_index(mu, nu);
                            for k in 0..d[2] {
                                let la = off[2] + k;
                                for l in 0..d[3] {
                                    let si = off[3] + l;
                                    if si > la {
                                        continue;
                                    }
                                    let v = data[((i * d[1] + j) * d[2] + k) * d[3] + l];
                                    out.push((pair_index(mn, pair_index(la, si)), v));
                                }
                            }
                        }
                    }
                }
                out
            })
            .collect();
        let mut data = vec![0.0; npair * (npair + 1) / 2];
        for chunk in chunks {
            for (idx, v) in chunk {
                data[idx] = v;
            }
        }
        EriTensor { nao, data }
    }

    /// Number of AOs.
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// (μν|λσ).
    pub fn get(&self, mu: usize, nu: usize, la: usize, si: usize) -> f64 {
        self.data[pair_index(pair_index(mu, nu), pair_index(la, si))]
    }

    /// Coulomb matrix J and, if requested, exchange matrix K for a symmetric
    /// density P. theory.typ eq:jk
    pub fn coulomb_exchange(&self, p: &Matrix, want_k: bool) -> (Matrix, Option<Matrix>) {
        let n = self.nao;
        let mut j = Matrix::zeros(n, n);
        let mut k = Matrix::zeros(n, n);
        let mut idx = 0;
        for i in 0..n {
            for jj in 0..=i {
                let ij = pair_index(i, jj);
                for kk in 0..=i {
                    let lmax = if kk == i { jj } else { kk };
                    for l in 0..=lmax {
                        let v = self.data[idx];
                        idx += 1;
                        if v == 0.0 {
                            continue;
                        }
                        let kl = pair_index(kk, l);
                        // Weight so that summing all 8 permutations counts each distinct one once.
                        let mut f = v;
                        if i == jj {
                            f *= 0.5;
                        }
                        if kk == l {
                            f *= 0.5;
                        }
                        if ij == kl {
                            f *= 0.5;
                        }
                        let jkl = 2.0 * f * p[(kk, l)];
                        let jij = 2.0 * f * p[(i, jj)];
                        j[(i, jj)] += jkl;
                        j[(jj, i)] += jkl;
                        j[(kk, l)] += jij;
                        j[(l, kk)] += jij;
                        if want_k {
                            // K_ac += P_bd for each permutation (ab|cd).
                            k[(i, kk)] += f * p[(jj, l)];
                            k[(jj, kk)] += f * p[(i, l)];
                            k[(i, l)] += f * p[(jj, kk)];
                            k[(jj, l)] += f * p[(i, kk)];
                            k[(kk, i)] += f * p[(l, jj)];
                            k[(l, i)] += f * p[(kk, jj)];
                            k[(kk, jj)] += f * p[(l, i)];
                            k[(l, jj)] += f * p[(kk, i)];
                        }
                    }
                }
            }
        }
        debug_assert_eq!(idx, self.data.len());
        (j, if want_k { Some(k) } else { None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisOptions;
    use crate::basis::BasisSetDefinition;
    use crate::hermite::{Primitive, eri_primitive};
    use crate::molecule::Unit;

    fn water() -> Molecule {
        Molecule::from_symbols(
            &["O", "H", "H"],
            &[[0.0, 0.1, 0.1173], [0.0, 0.7572, -0.4692], [0.2, -0.7572, -0.4692]],
            Unit::Angstrom,
            0,
        )
        .unwrap()
    }

    #[test]
    fn normalized_diagonal() {
        for cart in [false, true] {
            let b = Basis::new(&water(), &BasisSetDefinition::builtin("6-31g**").unwrap(), BasisOptions { cartesian_d: cart })
                .unwrap();
            let s = overlap(&b);
            for i in 0..b.nao() {
                assert!((s[(i, i)] - 1.0).abs() < 1e-13, "S[{i},{i}] = {}", s[(i, i)]);
            }
        }
    }

    #[test]
    fn eri_symmetry_and_primitive_sum() {
        // Small distorted molecule with d functions; compare contracted ERIs with
        // a direct sum of primitive ERIs and check permutational symmetry.
        let mol = Molecule::from_symbols(&["C", "H"], &[[0.0, 0.1, -0.2], [0.3, 0.2, 1.9]], Unit::Bohr, 0).unwrap();
        let b = Basis::new(&mol, &BasisSetDefinition::builtin("6-31g*").unwrap(), BasisOptions { cartesian_d: true })
            .unwrap();
        let eri = EriTensor::compute(&b, 0.0);
        // Expand each Cartesian AO into primitives.
        let mut aos: Vec<Vec<(f64, Primitive)>> = Vec::new();
        for sh in b.shells() {
            for lmn in cartesian_components(sh.l) {
                let c = cartesian_norm(lmn);
                aos.push(
                    sh.exponents
                        .iter()
                        .zip(&sh.coefficients)
                        .map(|(&e, &d)| (c * d, Primitive { exponent: e, center: sh.center, lmn }))
                        .collect(),
                );
            }
        }
        let n = b.nao();
        let direct = |i: usize, j: usize, k: usize, l: usize| {
            let mut s = 0.0;
            for (ci, gi) in &aos[i] {
                for (cj, gj) in &aos[j] {
                    for (ck, gk) in &aos[k] {
                        for (cl, gl) in &aos[l] {
                            s += ci * cj * ck * cl * eri_primitive(gi, gj, gk, gl);
                        }
                    }
                }
            }
            s
        };
        let picks = [(0, 0, 0, 0), (n - 1, 3, n - 2, 7), (10, 12, 5, 14), (2, 11, 13, 9), (14, 14, 12, 12)];
        for &(i, j, k, l) in &picks {
            let d = direct(i, j, k, l);
            assert!((eri.get(i, j, k, l) - d).abs() < 1e-12, "({i}{j}|{k}{l}) {} {d}", eri.get(i, j, k, l));
            for (a, bb, c, dd) in [(j, i, k, l), (i, j, l, k), (k, l, i, j), (l, k, j, i)] {
                assert!((direct(a, bb, c, dd) - d).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn jk_matches_dense_contraction() {
        let mol = water();
        let b = Basis::from_name(&mol, "6-31g*").unwrap();
        let eri = EriTensor::compute(&b, 0.0);
        let n = b.nao();
        let p = Matrix::from_fn(n, n, |i, j| ((i * 7 + j * 7 + 3 * i * j) % 11) as f64 * 0.01 + if i == j { 0.5 } else { 0.0 });
        let (j, k) = eri.coulomb_exchange(&p, true);
        let k = k.unwrap();
        for mu in 0..n {
            for nu in 0..n {
                let mut jr = 0.0;
                let mut kr = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        jr += p[(la, si)] * eri.get(mu, nu, la, si);
                        kr += p[(la, si)] * eri.get(mu, la, nu, si);
                    }
                }
                assert!((j[(mu, nu)] - jr).abs() < 1e-12);
                assert!((k[(mu, nu)] - kr).abs() < 1e-12);
            }
        }
    }
}

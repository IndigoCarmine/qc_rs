//! Atom-centred molecular integration grids (theory.typ section 8.4):
//! Gauss–Chebyshev radial grid with Becke's mapping (eq:radial), Gauss–Legendre ×
//! uniform-φ product angular grid (eq:angular, eq:gauss-legendre) and Becke's
//! fuzzy-cell partitioning with atomic size adjustment (eq:becke-mu … eq:becke-w).

use crate::error::{QcError, Result};
use crate::molecule::{BOHR_IN_ANGSTROM, Molecule};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::ops::Range;

/// Bragg–Slater radii H–Ar in Ångström (theory.typ section 8.4 table; noble gases
/// have conventional values).
const BRAGG_SLATER_ANGSTROM: [f64; 18] = [
    0.35, 0.35, 1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.45, 1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 0.90,
];

/// Points whose total weight w_rad w_ang w_Becke is below this are dropped. With
/// |f| ≲ 10⁴ and |v| χ χ ≲ 10⁴ anywhere, the dropped contributions are < 10⁻¹⁶ per
/// point, far below any measurable effect.
pub const WEIGHT_CUTOFF: f64 = 1e-20;

/// Bragg–Slater radius R_BS of element `z` in Bohr (theory.typ eq:bohr conversion).
pub fn bragg_slater_radius(z: u32) -> Option<f64> {
    if (1..=18).contains(&z) { Some(BRAGG_SLATER_ANGSTROM[z as usize - 1] / BOHR_IN_ANGSTROM) } else { None }
}

/// Grid size per atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSettings {
    /// Number of radial points n_r.
    pub n_radial: usize,
    /// Number of Gauss–Legendre points in cos θ (n_φ = 2 n_θ).
    pub n_theta: usize,
}

impl Default for GridSettings {
    /// n_r = 75, n_θ = 18 (theory.typ section 8.4 "既定の格子").
    fn default() -> Self {
        GridSettings { n_radial: 75, n_theta: 18 }
    }
}

impl GridSettings {
    /// The fine grid used for precise comparisons: n_r = 150, n_θ = 30.
    pub fn fine() -> Self {
        GridSettings { n_radial: 150, n_theta: 30 }
    }

    /// Number of angular points n_θ × 2 n_θ.
    pub fn n_angular(&self) -> usize {
        2 * self.n_theta * self.n_theta
    }
}

/// Radial nodes r_i and weights w_i^rad (including r²) for i = 1..n_r. theory.typ eq:radial
pub fn radial_grid(n: usize, r_m: f64) -> (Vec<f64>, Vec<f64>) {
    let mut r = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let h = PI / (n as f64 + 1.0);
    for i in 1..=n {
        let t = i as f64 * h;
        let x = t.cos();
        let ri = r_m * (1.0 + x) / (1.0 - x);
        r.push(ri);
        w.push(h * t.sin() * 2.0 * r_m / ((1.0 - x) * (1.0 - x)) * ri * ri);
    }
    (r, w)
}

/// Gauss–Legendre nodes u_j (descending) and weights ω_j on [-1, 1]. theory.typ eq:gauss-legendre
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut u = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    // P_n(x) and P_{n-1}(x) by the three-term recurrence.
    let legendre = |x: f64| {
        let (mut p0, mut p1) = (1.0, x);
        if n == 0 {
            return (1.0, 0.0);
        }
        for k in 1..n {
            let p2 = ((2 * k + 1) as f64 * x * p1 - k as f64 * p0) / (k + 1) as f64;
            p0 = p1;
            p1 = p2;
        }
        (p1, p0)
    };
    for j in 1..=n {
        let mut x = (PI * (j as f64 - 0.25) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (pn, pn1) = legendre(x);
            let dp = n as f64 * (x * pn - pn1) / (x * x - 1.0);
            let dx = pn / dp;
            x -= dx;
            if dx.abs() < 1e-16 {
                break;
            }
        }
        let (pn, pn1) = legendre(x);
        let dp = n as f64 * (x * pn - pn1) / (x * x - 1.0);
        u.push(x);
        w.push(2.0 / ((1.0 - x * x) * dp * dp));
    }
    (u, w)
}

/// Product angular grid: unit vectors n̂_jk and weights (summing to 4π). theory.typ eq:angular
pub fn angular_grid(n_theta: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
    let (u, om) = gauss_legendre(n_theta);
    let n_phi = 2 * n_theta;
    let mut pts = Vec::with_capacity(n_theta * n_phi);
    let mut wts = Vec::with_capacity(n_theta * n_phi);
    for (uj, wj) in u.iter().zip(&om) {
        let st = (1.0 - uj * uj).sqrt();
        for k in 0..n_phi {
            let phi = 2.0 * PI * (k as f64 + 0.5) / n_phi as f64;
            pts.push([st * phi.cos(), st * phi.sin(), *uj]);
            wts.push(wj * 2.0 * PI / n_phi as f64);
        }
    }
    (pts, wts)
}

/// s(ν) = ½[1 - p(p(p(ν)))], p(t) = 3t/2 - t³/2. theory.typ eq:becke-s
#[inline]
fn becke_s(nu: f64) -> f64 {
    let p = |t: f64| 1.5 * t - 0.5 * t * t * t;
    0.5 * (1.0 - p(p(p(nu))))
}

/// Pairwise data for Becke partitioning of a molecule.
struct BeckePartition {
    centers: Vec<[f64; 3]>,
    /// 1/R_AB (row-major, n×n).
    inv_r: Vec<f64>,
    /// a_AB (row-major, n×n). theory.typ eq:becke-a
    a: Vec<f64>,
}

impl BeckePartition {
    fn new(centers: Vec<[f64; 3]>, radii: &[f64]) -> Self {
        let n = centers.len();
        let mut inv_r = vec![0.0; n * n];
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                inv_r[i * n + j] = 1.0 / dist(&centers[i], &centers[j]);
                // theory.typ eq:becke-a
                let chi = radii[i] / radii[j];
                let u = (chi - 1.0) / (chi + 1.0);
                a[i * n + j] = (u / (u * u - 1.0)).clamp(-0.5, 0.5);
            }
        }
        BeckePartition { centers, inv_r, a }
    }

    /// w_Becke of atom `owner` at point `r`. theory.typ eq:becke-mu, eq:becke-w
    fn weight(&self, owner: usize, r: &[f64; 3], dists: &mut [f64], cell: &mut [f64]) -> f64 {
        let n = self.centers.len();
        if n == 1 {
            return 1.0;
        }
        for (d, c) in dists.iter_mut().zip(&self.centers) {
            *d = dist(r, c);
        }
        cell.fill(1.0);
        for i in 0..n {
            for j in 0..i {
                let mu = (dists[i] - dists[j]) * self.inv_r[i * n + j];
                let nu_ij = mu + self.a[i * n + j] * (1.0 - mu * mu);
                // μ_ji = -μ_ij and a_ji = -a_ij, so ν_ji = -ν_ij.
                cell[i] *= becke_s(nu_ij);
                cell[j] *= becke_s(-nu_ij);
            }
        }
        let tot: f64 = cell.iter().sum();
        if tot == 0.0 { 0.0 } else { cell[owner] / tot }
    }
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// One radial shell of one atom: all its points lie on a sphere of radius `radius`
/// around `center`, which lets basis-function screening work per block.
#[derive(Debug, Clone, PartialEq)]
pub struct GridBlock {
    /// Atom that owns the shell.
    pub atom: usize,
    /// Centre of the sphere (the atom position, Bohr).
    pub center: [f64; 3],
    /// Radius r_i of the sphere (Bohr).
    pub radius: f64,
    /// Indices of the block's points in [`MolecularGrid::points`].
    pub range: Range<usize>,
}

/// (atom, r_i, points, weights) of one radial shell during grid construction.
type BuiltShell = (usize, f64, Vec<[f64; 3]>, Vec<f64>);

/// The molecular grid: points, weights w_g = w_rad w_ang w_Becke and blocks.
#[derive(Debug, Clone)]
pub struct MolecularGrid {
    /// Grid points (Bohr).
    pub points: Vec<[f64; 3]>,
    /// Weights w_g.
    pub weights: Vec<f64>,
    /// Blocks (radial shells), covering `points` in order.
    pub blocks: Vec<GridBlock>,
    /// The settings used.
    pub settings: GridSettings,
}

impl MolecularGrid {
    /// Builds the grid of `mol` (theory.typ section 8.4). Points with weight below
    /// [`WEIGHT_CUTOFF`] are omitted.
    pub fn new(mol: &Molecule, settings: GridSettings) -> Result<Self> {
        if settings.n_radial == 0 || settings.n_theta == 0 {
            return Err(QcError::InvalidInput("grid needs at least one radial and one angular point".into()));
        }
        let atoms = mol.atoms();
        let radii: Vec<f64> = atoms
            .iter()
            .map(|a| bragg_slater_radius(a.z).ok_or_else(|| QcError::UnknownElement(format!("Z = {}", a.z))))
            .collect::<Result<_>>()?;
        let partition = BeckePartition::new(atoms.iter().map(|a| a.position).collect(), &radii);
        let (ang_pts, ang_w) = angular_grid(settings.n_theta);

        // Every (atom, radial shell) is built independently.
        let shells: Vec<(usize, f64, f64)> = atoms
            .iter()
            .enumerate()
            .flat_map(|(ia, a)| {
                // R_m = R_BS / 2, except R_m = R_BS for hydrogen (theory.typ eq:radial).
                let r_m = if a.z == 1 { radii[ia] } else { 0.5 * radii[ia] };
                let (r, w) = radial_grid(settings.n_radial, r_m);
                r.into_iter().zip(w).map(move |(ri, wi)| (ia, ri, wi))
            })
            .collect();
        let built: Vec<BuiltShell> = shells
            .par_iter()
            .map(|&(ia, ri, wr)| {
                let c = atoms[ia].position;
                let mut dists = vec![0.0; atoms.len()];
                let mut cell = vec![0.0; atoms.len()];
                let mut pts = Vec::with_capacity(ang_pts.len());
                let mut wts = Vec::with_capacity(ang_pts.len());
                for (n, wa) in ang_pts.iter().zip(&ang_w) {
                    let p = [c[0] + ri * n[0], c[1] + ri * n[1], c[2] + ri * n[2]];
                    let w = wr * wa * partition.weight(ia, &p, &mut dists, &mut cell);
                    if w >= WEIGHT_CUTOFF {
                        pts.push(p);
                        wts.push(w);
                    }
                }
                (ia, ri, pts, wts)
            })
            .collect();

        let total: usize = built.iter().map(|b| b.2.len()).sum();
        let mut points = Vec::with_capacity(total);
        let mut weights = Vec::with_capacity(total);
        let mut blocks = Vec::new();
        for (ia, ri, pts, wts) in built {
            if pts.is_empty() {
                continue;
            }
            let start = points.len();
            points.extend(pts);
            weights.extend(wts);
            blocks.push(GridBlock { atom: ia, center: atoms[ia].position, radius: ri, range: start..points.len() });
        }
        Ok(MolecularGrid { points, weights, blocks, settings })
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the grid has no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Σ_g w_g f(r_g).
    pub fn integrate(&self, f: impl Fn(&[f64; 3]) -> f64 + Sync) -> f64 {
        let parts: Vec<f64> = self
            .blocks
            .par_iter()
            .map(|b| b.range.clone().map(|g| self.weights[g] * f(&self.points[g])).sum())
            .collect();
        parts.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Unit;

    /// ∫ x^a y^b z^c dΩ over the unit sphere.
    fn sphere_monomial(a: usize, b: usize, c: usize) -> f64 {
        if a % 2 == 1 || b % 2 == 1 || c % 2 == 1 {
            return 0.0;
        }
        // 2 Γ((a+1)/2) Γ((b+1)/2) Γ((c+1)/2) / Γ((a+b+c+3)/2); with half-integer
        // arguments Γ(k + 1/2) = (2k-1)!! √π / 2^k.
        let g_half = |k: usize| crate::basis::odd_double_factorial(k) * PI.sqrt() / 2f64.powi(k as i32);
        let (ka, kb, kc) = (a / 2, b / 2, c / 2);
        let s = ka + kb + kc + 1; // (a+b+c+3)/2 = s + 1/2
        2.0 * g_half(ka) * g_half(kb) * g_half(kc) / g_half(s)
    }

    fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
    }

    #[test]
    fn gauss_legendre_is_exact() {
        for n in [1, 2, 5, 18, 30] {
            let (u, w) = gauss_legendre(n);
            assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-14);
            for k in 0..2 * n {
                let num: f64 = u.iter().zip(&w).map(|(x, wi)| wi * x.powi(k as i32)).sum();
                let exact = if k % 2 == 1 { 0.0 } else { 2.0 / (k as f64 + 1.0) };
                assert!((num - exact).abs() < 1e-14, "n={n} k={k}: {num} vs {exact}");
            }
        }
    }

    #[test]
    fn angular_grid_is_exact_to_degree_2n_minus_1() {
        for n in [4, 18, 30] {
            let (p, w) = angular_grid(n);
            assert!((w.iter().sum::<f64>() - 4.0 * PI).abs() < 1e-13);
            let deg = 2 * n - 1;
            for a in 0..=deg {
                for b in 0..=deg - a {
                    for c in 0..=deg - a - b {
                        let num: f64 = p
                            .iter()
                            .zip(&w)
                            .map(|(v, wi)| wi * v[0].powi(a as i32) * v[1].powi(b as i32) * v[2].powi(c as i32))
                            .sum();
                        let exact = sphere_monomial(a, b, c);
                        assert!((num - exact).abs() < 1e-13, "n={n} ({a},{b},{c}): {num} vs {exact}");
                    }
                }
            }
        }
        // ... and not beyond: z^(2n) is not integrated exactly.
        let n = 4;
        let (p, w) = angular_grid(n);
        let num: f64 = p.iter().zip(&w).map(|(v, wi)| wi * v[2].powi(2 * n as i32)).sum();
        assert!((num - sphere_monomial(0, 0, 2 * n)).abs() > 1e-6);
    }

    #[test]
    fn single_atom_radial_integrals() {
        let mol = Molecule::from_symbols(&["O"], &[[0.1, -0.2, 0.3]], Unit::Bohr, 0).unwrap();
        let c = mol.atoms()[0].position;
        // Achieved relative errors (O, R_m = 0.567 a0): default grid ≤ 9e-8 (worst for
        // the very tight α = 5000 Gaussian, which few radial points resolve), fine
        // grid ≤ 1.5e-9 (again α = 5000), ≤ 3.1e-12 for everything else.
        for (settings, tol, tol_tight) in [(GridSettings::default(), 1e-8, 2e-7), (GridSettings::fine(), 1e-11, 3e-9)] {
            let grid = MolecularGrid::new(&mol, settings).unwrap();
            for alpha in [0.1, 1.0, 30.0, 5000.0] {
                let num = grid.integrate(|p| (-alpha * dist2(p, &c)).exp());
                let exact = (PI / alpha).powf(1.5);
                let rel = (num / exact - 1.0).abs();
                println!("{settings:?} Gaussian alpha={alpha}: rel. error {rel:.2e}");
                let t = if alpha > 1000.0 { tol_tight } else { tol };
                assert!(rel < t, "{settings:?} Gaussian alpha={alpha}: {num} vs {exact}");
            }
            for zeta in [0.5, 2.0, 15.0] {
                let num = grid.integrate(|p| (-zeta * dist2(p, &c).sqrt()).exp());
                let exact = 8.0 * PI / zeta.powi(3);
                let rel = (num / exact - 1.0).abs();
                println!("{settings:?} Slater zeta={zeta}: rel. error {rel:.2e}");
                assert!(rel < tol, "{settings:?} Slater zeta={zeta}: {num} vs {exact}");
            }
        }
    }

    #[test]
    fn becke_weights_partition_unity() {
        let mol = Molecule::from_symbols(
            &["O", "H", "C"],
            &[[0.0, 0.0, 0.0], [1.8, 0.2, 0.0], [-0.5, 2.4, 0.7]],
            Unit::Bohr,
            0,
        )
        .unwrap();
        let radii: Vec<f64> = mol.atoms().iter().map(|a| bragg_slater_radius(a.z).unwrap()).collect();
        let part = BeckePartition::new(mol.atoms().iter().map(|a| a.position).collect(), &radii);
        let mut d = vec![0.0; 3];
        let mut c = vec![0.0; 3];
        for p in [[0.3, 0.1, -0.2], [1.0, 1.0, 1.0], [5.0, -3.0, 2.0], [0.9, 0.1, 0.0]] {
            let s: f64 = (0..3).map(|a| part.weight(a, &p, &mut d, &mut c)).sum();
            assert!((s - 1.0).abs() < 1e-14);
        }
        // At a nucleus the owner gets everything.
        assert!((part.weight(1, &[1.8, 0.2, 0.0], &mut d, &mut c) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn multicentre_gaussians_through_becke_partition() {
        let mol = Molecule::from_symbols(
            &["O", "H", "H"],
            &[[0.0, 0.0, 0.2217], [0.0, 1.4309, -0.8867], [0.0, -1.4309, -0.8867]],
            Unit::Bohr,
            0,
        )
        .unwrap();
        // Gaussians on the atoms, plus one off-centre (bond midpoint).
        let gauss: Vec<([f64; 3], f64, f64)> = vec![
            (mol.atoms()[0].position, 3.0, 1.0),
            (mol.atoms()[0].position, 0.25, 0.7),
            (mol.atoms()[1].position, 0.8, -0.4),
            (mol.atoms()[2].position, 1.5, 2.0),
            ([0.0, 0.7, -0.33], 0.6, 0.5),
        ];
        let exact: f64 = gauss.iter().map(|(_, a, c)| c * (PI / a).powf(1.5)).sum();
        // Achieved: default 1.9e-7, fine 1.4e-10 (the Becke cell boundaries make the
        // integrand less smooth than on a single atom).
        for (settings, tol) in [(GridSettings::default(), 1e-6), (GridSettings::fine(), 1e-9)] {
            let grid = MolecularGrid::new(&mol, settings).unwrap();
            let num = grid.integrate(|p| gauss.iter().map(|(c, a, k)| k * (-a * dist2(p, c)).exp()).sum());
            println!("{settings:?} multicentre: rel. error {:.2e}", (num / exact - 1.0).abs());
            assert!((num - exact).abs() < tol * exact.abs(), "{settings:?}: {num} vs {exact}");
            // z-moment of the off-centre Gaussian: ∫ z e^{-a|r-c|²} = c_z (π/a)^{3/2}.
            let (c, a, _) = gauss[4];
            let num = grid.integrate(|p| p[2] * (-a * dist2(p, &c)).exp());
            let ex = c[2] * (PI / a).powf(1.5);
            println!("{settings:?} off-centre moment: rel. error {:.2e}", (num / ex - 1.0).abs());
            assert!((num - ex).abs() < tol * 10.0 * ex.abs(), "{settings:?} moment: {num} vs {ex}");
        }
    }

    #[test]
    fn three_heavy_atoms_with_different_radii() {
        let mol = Molecule::from_symbols(
            &["Cl", "C", "H"],
            &[[0.0, 0.0, 0.0], [3.3, 0.0, 0.0], [4.0, 1.9, 0.0]],
            Unit::Bohr,
            0,
        )
        .unwrap();
        let grid = MolecularGrid::new(&mol, GridSettings::fine()).unwrap();
        let gauss: Vec<([f64; 3], f64)> = mol.atoms().iter().map(|a| (a.position, 0.9)).collect();
        let exact = 3.0 * (PI / 0.9f64).powf(1.5);
        let num = grid.integrate(|p| gauss.iter().map(|(c, a)| (-a * dist2(p, c)).exp()).sum());
        println!("Cl-C-H: rel. error {:.2e}", (num / exact - 1.0).abs());
        assert!((num / exact - 1.0).abs() < 1e-8, "{num} vs {exact}");
    }
}

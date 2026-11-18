//! Small dense linear algebra: row-major matrix, symmetric eigensolver, linear solve.

use crate::error::{QcError, Result};
use std::ops::{Add, Index, IndexMut, Mul, Sub};

/// Dense row-major matrix of `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Zero matrix of the given shape.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    /// Builds a matrix from a function of (row, column).
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Matrix { rows, cols, data }
    }

    /// Builds a matrix from row-major data.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(QcError::InvalidInput("matrix data has wrong length".into()));
        }
        Ok(Matrix { rows, cols, data })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major data.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutable row-major data.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Transpose.
    pub fn transpose(&self) -> Matrix {
        Matrix::from_fn(self.cols, self.rows, |i, j| self[(j, i)])
    }

    /// Matrix product `self * other`.
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "matmul shape mismatch");
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            let orow = &mut out.data[i * other.cols..(i + 1) * other.cols];
            for k in 0..self.cols {
                let a = self.data[i * self.cols + k];
                if a == 0.0 {
                    continue;
                }
                let brow = &other.data[k * other.cols..(k + 1) * other.cols];
                for (o, b) in orow.iter_mut().zip(brow) {
                    *o += a * b;
                }
            }
        }
        out
    }

    /// Sum of element-wise products, i.e. Tr(Aᵀ B) (= Tr(A B) for symmetric A).
    pub fn dot(&self, other: &Matrix) -> f64 {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "dot shape mismatch");
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }

    /// Largest absolute element.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0, |m, x| m.max(x.abs()))
    }

    /// Multiplies every element by `s`.
    pub fn scaled(&self, s: f64) -> Matrix {
        Matrix { rows: self.rows, cols: self.cols, data: self.data.iter().map(|x| x * s).collect() }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

impl Add for &Matrix {
    type Output = Matrix;
    fn add(self, o: &Matrix) -> Matrix {
        assert_eq!((self.rows, self.cols), (o.rows, o.cols), "add shape mismatch");
        Matrix { rows: self.rows, cols: self.cols, data: self.data.iter().zip(&o.data).map(|(a, b)| a + b).collect() }
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, o: &Matrix) -> Matrix {
        assert_eq!((self.rows, self.cols), (o.rows, o.cols), "sub shape mismatch");
        Matrix { rows: self.rows, cols: self.cols, data: self.data.iter().zip(&o.data).map(|(a, b)| a - b).collect() }
    }
}

impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, o: &Matrix) -> Matrix {
        self.matmul(o)
    }
}

/// Eigen-decomposition of a symmetric matrix: ascending eigenvalues, eigenvectors as columns.
#[derive(Debug, Clone)]
pub struct SymmetricEigen {
    /// Eigenvalues in ascending order.
    pub values: Vec<f64>,
    /// Eigenvectors stored as columns, in the order of `values`.
    pub vectors: Matrix,
}

/// Rows handed to one rayon task in the row-parallel loops of `symmetric_eigen`.
const EIGEN_ROWS_PER_TASK: usize = 16;

/// Implicit QL iterations allowed per eigenvalue before giving up.
const QL_MAX_ITER: usize = 50;

/// QL rotations (in units of n) collected before they are applied to the eigenvectors
/// in one pass over the rows.
const QL_ROTATIONS_PER_ROW_PASS: usize = 16;

/// Diagonalizes a symmetric matrix: Householder tridiagonalization followed by the
/// implicitly shifted QL algorithm. theory.typ eq:householder eq:ql-shift UNCHECKED(Golub2013)
///
/// Every loop that touches an n × n matrix runs along rows (row-major storage) and is
/// split over rows only, so the result does not depend on the number of threads.
pub fn symmetric_eigen(a: &Matrix) -> Result<SymmetricEigen> {
    use rayon::prelude::*;

    if a.rows != a.cols {
        return Err(QcError::InvalidInput("eigen: matrix is not square".into()));
    }
    let n = a.rows;
    // Work on the exactly symmetric part.
    let mut m = Matrix::from_fn(n, n, |i, j| 0.5 * (a[(i, j)] + a[(j, i)]));
    if !m.data.iter().all(|x| x.is_finite()) {
        return Err(QcError::InvalidInput("eigen: non-finite matrix".into()));
    }

    // Step 1: T = Qᵀ A Q with Q = P_0 P_1 ⋯ P_{n-3} (theory.typ eq:householder).
    // d is the diagonal of T, e[i] couples i and i + 1.
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    let mut reflectors: Vec<(usize, Vec<f64>, f64)> = Vec::new();
    for k in 0..n.saturating_sub(2) {
        d[k] = m[(k, k)];
        // x = A[k+1.., k], read from row k (A is symmetric).
        let x = &m.data[k * n + k + 1..(k + 1) * n];
        let scale = x[1..].iter().fold(0.0f64, |s, v| s.max(v.abs()));
        if scale == 0.0 {
            // Column already in tridiagonal form: P_k = I.
            e[k] = x[0];
            continue;
        }
        let scale = scale.max(x[0].abs());
        let tail: f64 = x[1..].iter().map(|v| (v / scale).powi(2)).sum();
        if tail == 0.0 {
            // Column already in tridiagonal form: P_k = I.
            e[k] = x[0];
            continue;
        }
        let mut v: Vec<f64> = x.iter().map(|xi| xi / scale).collect();
        let norm = (v[0] * v[0] + tail).sqrt();
        let alpha = if v[0] >= 0.0 { -norm } else { norm };
        v[0] -= alpha;
        let beta = 2.0 / (v[0] * v[0] + tail);
        e[k] = alpha * scale;
        // A22 <- P A22 P with p = β A22 v, w = p − (β pᵀv / 2) v: A22 −= v wᵀ + w vᵀ.
        let off = k + 1;
        let p: Vec<f64> = m.data[off * n..]
            .par_chunks(n)
            .with_min_len(EIGEN_ROWS_PER_TASK)
            .map(|row| beta * row[off..].iter().zip(&v).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let kk = 0.5 * beta * p.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
        let w: Vec<f64> = p.iter().zip(&v).map(|(pi, vi)| pi - kk * vi).collect();
        m.data[off * n..].par_chunks_mut(n).with_min_len(EIGEN_ROWS_PER_TASK).enumerate().for_each(|(i, row)| {
            let (vi, wi) = (v[i], w[i]);
            for ((r, vj), wj) in row[off..].iter_mut().zip(&v).zip(&w) {
                *r -= vi * wj + wi * vj;
            }
        });
        reflectors.push((k, v, beta));
    }
    if n >= 2 {
        d[n - 2] = m[(n - 2, n - 2)];
        e[n - 2] = m[(n - 2, n - 1)];
    }
    if n >= 1 {
        d[n - 1] = m[(n - 1, n - 1)];
    }

    // Q = P_0 P_1 ⋯, accumulated row by row: Q[r, k+1..] −= β (Q[r, k+1..]·v) vᵀ for
    // k = 0, 1, … in turn. Rows are independent, so one parallel pass applies all P_k.
    let mut z = Matrix::identity(n);
    z.data.par_chunks_mut(n.max(1)).with_min_len(EIGEN_ROWS_PER_TASK).for_each(|row| {
        for (k, v, beta) in &reflectors {
            let off = k + 1;
            let t = beta * row[off..].iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
            if t != 0.0 {
                for (r, vj) in row[off..].iter_mut().zip(v) {
                    *r -= t * vj;
                }
            }
        }
    });
    drop(reflectors);

    // Step 2: implicitly shifted QL on (d, e) (theory.typ eq:ql-shift). The plane
    // rotations are recorded in order and applied to the columns of Z in batches, one
    // row of Z at a time (each row sees the same sequence of rotations as if they were
    // applied one by one), so that Z ends up holding Q times the eigenvectors of T.
    let apply = |z: &mut Matrix, rotations: &[(usize, f64, f64)]| {
        z.data.par_chunks_mut(n).with_min_len(EIGEN_ROWS_PER_TASK).for_each(|row| {
            for &(i, c, s) in rotations {
                let zi = row[i];
                let zi1 = row[i + 1];
                row[i + 1] = s * zi + c * zi1;
                row[i] = c * zi - s * zi1;
            }
        });
    };
    let batch = QL_ROTATIONS_PER_ROW_PASS * n;
    let mut rotations: Vec<(usize, f64, f64)> = Vec::with_capacity(batch + n);
    for l in 0..n {
        let mut iter = 0;
        loop {
            // Find the first negligible e[m] at or after l (e[n-1] = 0 always).
            let mut mm = l;
            while mm + 1 < n {
                let dd = d[mm].abs() + d[mm + 1].abs();
                if e[mm].abs() <= f64::EPSILON * dd {
                    break;
                }
                mm += 1;
            }
            if mm == l {
                break;
            }
            iter += 1;
            if iter > QL_MAX_ITER {
                return Err(QcError::EigenNotConverged);
            }
            // Shift: eigenvalue of the leading 2 × 2 block closer to d[l].
            let delta = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let root = delta.hypot(1.0);
            // sgn(0) = +1, also for delta = -0.0.
            let sgn_root = if delta >= 0.0 { root } else { -root };
            let mut g = d[mm] - d[l] + e[l] / (delta + sgn_root);
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut underflow = false;
            for i in (l..mm).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    // The rotation underflowed: T split at i + 1; restart this l.
                    d[i + 1] -= p;
                    e[mm] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                let t = (d[i] - g) * s + 2.0 * c * b;
                p = s * t;
                d[i + 1] = g + p;
                g = c * t - b;
                rotations.push((i, c, s));
            }
            if rotations.len() >= batch {
                apply(&mut z, &rotations);
                rotations.clear();
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[mm] = 0.0;
        }
    }

    if !rotations.is_empty() {
        apply(&mut z, &rotations);
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[i].total_cmp(&d[j]));
    let values = order.iter().map(|&i| d[i]).collect();
    let vectors = Matrix::from_fn(n, n, |r, c| z[(r, order[c])]);
    Ok(SymmetricEigen { values, vectors })
}

/// Solves `A x = b` by Gaussian elimination with partial pivoting.
pub fn solve(a: &Matrix, b: &[f64]) -> Result<Vec<f64>> {
    let n = a.rows;
    if a.cols != n || b.len() != n {
        return Err(QcError::InvalidInput("solve: dimension mismatch".into()));
    }
    let scale = a.max_abs();
    if scale == 0.0 || !scale.is_finite() {
        return Err(QcError::SingularMatrix);
    }
    let mut m = a.clone();
    let mut x = b.to_vec();
    for k in 0..n {
        let piv = (k..n).max_by(|&i, &j| m[(i, k)].abs().total_cmp(&m[(j, k)].abs())).unwrap();
        if m[(piv, k)].abs() <= 1e-14 * scale {
            return Err(QcError::SingularMatrix);
        }
        if piv != k {
            for j in 0..n {
                let tmp = m[(k, j)];
                m[(k, j)] = m[(piv, j)];
                m[(piv, j)] = tmp;
            }
            x.swap(k, piv);
        }
        for i in k + 1..n {
            let f = m[(i, k)] / m[(k, k)];
            if f == 0.0 {
                continue;
            }
            for j in k..n {
                m[(i, j)] -= f * m[(k, j)];
            }
            x[i] -= f * x[k];
        }
    }
    for k in (0..n).rev() {
        let mut s = x[k];
        for j in k + 1..n {
            s -= m[(k, j)] * x[j];
        }
        x[k] = s / m[(k, k)];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_matrix(n: usize) -> Matrix {
        // Deterministic pseudo-random symmetric matrix with a degenerate pair.
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let x = rnd();
                m[(i, j)] = x;
                m[(j, i)] = x;
            }
        }
        m
    }

    #[test]
    fn eigen_reconstructs_matrix() {
        for n in [1, 2, 5, 20] {
            let a = test_matrix(n);
            let e = symmetric_eigen(&a).unwrap();
            let v = &e.vectors;
            // Orthonormality
            let vtv = v.transpose().matmul(v);
            assert!((&vtv - &Matrix::identity(n)).max_abs() < 1e-12);
            // A V = V diag(w)
            let av = a.matmul(v);
            for i in 0..n {
                for j in 0..n {
                    assert!((av[(i, j)] - v[(i, j)] * e.values[j]).abs() < 1e-12);
                }
            }
            for w in e.values.windows(2) {
                assert!(w[0] <= w[1]);
            }
            // Trace preserved
            let tr: f64 = (0..n).map(|i| a[(i, i)]).sum();
            assert!((tr - e.values.iter().sum::<f64>()).abs() < 1e-12);
        }
    }

    #[test]
    fn eigen_degenerate() {
        // diag(1,1,2) rotated: eigenvalues 1,1,2
        let a = Matrix::from_vec(3, 3, vec![1.5, 0.5, 0.0, 0.5, 1.5, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let e = symmetric_eigen(&a).unwrap();
        assert!((e.values[0] - 1.0).abs() < 1e-14);
        assert!((e.values[1] - 1.0).abs() < 1e-14);
        assert!((e.values[2] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn eigen_large_relative_accuracy() {
        let n = 200;
        let a = test_matrix(n);
        let e = symmetric_eigen(&a).unwrap();
        let v = &e.vectors;
        let vtv = v.transpose().matmul(v);
        assert!((&vtv - &Matrix::identity(n)).max_abs() < 1e-12);
        // ‖A‖₂ = max |λ|.
        let norm = e.values.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let av = a.matmul(v);
        let mut resid = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                resid = resid.max((av[(i, j)] - v[(i, j)] * e.values[j]).abs());
            }
        }
        assert!(resid < 1e-12 * norm, "max |AV - VΛ| = {resid:e}, ‖A‖ = {norm}");
        for w in e.values.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn eigen_special_structures() {
        // Already diagonal (unsorted), zero matrix, and a zero-diagonal 2 × 2 block.
        let a = Matrix::from_vec(3, 3, vec![3.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0]).unwrap();
        let e = symmetric_eigen(&a).unwrap();
        assert_eq!(e.values, vec![-1.0, 2.0, 3.0]);
        assert_eq!(e.vectors[(1, 0)].abs(), 1.0);
        let e = symmetric_eigen(&Matrix::zeros(4, 4)).unwrap();
        assert_eq!(e.values, vec![0.0; 4]);
        assert!((&e.vectors.transpose().matmul(&e.vectors) - &Matrix::identity(4)).max_abs() < 1e-15);
        let a = Matrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let e = symmetric_eigen(&a).unwrap();
        assert!((e.values[0] + 1.0).abs() < 1e-15 && (e.values[1] - 1.0).abs() < 1e-15);
        assert!(symmetric_eigen(&Matrix::zeros(0, 0)).unwrap().values.is_empty());
    }

    #[test]
    fn solve_pivoting() {
        let a = Matrix::from_vec(3, 3, vec![0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
        let x_true = [1.0, -2.0, 0.5];
        let b: Vec<f64> = (0..3).map(|i| (0..3).map(|j| a[(i, j)] * x_true[j]).sum()).collect();
        let x = solve(&a, &b).unwrap();
        for i in 0..3 {
            assert!((x[i] - x_true[i]).abs() < 1e-14);
        }
        let s = Matrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert_eq!(solve(&s, &[1.0, 1.0]), Err(QcError::SingularMatrix));
    }
}

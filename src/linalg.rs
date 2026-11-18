//! Small dense linear algebra: row-major matrix, Jacobi symmetric eigensolver, linear solve.

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

/// Diagonalizes a symmetric matrix with the cyclic Jacobi method. theory.typ eq:jacobi UNCHECKED(Golub2013)
pub fn symmetric_eigen(a: &Matrix) -> Result<SymmetricEigen> {
    if a.rows != a.cols {
        return Err(QcError::InvalidInput("eigen: matrix is not square".into()));
    }
    let n = a.rows;
    // Work on the exactly symmetric part.
    let mut m = Matrix::from_fn(n, n, |i, j| 0.5 * (a[(i, j)] + a[(j, i)]));
    let mut v = Matrix::identity(n);
    let total: f64 = m.data.iter().map(|x| x * x).sum();
    if !total.is_finite() {
        return Err(QcError::InvalidInput("eigen: non-finite matrix".into()));
    }
    let mut converged = false;
    let mut polish_done = false;
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off += m[(i, j)] * m[(i, j)];
                }
            }
        }
        // The off-diagonal squared sum is <= 1e-22 of the total. That bounds the
        // off-diagonal elements only at ~1e-11 ‖A‖, which leaves eigenvector errors of
        // ~1e-10 in SCF use. Because Jacobi converges quadratically, one more sweep
        // takes them to round-off level at little cost.
        if off <= 1e-22 * total {
            if polish_done || off == 0.0 {
                converged = true;
                break;
            }
            polish_done = true;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[(p, q)];
                if apq == 0.0 {
                    continue;
                }
                // theory.typ eq:jacobi (sgn(0) taken as +1)
                let theta = (m[(q, q)] - m[(p, p)]) / (2.0 * apq);
                let t = if theta.abs() > 1e150 {
                    0.5 / theta
                } else {
                    let sgn = if theta >= 0.0 { 1.0 } else { -1.0 };
                    sgn / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // A <- Pᵀ A P with P_pp = P_qq = c, P_pq = s, P_qp = -s.
                for r in 0..n {
                    let arp = m[(r, p)];
                    let arq = m[(r, q)];
                    m[(r, p)] = c * arp - s * arq;
                    m[(r, q)] = s * arp + c * arq;
                }
                for r in 0..n {
                    let apr = m[(p, r)];
                    let aqr = m[(q, r)];
                    m[(p, r)] = c * apr - s * aqr;
                    m[(q, r)] = s * apr + c * aqr;
                }
                m[(p, q)] = 0.0;
                m[(q, p)] = 0.0;
                for r in 0..n {
                    let vrp = v[(r, p)];
                    let vrq = v[(r, q)];
                    v[(r, p)] = c * vrp - s * vrq;
                    v[(r, q)] = s * vrp + c * vrq;
                }
            }
        }
    }
    if !converged {
        return Err(QcError::EigenNotConverged);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| m[(i, i)].total_cmp(&m[(j, j)]));
    let values = order.iter().map(|&i| m[(i, i)]).collect();
    let vectors = Matrix::from_fn(n, n, |r, c| v[(r, order[c])]);
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
    fn jacobi_reconstructs_matrix() {
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
    fn jacobi_degenerate() {
        // diag(1,1,2) rotated: eigenvalues 1,1,2
        let a = Matrix::from_vec(3, 3, vec![1.5, 0.5, 0.0, 0.5, 1.5, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let e = symmetric_eigen(&a).unwrap();
        assert!((e.values[0] - 1.0).abs() < 1e-14);
        assert!((e.values[1] - 1.0).abs() < 1e-14);
        assert!((e.values[2] - 2.0).abs() < 1e-14);
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

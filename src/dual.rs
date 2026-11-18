//! Forward-mode dual numbers carrying a value and its partial derivatives with
//! respect to ρ and σ (theory.typ eq:dual).
//!
//! `Dual::rho(ρ)` = (ρ, 1, 0) and `Dual::sigma(σ)` = (σ, 0, 1); evaluating an
//! energy density written with these operations yields (f, ∂f/∂ρ, ∂f/∂σ).

use crate::boys::boys;
use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// x̂ = (x, x_ρ, x_σ). theory.typ eq:dual
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Value x.
    pub v: f64,
    /// ∂x/∂ρ.
    pub d_rho: f64,
    /// ∂x/∂σ.
    pub d_sigma: f64,
}

impl Dual {
    /// A constant (zero derivatives).
    pub const fn constant(v: f64) -> Self {
        Dual { v, d_rho: 0.0, d_sigma: 0.0 }
    }

    /// The independent variable ρ̂ = (ρ, 1, 0).
    pub const fn rho(v: f64) -> Self {
        Dual { v, d_rho: 1.0, d_sigma: 0.0 }
    }

    /// The independent variable σ̂ = (σ, 0, 1).
    pub const fn sigma(v: f64) -> Self {
        Dual { v, d_rho: 0.0, d_sigma: 1.0 }
    }

    /// Chain rule g(x̂) = (g(x), g'(x) x_ρ, g'(x) x_σ) given g(x) and g'(x). theory.typ eq:dual
    #[inline]
    fn chain(self, g: f64, dg: f64) -> Self {
        Dual { v: g, d_rho: dg * self.d_rho, d_sigma: dg * self.d_sigma }
    }

    /// x^p for real p (x > 0).
    #[inline]
    pub fn powf(self, p: f64) -> Self {
        let g = self.v.powf(p);
        self.chain(g, p * g / self.v)
    }

    /// x^n for integer n.
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        let dg = if n == 0 { 0.0 } else { n as f64 * self.v.powi(n - 1) };
        self.chain(self.v.powi(n), dg)
    }

    /// √x.
    #[inline]
    pub fn sqrt(self) -> Self {
        let g = self.v.sqrt();
        self.chain(g, 0.5 / g)
    }

    /// x^(1/3).
    #[inline]
    pub fn cbrt(self) -> Self {
        let g = self.v.cbrt();
        self.chain(g, g / (3.0 * self.v))
    }

    /// eˣ.
    #[inline]
    pub fn exp(self) -> Self {
        let g = self.v.exp();
        self.chain(g, g)
    }

    /// eˣ − 1, accurate for small |x| (same derivative as eˣ).
    #[inline]
    pub fn exp_m1(self) -> Self {
        self.chain(self.v.exp_m1(), self.v.exp())
    }

    /// erf x = (2x/√π) F₀(x²) from the Boys function; erf'(x) = (2/√π) e^{−x²}.
    /// theory.typ section 9 (short-range B88)
    #[inline]
    pub fn erf(self) -> Self {
        self.chain(erf(self.v), 2.0 / PI.sqrt() * (-self.v * self.v).exp())
    }

    /// ln x.
    #[inline]
    pub fn ln(self) -> Self {
        self.chain(self.v.ln(), 1.0 / self.v)
    }

    /// arctan x; arctan'(x) = 1/(1 + x²).
    #[inline]
    pub fn atan(self) -> Self {
        self.chain(self.v.atan(), 1.0 / (1.0 + self.v * self.v))
    }

    /// asinh x; asinh'(x) = 1/√(1 + x²).
    #[inline]
    pub fn asinh(self) -> Self {
        self.chain(self.v.asinh(), 1.0 / (1.0 + self.v * self.v).sqrt())
    }

    /// 1/x.
    #[inline]
    pub fn recip(self) -> Self {
        let g = 1.0 / self.v;
        self.chain(g, -g * g)
    }
}

/// erf x = (2x/√π) F₀(x²) (Boys function, theory.typ eq:boys).
pub fn erf(x: f64) -> f64 {
    let mut f = [0.0];
    boys(0, x * x, &mut f);
    2.0 * x / PI.sqrt() * f[0]
}

impl From<f64> for Dual {
    fn from(v: f64) -> Self {
        Dual::constant(v)
    }
}

impl Add for Dual {
    type Output = Dual;
    #[inline]
    fn add(self, o: Dual) -> Dual {
        Dual { v: self.v + o.v, d_rho: self.d_rho + o.d_rho, d_sigma: self.d_sigma + o.d_sigma }
    }
}

impl Sub for Dual {
    type Output = Dual;
    #[inline]
    fn sub(self, o: Dual) -> Dual {
        Dual { v: self.v - o.v, d_rho: self.d_rho - o.d_rho, d_sigma: self.d_sigma - o.d_sigma }
    }
}

impl Mul for Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, o: Dual) -> Dual {
        // theory.typ eq:dual (product rule)
        Dual {
            v: self.v * o.v,
            d_rho: self.d_rho * o.v + self.v * o.d_rho,
            d_sigma: self.d_sigma * o.v + self.v * o.d_sigma,
        }
    }
}

impl Div for Dual {
    type Output = Dual;
    #[inline]
    fn div(self, o: Dual) -> Dual {
        let q = self.v / o.v;
        Dual { v: q, d_rho: (self.d_rho - q * o.d_rho) / o.v, d_sigma: (self.d_sigma - q * o.d_sigma) / o.v }
    }
}

impl Neg for Dual {
    type Output = Dual;
    #[inline]
    fn neg(self) -> Dual {
        Dual { v: -self.v, d_rho: -self.d_rho, d_sigma: -self.d_sigma }
    }
}

macro_rules! scalar_ops {
    ($($tr:ident $f:ident),*) => {$(
        impl $tr<f64> for Dual {
            type Output = Dual;
            #[inline]
            fn $f(self, o: f64) -> Dual {
                $tr::$f(self, Dual::constant(o))
            }
        }
        impl $tr<Dual> for f64 {
            type Output = Dual;
            #[inline]
            fn $f(self, o: Dual) -> Dual {
                $tr::$f(Dual::constant(self), o)
            }
        }
    )*};
}
scalar_ops!(Add add, Sub sub, Mul mul, Div div);

#[cfg(test)]
mod tests {
    use super::*;

    /// A test expression that exercises every operation.
    fn expr(r: Dual, s: Dual) -> Dual {
        let a = r.powf(4.0 / 3.0) * s.sqrt() / (1.0 + r.cbrt());
        let b = (s * r).asinh() - (r / s).atan() + (-r).exp() * 2.0;
        let c = (r + s).ln() * r.recip() - s.powi(3) + 0.5 * r.powi(-2);
        a + b - c * (3.0 - r)
    }

    #[test]
    fn derivatives_match_finite_differences() {
        for &(r0, s0) in &[(0.7, 1.3), (2.5, 0.2), (0.05, 3.0), (1.0, 1.0)] {
            let d = expr(Dual::rho(r0), Dual::sigma(s0));
            let f = |r: f64, s: f64| expr(Dual::constant(r), Dual::constant(s)).v;
            let h = 1e-5;
            let fr = (f(r0 + h, s0) - f(r0 - h, s0)) / (2.0 * h);
            let fs = (f(r0, s0 + h) - f(r0, s0 - h)) / (2.0 * h);
            assert!((d.v - f(r0, s0)).abs() < 1e-15 * d.v.abs().max(1.0));
            assert!((d.d_rho - fr).abs() < 1e-7 * fr.abs().max(1.0), "d/drho {} vs {}", d.d_rho, fr);
            assert!((d.d_sigma - fs).abs() < 1e-7 * fs.abs().max(1.0), "d/dsigma {} vs {}", d.d_sigma, fs);
        }
    }

    #[test]
    fn erf_values_and_derivative() {
        // Reference values of erf (Abramowitz & Stegun table 7.1 / 20-digit values).
        let table = [
            (0.0, 0.0),
            (1e-10, 1.1283791670955126e-10),
            (0.1, 0.1124629160182849),
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (2.0, 0.9953222650189527),
            (3.0, 0.9999779095030014),
            (5.0, 0.9999999999984626),
            (6.0, 1.0),
            (40.0, 1.0),
        ];
        for (x, e) in table {
            // A few units in the last place (F₀ is accurate to ~1e-15 relative).
            assert!((erf(x) - e).abs() <= 1e-15 * e, "erf({x}) = {} vs {e}", erf(x));
            assert_eq!(erf(-x), -erf(x));
        }
        // Derivative against central differences, across the Boys-function switch at x² = 35.
        for x in [-2.0, 0.01, 0.3, 1.0, 1.7, 2.5, 35f64.sqrt() - 1e-3, 35f64.sqrt() + 1e-3, 7.0] {
            let d = Dual::rho(x).erf();
            let h = 1e-5;
            let fd = (erf(x + h) - erf(x - h)) / (2.0 * h);
            // Rounding in the difference quotient limits the comparison to ~1e-11 absolute.
            assert!((d.d_rho - fd).abs() < 1e-9 * fd.abs() + 1e-10, "x={x}: {} vs {fd}", d.d_rho);
        }
    }

    #[test]
    fn elementary_derivatives() {
        let x = 0.37;
        let h = 1e-6;
        type Case = (&'static str, fn(Dual) -> Dual, fn(f64) -> f64);
        let cases: [Case; 11] = [
            ("powf", |d| d.powf(-1.0 / 3.0), |v| v.powf(-1.0 / 3.0)),
            ("powi", |d| d.powi(5), |v| v.powi(5)),
            ("sqrt", |d| d.sqrt(), |v| v.sqrt()),
            ("cbrt", |d| d.cbrt(), |v| v.cbrt()),
            ("exp", |d| d.exp(), |v| v.exp()),
            ("ln", |d| d.ln(), |v| v.ln()),
            ("atan", |d| d.atan(), |v| v.atan()),
            ("asinh", |d| d.asinh(), |v| v.asinh()),
            ("recip", |d| d.recip(), |v| 1.0 / v),
            ("exp_m1", |d| d.exp_m1(), |v| v.exp_m1()),
            ("erf", |d| d.erf(), erf),
        ];
        for (name, fd, fv) in cases {
            let d = fd(Dual::sigma(x));
            let num = (fv(x + h) - fv(x - h)) / (2.0 * h);
            assert_eq!(d.d_rho, 0.0, "{name}");
            assert!((d.d_sigma - num).abs() < 1e-8 * num.abs().max(1.0), "{name}: {} vs {num}", d.d_sigma);
        }
    }
}

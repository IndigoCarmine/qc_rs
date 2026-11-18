//! Closed-shell exchange-correlation functionals (theory.typ section 9).
//!
//! Every component is written once as an energy density f(ρ, σ) per volume in
//! terms of [`Dual`] numbers; v_ρ = ∂f/∂ρ and v_σ = ∂f/∂σ come out of the
//! forward-mode differentiation (theory.typ eq:dual).
//!
//! One file per component, each headed by its literature reference:
//! `slater.rs`, `b88.rs`, `vwn.rs` (VWN5 and VWN-RPA share one formula), `lyp.rs`
//! and `cam_b88.rs` (short-range B88 of CAM-B3LYP). This file holds the mixtures
//! (BLYP, B3LYP, CAM-B3LYP, …).

mod b88;
mod cam_b88;
mod lyp;
mod slater;
mod vwn;

use crate::dual::Dual;
use crate::error::{QcError, Result};
use crate::integrals::ExchangeKernel;
use b88::b88;
use cam_b88::cam_b88;
use lyp::lyp;
use slater::slater;
use vwn::{VWN_RPA, VWN5, vwn};

/// A single exchange or correlation energy density.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum XcComponent {
    /// Slater (Dirac) LDA exchange. theory.typ eq:slater
    SlaterExchange,
    /// Becke 88 GGA exchange (includes the Slater part). theory.typ eq:b88
    B88Exchange,
    /// VWN correlation, parametrization 5 (VWN5). theory.typ eq:vwn
    Vwn5Correlation,
    /// VWN correlation, RPA parametrization (VWN-RPA, "VWN3" in Gaussian). theory.typ eq:vwn
    VwnRpaCorrelation,
    /// Lee–Yang–Parr GGA correlation, closed-shell form. theory.typ eq:lyp
    LypCorrelation,
    /// B88 exchange for the part [1 − α − β erf(ωr)]/r of the Coulomb operator
    /// (Coulomb-attenuating method; the rest is exact exchange with the same α, β, ω).
    /// theory.typ eq:cam-b88
    CamB88Exchange {
        /// Short-range fraction α of exact exchange.
        alpha: f64,
        /// Long-range addition β of exact exchange.
        beta: f64,
        /// Range-separation parameter ω (Bohr⁻¹).
        omega: f64,
    },
}

impl XcComponent {
    /// Whether the component depends on σ = |∇ρ|².
    pub fn is_gga(self) -> bool {
        matches!(self, XcComponent::B88Exchange | XcComponent::LypCorrelation | XcComponent::CamB88Exchange { .. })
    }

    /// Energy density f(ρ̂, σ̂) as a dual number (ρ > 0, σ > 0).
    pub fn energy_density(self, rho: Dual, sigma: Dual) -> Dual {
        match self {
            XcComponent::SlaterExchange => slater(rho),
            XcComponent::B88Exchange => b88(rho, sigma),
            XcComponent::Vwn5Correlation => vwn(rho, VWN5),
            XcComponent::VwnRpaCorrelation => vwn(rho, VWN_RPA),
            XcComponent::LypCorrelation => lyp(rho, sigma),
            XcComponent::CamB88Exchange { alpha, beta, omega } => cam_b88(rho, sigma, alpha, beta, omega),
        }
    }
}

/// f, ∂f/∂ρ and ∂f/∂σ at one point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct XcValues {
    /// Energy density f (per volume).
    pub f: f64,
    /// v_ρ = ∂f/∂ρ.
    pub v_rho: f64,
    /// v_σ = ∂f/∂σ.
    pub v_sigma: f64,
}

/// f = Σ_k c_k f_k plus exact exchange with the kernel [α + β erf(ωr)]/r
/// (theory.typ section 9.5 and eq:cam-b3lyp; an ordinary hybrid has α = a_HF, β = 0).
#[derive(Debug, Clone, PartialEq)]
pub struct Functional {
    name: String,
    terms: Vec<(f64, XcComponent)>,
    exact_exchange: ExchangeKernel,
}

impl Functional {
    /// General functional: linear combination `terms` of components and a global
    /// fraction a_HF of exact exchange.
    pub fn new(name: &str, terms: Vec<(f64, XcComponent)>, exact_exchange: f64) -> Self {
        Self::with_exchange_kernel(name, terms, ExchangeKernel::global(exact_exchange))
    }

    /// General functional with an exact-exchange kernel [α + β erf(ωr)]/r
    /// (theory.typ eq:cam-k). The density-functional exchange in `terms` must cover the
    /// complementary part of the operator (e.g. [`XcComponent::CamB88Exchange`] with the
    /// same α, β, ω).
    pub fn with_exchange_kernel(name: &str, terms: Vec<(f64, XcComponent)>, kernel: ExchangeKernel) -> Self {
        Functional { name: name.to_string(), terms, exact_exchange: kernel }
    }

    /// Hartree–Fock: f = 0, a_HF = 1.
    pub fn hf() -> Self {
        Self::new("HF", vec![], 1.0)
    }

    /// Slater exchange only.
    pub fn slater() -> Self {
        Self::new("Slater", vec![(1.0, XcComponent::SlaterExchange)], 0.0)
    }

    /// SVWN5 (LDA): Slater exchange + VWN5 correlation (references in `slater.rs`, `vwn.rs`).
    pub fn svwn5() -> Self {
        Self::new("SVWN5", vec![(1.0, XcComponent::SlaterExchange), (1.0, XcComponent::Vwn5Correlation)], 0.0)
    }

    /// Slater + VWN-RPA.
    pub fn svwn_rpa() -> Self {
        Self::new("SVWN-RPA", vec![(1.0, XcComponent::SlaterExchange), (1.0, XcComponent::VwnRpaCorrelation)], 0.0)
    }

    /// Becke 88 exchange only.
    pub fn b88() -> Self {
        Self::new("B88", vec![(1.0, XcComponent::B88Exchange)], 0.0)
    }

    /// BLYP: B88 exchange + LYP correlation (references in `b88.rs` and `lyp.rs`).
    pub fn blyp() -> Self {
        Self::new("BLYP", vec![(1.0, XcComponent::B88Exchange), (1.0, XcComponent::LypCorrelation)], 0.0)
    }

    /// B3LYP with VWN-RPA (Gaussian's definition; PySCF "b3lypg").
    ///
    /// References:
    /// - A. D. Becke, J. Chem. Phys. 98, 5648 (1993): the three-parameter hybrid
    ///   (exact exchange mixed with Slater, B88 and a gradient-corrected
    ///   correlation; originally with PW91 correlation), parameters 0.20, 0.72, 0.81.
    /// - P. J. Stephens, F. J. Devlin, C. F. Chabalowski, M. J. Frisch,
    ///   J. Phys. Chem. 98, 11623 (1994): the same three parameters with LYP + VWN
    ///   correlation, i.e. "B3LYP" as implemented in Gaussian (VWN-RPA).
    pub fn b3lyp() -> Self {
        Self::new("B3LYP", Self::b3lyp_terms(XcComponent::VwnRpaCorrelation), 0.20)
    }

    /// B3LYP with VWN5 (PySCF "b3lyp5"): same mixing as [`Functional::b3lyp`], but the
    /// local correlation is the Monte Carlo fit VWN5 instead of VWN-RPA. Several
    /// programs use this variant under the name B3LYP, so check which one a
    /// reference value was computed with.
    pub fn b3lyp5() -> Self {
        Self::new("B3LYP5", Self::b3lyp_terms(XcComponent::Vwn5Correlation), 0.20)
    }

    /// CAM-B3LYP (theory.typ eq:cam-b3lyp): short-range B88 exchange (eq:cam-b88) +
    /// 0.81 LYP + 0.19 VWN5, exact exchange with α = 0.19, β = 0.46, ω = 0.33 Bohr⁻¹
    /// (PySCF "camb3lyp"). References in `cam_b88.rs`, `lyp.rs` and `vwn.rs`.
    pub fn cam_b3lyp() -> Self {
        let kernel = ExchangeKernel { alpha: 0.19, beta: 0.46, omega: 0.33 };
        let x = XcComponent::CamB88Exchange { alpha: kernel.alpha, beta: kernel.beta, omega: kernel.omega };
        Self::with_exchange_kernel(
            "CAM-B3LYP",
            vec![(1.0, x), (0.81, XcComponent::LypCorrelation), (0.19, XcComponent::Vwn5Correlation)],
            kernel,
        )
    }

    fn b3lyp_terms(vwn: XcComponent) -> Vec<(f64, XcComponent)> {
        vec![
            (0.08, XcComponent::SlaterExchange),
            (0.72, XcComponent::B88Exchange),
            (0.19, vwn),
            (0.81, XcComponent::LypCorrelation),
        ]
    }

    /// Looks up a functional by name (case-insensitive, spaces ignored). Accepts the
    /// names of theory.typ section 9.5 and PySCF-style spellings:
    /// "hf"; "slater", "slater,", "lda,"; "svwn5", "slater,vwn5"; "svwn-rpa", "svwn_rpa",
    /// "slater,vwn_rpa"; "b88", "b88,"; "blyp", "b88,lyp"; "b3lyp", "b3lypg"; "b3lyp5";
    /// "cam-b3lyp", "camb3lyp", "cam_b3lyp".
    pub fn from_name(name: &str) -> Result<Self> {
        let key: String = name.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_ascii_lowercase();
        Ok(match key.as_str() {
            "hf" | "rhf" => Self::hf(),
            "slater" | "slater," | "lda," | "s" => Self::slater(),
            "svwn5" | "slater,vwn5" | "lda,vwn5" => Self::svwn5(),
            "svwn-rpa" | "svwn_rpa" | "svwnrpa" | "slater,vwn_rpa" | "slater,vwn-rpa" | "lda,vwn_rpa" => {
                Self::svwn_rpa()
            }
            "b88" | "b88," | "becke" | "b" => Self::b88(),
            "blyp" | "b88,lyp" => Self::blyp(),
            "b3lyp" | "b3lypg" => Self::b3lyp(),
            "b3lyp5" => Self::b3lyp5(),
            "cam-b3lyp" | "camb3lyp" | "cam_b3lyp" => Self::cam_b3lyp(),
            _ => return Err(QcError::InvalidInput(format!("unknown functional '{name}'"))),
        })
    }

    /// Display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The (coefficient, component) list.
    pub fn terms(&self) -> &[(f64, XcComponent)] {
        &self.terms
    }

    /// a_HF of an ordinary hybrid; for a range-separated functional the short-range
    /// fraction α (see [`Functional::exchange_kernel`] for the full kernel).
    pub fn exact_exchange_fraction(&self) -> f64 {
        self.exact_exchange.alpha
    }

    /// The exact-exchange kernel (α, β, ω) of theory.typ eq:cam-k.
    pub fn exchange_kernel(&self) -> ExchangeKernel {
        self.exact_exchange
    }

    /// Whether any term needs σ (and therefore ∇ρ and AO gradients).
    pub fn is_gga(&self) -> bool {
        self.terms.iter().any(|(_, c)| c.is_gga())
    }

    /// Whether there is a density-functional part at all.
    pub fn has_density_part(&self) -> bool {
        self.terms.iter().any(|(c, _)| *c != 0.0)
    }

    /// f, v_ρ, v_σ at (ρ, σ), for ρ > 0 and σ > 0 (cut-offs are applied by the caller).
    pub fn eval(&self, rho: f64, sigma: f64) -> XcValues {
        let (r, s) = (Dual::rho(rho), Dual::sigma(sigma));
        let mut tot = Dual::constant(0.0);
        for &(c, comp) in &self.terms {
            tot = tot + c * comp.energy_density(r, s);
        }
        XcValues { f: tot.v, v_rho: tot.d_rho, v_sigma: tot.d_sigma }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const ALL: [XcComponent; 6] = [
        XcComponent::SlaterExchange,
        XcComponent::B88Exchange,
        XcComponent::Vwn5Correlation,
        XcComponent::VwnRpaCorrelation,
        XcComponent::LypCorrelation,
        XcComponent::CamB88Exchange { alpha: 0.19, beta: 0.46, omega: 0.33 },
    ];

    #[test]
    fn potentials_match_finite_differences() {
        for comp in ALL {
            let fun = Functional::new("t", vec![(1.0, comp)], 0.0);
            for &(rho, sigma) in &[(1e-3, 1e-6), (0.05, 0.01), (0.7, 0.3), (12.0, 50.0), (300.0, 1e5)] {
                let x = fun.eval(rho, sigma);
                let hr = 1e-5 * rho;
                let hs = 1e-5 * sigma;
                let fr = (fun.eval(rho + hr, sigma).f - fun.eval(rho - hr, sigma).f) / (2.0 * hr);
                let fs = (fun.eval(rho, sigma + hs).f - fun.eval(rho, sigma - hs).f) / (2.0 * hs);
                assert!((x.v_rho - fr).abs() <= 1e-7 * fr.abs().max(1e-6), "{comp:?} v_rho {} vs {fr}", x.v_rho);
                assert!(
                    (x.v_sigma - fs).abs() <= 1e-6 * fs.abs().max(1e-8),
                    "{comp:?} v_sigma {} vs {fs}",
                    x.v_sigma
                );
                if !comp.is_gga() {
                    assert_eq!(x.v_sigma, 0.0);
                }
            }
        }
    }

    #[test]
    fn b88_reduces_to_slater_without_gradient() {
        let rho = 0.3;
        let s = Functional::slater().eval(rho, 1e-40);
        let b = Functional::b88().eval(rho, 1e-40);
        assert!((s.f - b.f).abs() < 1e-14);
        assert!((s.v_rho - b.v_rho).abs() < 1e-14);
        // Slater exchange per particle -3/4 (3ρ/π)^(1/3).
        assert!((s.f / rho + 0.75 * (3.0 * rho / PI).cbrt()).abs() < 1e-15);
    }

    #[test]
    fn hybrids_are_linear_combinations() {
        let (rho, sigma) = (0.4, 0.2);
        let c = |comp: XcComponent| Functional::new("t", vec![(1.0, comp)], 0.0).eval(rho, sigma).f;
        let b3 = Functional::b3lyp().eval(rho, sigma).f;
        let expect = 0.08 * c(XcComponent::SlaterExchange)
            + 0.72 * c(XcComponent::B88Exchange)
            + 0.19 * c(XcComponent::VwnRpaCorrelation)
            + 0.81 * c(XcComponent::LypCorrelation);
        assert!((b3 - expect).abs() < 1e-14);
        assert_eq!(Functional::b3lyp().exact_exchange_fraction(), 0.2);
        assert_eq!(Functional::b3lyp().exchange_kernel(), ExchangeKernel::global(0.2));
        let cam = Functional::cam_b3lyp();
        let x = XcComponent::CamB88Exchange { alpha: 0.19, beta: 0.46, omega: 0.33 };
        let expect = c(x) + 0.81 * c(XcComponent::LypCorrelation) + 0.19 * c(XcComponent::Vwn5Correlation);
        assert!((cam.eval(rho, sigma).f - expect).abs() < 1e-14);
        assert_eq!(cam.exchange_kernel(), ExchangeKernel { alpha: 0.19, beta: 0.46, omega: 0.33 });
        assert_eq!(cam.exact_exchange_fraction(), 0.19);
        assert!(cam.exchange_kernel().is_range_separated() && !Functional::b3lyp().exchange_kernel().is_range_separated());
    }

    #[test]
    fn lookup_by_name() {
        assert_eq!(Functional::from_name("B3LYPG").unwrap(), Functional::b3lyp());
        assert_eq!(Functional::from_name("b3lyp").unwrap(), Functional::b3lyp());
        assert_eq!(Functional::from_name("b3lyp5").unwrap(), Functional::b3lyp5());
        assert_eq!(Functional::from_name("slater, vwn5").unwrap(), Functional::svwn5());
        assert_eq!(Functional::from_name("slater,vwn_rpa").unwrap(), Functional::svwn_rpa());
        assert_eq!(Functional::from_name("b88,lyp").unwrap(), Functional::blyp());
        assert_eq!(Functional::from_name("b88,").unwrap(), Functional::b88());
        assert_eq!(Functional::from_name("slater,").unwrap(), Functional::slater());
        assert_eq!(Functional::from_name("HF").unwrap(), Functional::hf());
        for name in ["cam-b3lyp", "CAM-B3LYP", "camb3lyp", "cam_b3lyp"] {
            assert_eq!(Functional::from_name(name).unwrap(), Functional::cam_b3lyp());
        }
        assert!(Functional::cam_b3lyp().is_gga());
        assert!(!Functional::hf().has_density_part());
        assert!(Functional::blyp().is_gga());
        assert!(!Functional::svwn5().is_gga());
        assert!(Functional::from_name("pbe").is_err());
    }
}

# Basis set data

The files in this directory are **not original work of the qc_rs project and are
not covered by the copyright line in the top-level `LICENSE`**. They belong to,
and are attributed to, the Basis Set Exchange:

- Source: [Basis Set Exchange](https://www.basissetexchange.org/) (Molecular
  Sciences Software Institute, MolSSI, and Pacific Northwest National Laboratory /
  EMSL), downloaded on 2026-11-18 through its web API in Gaussian94 format,
  unmodified. Each file keeps the header written by the Basis Set Exchange
  (version 0.12; "Data from Gaussian 09/GAMESS").
- Files: `sto-3g.gbs`, `6-31g.gbs`, `6-31gs.gbs` (6-31G*), `6-31gss.gbs` (6-31G**),
  elements H–Ar.
- The Basis Set Exchange software and the data distributed with it are published
  under the BSD 3-Clause license, Copyright (c) 2020 The Molecular Sciences Software
  Institute, Virginia Tech (https://github.com/MolSSI-BSE/basis_set_exchange).
  The Basis Set Exchange website does not state separate redistribution terms for
  the data.

If you use these data, cite the Basis Set Exchange:

- B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson, T. L. Windus,
  J. Chem. Inf. Model. 59, 4814 (2019), doi:10.1021/acs.jcim.9b00725
- D. Feller, J. Comput. Chem. 17, 1571 (1996)
- K. L. Schuchardt et al., J. Chem. Inf. Model. 47, 1045 (2007), doi:10.1021/ci600510j

and the papers that define the basis sets themselves:

- STO-3G: W. J. Hehre, R. F. Stewart, J. A. Pople, J. Chem. Phys. 51, 2657 (1969)
- 6-31G: R. Ditchfield, W. J. Hehre, J. A. Pople, J. Chem. Phys. 54, 724 (1971);
  W. J. Hehre, R. Ditchfield, J. A. Pople, J. Chem. Phys. 56, 2257 (1972)
- Polarization functions (6-31G*, 6-31G**): P. C. Hariharan, J. A. Pople,
  Theor. Chim. Acta 28, 213 (1973); second-row elements: M. M. Francl et al.,
  J. Chem. Phys. 77, 3654 (1982)

The exponents and contraction coefficients have not been compared with the tables
in those papers by this project.

Note for binary distributions: the library embeds the text of these files at
compile time (`include_str!` in `src/basis.rs`), so a compiled program contains
the data and should carry this attribution.

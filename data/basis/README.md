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
- Files `6-311g.gbs` (6-311G), `6-311gs.gbs` (6-311G*) and `6-311gss.gbs`
  (6-311G**), elements H–Ar, downloaded on 2026-11-19 in the same way from
  `https://www.basissetexchange.org/api/basis/<name>/format/gaussian94/?elements=1-18`
  with the Basis Set Exchange names `6-311g`, `6-311g_st_` and `6-311g_st__st_`,
  unmodified. Their headers read version 0.12, "Data from the Original Basis Set
  Exchange". 6-311G contains S, P and SP shells; 6-311G* and 6-311G** add D
  shells. None of the three has F shells.
- Basis sets with f functions, downloaded on 2026-11-19 in the same way and
  unmodified. 6-311G(2df,2p) was the set first planned, but the Basis Set Exchange
  has no basis set of that name, so these three were taken instead. All three
  headers read version 0.12, "Data from the Original Basis Set Exchange" (the
  6-31G(2df,p) header writes "original" in lower case).
  - `6-31g_2df_p.gbs` (6-31G(2df,p), BSE name `6-31g(2df,p)`, `?elements=1-18`):
    elements H–Ar. Shells S, P, D, SP and F; F shells on Li–Ar. In these data
    H and He carry two P shells.
  - `6-311g_2df_2pd.gbs` (6-311G(2df,2pd), BSE name `6-311g(2df,2pd)`,
    `?elements=1-10`): elements H–Ne only. The Basis Set Exchange has no Na–Ar
    for this set (it does have K and Ca, which are outside this program's range).
    Shells S, P, D, SP and F; F shells on Li–Ne.
  - `6-311ppg_3df_3pd.gbs` (6-311++G(3df,3pd), BSE name `6-311++g(3df,3pd)`,
    `?elements=1,3-18`): elements H and Li–Ar. The Basis Set Exchange has no He
    for this set. Includes diffuse functions. Shells S, P, D, SP and F; F shells
    on Li–Ar.
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
- 6-311G, H and Li–Ne, and its polarization functions (6-311G*, 6-311G**) for
  H–Ne: R. Krishnan, J. S. Binkley, R. Seeger, J. A. Pople, J. Chem. Phys. 72,
  650 (1980), doi:10.1063/1.438955
- 6-311G, He: Gaussian 09 Revision E.01, M. J. Frisch et al., Gaussian, Inc.,
  Wallingford, CT (2009)
- 6-311G, Na–Ar: A. D. McLean, G. S. Chandler, J. Chem. Phys. 72, 5639 (1980),
  doi:10.1063/1.438980
- Polarization functions of 6-311G* and 6-311G** for Na–Ar: M. M. Francl et al.,
  J. Chem. Phys. 77, 3654 (1982), doi:10.1063/1.444267
- 6-31G(2df,p): valence and core functions as for 6-31G above, plus
  J. D. Dill, J. A. Pople, J. Chem. Phys. 62, 2921 (1975),
  doi:10.1063/1.430801 (Li, B); J. S. Binkley, J. A. Pople, J. Chem. Phys. 66,
  879 (1977), doi:10.1063/1.433929 (Be); M. S. Gordon et al., J. Am. Chem. Soc.
  104, 2797 (1982), doi:10.1021/ja00374a017 (Na–Ar core); Francl et al. 1982
  (Na–Ar valence and 2d). The Basis Set Exchange gives no reference for the He
  and Ne valence functions.
- f and multiple polarization functions (6-31G(2df,p), 6-311G(2df,2pd),
  6-311++G(3df,3pd)): Krishnan et al. 1980 (above); M. J. Frisch, J. A. Pople,
  J. S. Binkley, J. Chem. Phys. 80, 3265 (1984), doi:10.1063/1.447079; for Ar f
  functions: L. A. Curtiss, K. Raghavachari, P. C. Redfern, V. Rassolov,
  J. A. Pople, J. Chem. Phys. 109, 7764 (1998), doi:10.1063/1.477422;
  for the Na–Ar polarization of 6-311++G(3df,3pd): Francl et al. 1982
- 6-311G(2df,2pd) and 6-311++G(3df,3pd) valence functions: as for 6-311G above
  (Krishnan et al. 1980; Gaussian 09 for He; McLean and Chandler 1980 for Na–Ar)
- Diffuse functions of 6-311++G(3df,3pd): T. Clark, J. Chandrasekhar,
  G. W. Spitznagel, P. v. R. Schleyer, J. Comput. Chem. 4, 294 (1983),
  doi:10.1002/jcc.540040303 (H, Li–F); G. W. Spitznagel, T. Clark,
  P. v. R. Schleyer, W. J. Hehre, J. Comput. Chem. 8, 1109 (1987),
  doi:10.1002/jcc.540080807 (Na–Cl); P. M. W. Gill, B. G. Johnson, J. A. Pople,
  M. J. Frisch, Chem. Phys. Lett. 197, 499 (1992),
  doi:10.1016/0009-2614(92)85807-m (Ne, Ar)

(The 6-311G and f-function references are as listed by the Basis Set Exchange
references API,
`https://www.basissetexchange.org/api/references/<name>/format/txt/?elements=...`,
with the same element list as the data request.)

The exponents and contraction coefficients have not been compared with the tables
in those papers by this project.

Note for binary distributions: the library embeds the text of these files at
compile time (`include_str!` in `src/basis.rs`), so a compiled program contains
the data and should carry this attribution. All files listed above are embedded.

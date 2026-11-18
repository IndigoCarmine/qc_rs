# DFT-D3 dispersion data

The files in this directory are **not original work of the qc_rs project and are
not covered by the copyright line in the top-level `LICENSE`**. The data belong to,
and are attributed to, the authors of tad-dftd3 and tad-mctc, who distribute them
under the Apache License 2.0:

- Source: the Python packages
  [tad-dftd3](https://github.com/dftd3/tad-dftd3) **0.7.0** (authors listed in its
  package metadata: Sebastian Ehlert, Marvin Friede) and
  [tad-mctc](https://github.com/tad-mctc/tad-mctc) **0.8.0** (author: Marvin Friede),
  installed from PyPI on 2026-11-18. The package metadata of both versions says
  `License: Apache-2.0`, and both ship the same, unmodified Apache License 2.0 text,
  copied here as `LICENSE-APACHE-2.0.txt`. Neither package ships a `NOTICE` file.
  The exact wording of the copyright notices of the two projects has not been
  checked (the license file in the packages leaves the copyright line as the
  template `[yyyy] [name of copyright owner]`).
- Files: `d3_h_ar.dat` (the data), `LICENSE-APACHE-2.0.txt` (the license).
- The values themselves go back to the original DFT-D3 papers and program by
  S. Grimme and co-workers (see the citations below). This project relies on
  tad-dftd3 and tad-mctc distributing them under Apache-2.0; how they came to be
  distributed under that license has not been checked by this project.

## Changes made by qc_rs

`d3_h_ar.dat` is a modified form of the source data, as Apache-2.0 section 4(b)
requires us to state:

- **Subset:** only H–Ar (Z = 1..18) and only the quantities listed below.
- **Format conversion:** the source keeps the data as PyTorch tensors. They were
  read through the public API (`tad_dftd3.reference.Reference`,
  `tad_dftd3.data.R4R2`, `tad_mctc.data.COV_D3`, `tad_mctc.data.VDW_PAIRWISE`) by
  `tools/extract_d3_data.py` and written as text. No source code of tad-dftd3,
  tad-mctc, dftd3 or simple-dftd3 was read.
- **Number formatting:** tad-dftd3 stores the reference CNs, C6 and r4r2 as float32,
  and the r0ab values are float32 values as well. These are written as the shortest
  decimal that reads back as the same float32 (for example the H–H C6 is stored as
  the float32 nearest to 3.0267 and written as `3.0267`). Reading the file as
  float64 therefore gives the decimal value, which differs from tad's float32 value
  by at most half a float32 unit in the last place (relative ~6e-8). `rcov` is
  float64 in the source and written in full.

The extraction script checks, after writing, that every value read back equals the
source value (at float32 precision for the float32 data), and that D3(BJ) energies
of several small molecules recomputed from this file alone agree with
`tad_dftd3.dftd3` to a relative 1e-6 (observed: ~1e-8).

## File format and meaning of the quantities

Plain text; `#` starts a comment line. Three kinds of records:

```
element Z symbol rcov r4r2 nref cnref_1 ... cnref_nref
r0ab Za Zb value                (Za <= Zb; the value is symmetric)
c6 Za Zb i j value              (Za <= Zb; i = reference of Za, j = reference of Zb)
```

- `rcov` — covalent radius for the D3 coordination number, **in bohr and already
  multiplied by k2 = 4/3**. Source: Pyykkö and Atsumi, values for metals decreased
  by 10 % (docstring of `tad_mctc.data.COV_D3`). Checked: for H and C the value
  equals 0.32 Å and 0.75 Å × 4/3 / 0.52917721067 Å to 1e-15, and the D3 energies
  are reproduced with CN_A = Σ_B 1 / (1 + exp(−16 ((rcov_A + rcov_B) / r_AB − 1))),
  i.e. with no further scaling of `rcov`.
- `r4r2` — the per-atom factor used for C8, **in bohr**, entered **as is** in
  C8_AB = 3 C6_AB r4r2_A r4r2_B (checked: this reproduces the tad-dftd3 D3(BJ)
  energies). It is therefore the square root of the ⟨r⁴⟩/⟨r²⟩-based quantity of
  Grimme et al. (2010), not ⟨r⁴⟩/⟨r²⟩ itself. Whether the source's definition
  includes the factor ½√Z of the 2010 paper is **unconfirmed** (only the
  combination above was checked). The docstring of `tad_dftd3.data.r4r2` says the
  values are PBE0/def2-QZVP atomic values by S. Grimme (2010), with the rare gases
  recalculated by J. Mewes with PBE0/aug-cc-pVQZ (2018), so He, Ne and Ar may differ
  from implementations that use the 2010 values.
- `nref`, `cnref_k` — the number of reference systems of the element and their
  coordination numbers, in the order of the source (H lists CN = 0.9118 first and
  CN = 0 second; all other elements start with CN = 0). Unused slots of the source
  (padded with −1, with zero C6) are dropped.
- `c6 Za Zb i j` — reference C6 in **hartree·bohr⁶** for atom A of element Za in its
  reference i and atom B of element Zb in its reference j (1-based, in the order of
  the `element` line). Only Za ≤ Zb is written; the other half follows from
  C6(Zb, Za, j, i) = C6(Za, Zb, i, j) (checked on the source for all of H–Ar). For
  Za = Zb all i, j are written. The interpolation that reproduced tad-dftd3 is the
  normalized Gaussian weight W_A,i = exp(−4 (CN_A − cnref_i)²) / Σ_k exp(−4 (CN_A −
  cnref_k)²) and C6_AB = Σ_ij W_A,i W_B,j C6(Za, Zb, i, j). How the source treats the
  case where all weights underflow (very large CN) has **not been checked**.
- `r0ab` — the pairwise cutoff radius R0_AB, **in bohr**. Checked: the
  Axilrod–Teller–Muto three-body energy of tad-dftd3 (which uses these radii) is
  reproduced with the damping 1 / (1 + 6 (r̄ / (4/3 · R̄0))^−16) only when the
  values are taken as bohr. tad-dftd3 0.7.0 has no zero damping, so the use of these
  radii in D3(0) could not be checked against it; the D3(0) formula is the one in
  `docs/theory.typ`.

The constants k1 = 16, k2 = 4/3 (already in `rcov`), k3 = 4 and the damping
parameters of each functional are **not** in this file.

## Citation

If you use these data, cite the tad-dftd3 / tad-mctc paper:

- M. Friede, C. Hölzer, S. Ehlert, S. Grimme, J. Chem. Phys. 161, 062501 (2024),
  doi:10.1063/5.0216715

and the papers that define DFT-D3 and its data:

- S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys. 132, 154104 (2010),
  doi:10.1063/1.3382344 (D3, reference C6, coordination number, zero damping)
- S. Grimme, S. Ehrlich, L. Goerigk, J. Comput. Chem. 32, 1456 (2011),
  doi:10.1002/jcc.21759 (Becke–Johnson damping)
- P. Pyykkö, M. Atsumi, Chem. Eur. J. 15, 188–197 (2009) (covalent radii; as given
  in the tad-mctc docstring, not checked against the journal)

## Regenerating

From the repository root, in a throw-away environment (never in the project's
`.venv`):

```
uv run --no-project --with tad-dftd3==0.7.0 --with tad-mctc==0.8.0 \
    python -I tools/extract_d3_data.py data/dispersion/d3_h_ar.dat
```

Note for binary distributions: if the library embeds this file at compile time,
a compiled program contains the data and must carry this attribution and the
Apache License 2.0 text (`LICENSE-APACHE-2.0.txt`).

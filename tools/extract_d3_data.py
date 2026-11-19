"""Write the DFT-D3 reference data for H-Ar (Z = 1..18) to a plain-text file.

The data come from the Apache-2.0 packages tad-dftd3 and tad-mctc and are read only
through their public API (no source code of tad-dftd3, tad-mctc, dftd3 or
simple-dftd3 was read to write this script). See data/dispersion/README.md for the
meaning of every quantity and for the license.

Run it in a throw-away environment, never in the project's .venv (from the
repository root):

    uv run --no-project --with tad-dftd3==0.7.0 --with tad-mctc==0.8.0 \
        python -I tools/extract_d3_data.py data/dispersion/d3_h_ar.dat

After writing, the script reads the file back and checks it against tad-dftd3:
it recomputes D3(BJ) energies of a few small molecules from the file alone (with
the formulas of Grimme et al. 2010 and 2011) and compares them with
tad_dftd3.dftd3. It exits with status 1 if anything disagrees.
"""
import math
import sys

import numpy as np
import torch
import tad_dftd3 as d3
import tad_mctc as mctc

MAX_Z = 18
EXPECTED_VERSIONS = {"tad_dftd3": "0.7.0", "tad_mctc": "0.8.0"}


def f32(x):
    """Shortest decimal that reads back as the same float32 value."""
    return np.format_float_positional(np.float32(x), unique=True, trim="-")


def f64(x):
    return repr(float(x))


def is_float32_exact(t):
    return bool(torch.equal(t.double(), t.float().double()))


def extract():
    ref = d3.reference.Reference()
    cn = ref.cn
    c6 = ref.c6
    rcov = mctc.data.COV_D3()
    r4r2 = d3.data.R4R2()
    r0ab = mctc.data.VDW_PAIRWISE()

    # Formatting rules: values that tad stores as float32 are written as the
    # shortest decimal that rounds to the same float32; float64 values in full.
    assert cn.dtype == torch.float32 and c6.dtype == torch.float32
    assert r4r2.dtype == torch.float32
    assert rcov.dtype == torch.float64
    # VDW_PAIRWISE comes back as float64 but every entry is a float32 value.
    assert is_float32_exact(r0ab[: MAX_Z + 1, : MAX_Z + 1])

    elements = []
    for z in range(1, MAX_Z + 1):
        refs = [i for i in range(cn.shape[1]) if cn[z, i] >= 0.0]
        # Valid references come first; unused slots are padded with -1.
        assert refs == list(range(len(refs))), (z, refs)
        elements.append((z, refs))

    nref = {z: len(refs) for z, refs in elements}
    for za, _ in elements:
        for zb, _ in elements:
            block = c6[za, zb]
            # C6 is symmetric under swapping the atoms together with their references.
            assert torch.equal(block, c6[zb, za].T)
            # Entries beyond the valid references are zero.
            assert torch.count_nonzero(block[nref[za]:, :]) == 0
            assert torch.count_nonzero(block[:, nref[zb]:]) == 0
            assert r0ab[za, zb] == r0ab[zb, za]
    return cn, c6, rcov, r4r2, r0ab, elements


def write(path, cn, c6, rcov, r4r2, r0ab, elements):
    lines = [
        "# DFT-D3 reference data for H-Ar (Z = 1..18).",
        f"# Extracted with tools/extract_d3_data.py from tad-dftd3 {d3.__version__} "
        f"and tad-mctc {mctc.__version__} (Apache-2.0).",
        "# Not original work of qc_rs; see data/dispersion/README.md for the source,",
        "# the license (LICENSE-APACHE-2.0.txt) and the meaning of every column.",
        "# Units: bohr for rcov and r0ab; r4r2 in bohr (see README); C6 in hartree*bohr^6.",
        "#",
        "# element Z symbol rcov r4r2 nref cnref_1 ... cnref_nref",
        "# r0ab Za Zb value                 (Za <= Zb)",
        "# c6 Za Zb i j value               (Za <= Zb; i: reference of Za, j: reference of Zb, 1-based)",
    ]
    for z, refs in elements:
        cns = " ".join(f32(cn[z, i]) for i in refs)
        lines.append(
            f"element {z} {mctc.data.Z2S[z]} {f64(rcov[z])} {f32(r4r2[z])} {len(refs)} {cns}"
        )
    for za, _ in elements:
        for zb, _ in elements:
            if za <= zb:
                lines.append(f"r0ab {za} {zb} {f32(r0ab[za, zb])}")
    for za, refs_a in elements:
        for zb, refs_b in elements:
            if za > zb:
                continue
            for i in refs_a:
                for j in refs_b:
                    lines.append(f"c6 {za} {zb} {i + 1} {j + 1} {f32(c6[za, zb, i, j])}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def read(path):
    """Parse the file with plain Python (no tad); returns float64 data."""
    el, r0, c6 = {}, {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            tok = line.split()
            if tok[0] == "element":
                z, n = int(tok[1]), int(tok[5])
                el[z] = dict(rcov=float(tok[3]), r4r2=float(tok[4]),
                             cnref=[float(x) for x in tok[6:6 + n]])
                assert len(tok) == 6 + n
            elif tok[0] == "r0ab":
                za, zb = int(tok[1]), int(tok[2])
                r0[za, zb] = r0[zb, za] = float(tok[3])
            elif tok[0] == "c6":
                za, zb, i, j = (int(x) for x in tok[1:5])
                c6[za, zb, i - 1, j - 1] = c6[zb, za, j - 1, i - 1] = float(tok[5])
            else:
                raise ValueError(f"unknown record: {line!r}")
    return el, r0, c6


def bj_energy(el, c6tab, numbers, pos, a1, s8, a2, s6=1.0):
    """Two-body D3(BJ) energy from the file data, Grimme et al. 2010/2011."""
    n = len(numbers)
    dist = [[math.dist(pos[a], pos[b]) for b in range(n)] for a in range(n)]
    # Coordination number with k1 = 16; rcov already includes the 4/3 scaling.
    cn = [sum(1.0 / (1.0 + math.exp(-16.0 * ((el[numbers[a]]["rcov"] + el[numbers[b]]["rcov"])
                                               / dist[a][b] - 1.0)))
              for b in range(n) if b != a) for a in range(n)]
    # Gaussian weights with k3 = 4, normalized over the references of the atom.
    w = []
    for a in range(n):
        g = [math.exp(-4.0 * (cn[a] - c) ** 2) for c in el[numbers[a]]["cnref"]]
        s = sum(g)
        w.append([x / s for x in g])
    e = 0.0
    for a in range(n):
        for b in range(a):
            za, zb = numbers[a], numbers[b]
            c6 = sum(w[a][i] * w[b][j] * c6tab[za, zb, i, j]
                     for i in range(len(w[a])) for j in range(len(w[b])))
            c8 = 3.0 * c6 * el[za]["r4r2"] * el[zb]["r4r2"]
            r0 = math.sqrt(c8 / c6)
            f = a1 * r0 + a2
            r = dist[a][b]
            e -= s6 * c6 / (r ** 6 + f ** 6) + s8 * c8 / (r ** 8 + f ** 8)
    return e


def check(path, cn, c6, rcov, r4r2, r0ab, elements):
    el, r0, c6tab = read(path)
    ok = True
    if sorted(el) != list(range(1, MAX_Z + 1)):
        print("FAIL: elements", sorted(el))
        ok = False
    # Every value read back equals tad's value (float32 data at float32 precision).
    for z, refs in elements:
        ok &= el[z]["rcov"] == float(rcov[z])
        ok &= np.float32(el[z]["r4r2"]) == r4r2[z].numpy()
        ok &= all(np.float32(el[z]["cnref"][k]) == cn[z, i].numpy() for k, i in enumerate(refs))
    for za, refs_a in elements:
        for zb, refs_b in elements:
            ok &= np.float32(r0[za, zb]) == np.float32(r0ab[za, zb].item())
            ok &= all(np.float32(c6tab[za, zb, i, j]) == c6[za, zb, i, j].numpy()
                      for i in refs_a for j in refs_b)
    if not ok:
        print("FAIL: values read back differ from tad")

    # Whole-model check: D3(BJ) energies from the file vs tad_dftd3.dftd3.
    # Any damping parameters serve here; these are of the size used for B3LYP.
    a1, s8, a2 = 0.3981, 1.9889, 4.4211
    param = {"s6": torch.tensor(1.0, dtype=torch.float64),
             "a1": torch.tensor(a1, dtype=torch.float64),
             "s8": torch.tensor(s8, dtype=torch.float64),
             "a2": torch.tensor(a2, dtype=torch.float64),
             "s9": torch.tensor(0.0, dtype=torch.float64)}
    molecules = {  # bohr
        "H2O": ([8, 1, 1], [[0.0, 0.0, -0.7357858611], [1.4418315287, 0.0, 0.3678929305],
                            [-1.4418315287, 0.0, 0.3678929305]]),
        "CH3F+Ne": ([6, 9, 1, 1, 1, 10], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.6], [1.94, 0.0, -0.69],
                                         [-0.97, 1.68, -0.69], [-0.97, -1.68, -0.69],
                                         [0.3, 0.2, 8.0]]),
        "LiCl/SiH4 pair": ([3, 17, 14, 1, 1, 1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.8],
                                                     [6.0, 1.0, 1.0], [7.6, 2.0, 1.5],
                                                     [4.6, 2.1, 0.8], [6.2, -0.5, 2.6],
                                                     [5.8, 0.4, -1.4]]),
        "Na Mg Al P S Ar B Be He N O": ([11, 12, 13, 15, 16, 18, 5, 4, 2, 7, 8],
                                        [[0.0, 0.0, 0.0], [5.5, 0.0, 0.0], [0.0, 5.5, 0.0],
                                         [0.0, 0.0, 5.5], [5.0, 5.0, 0.5], [-5.0, 2.0, 3.0],
                                         [2.0, -4.0, 1.0], [-3.0, -3.0, -3.0],
                                         [8.0, 8.0, 8.0], [1.0, 1.5, -4.0], [3.2, 1.5, -4.5]]),
    }
    for name, (numbers, pos) in molecules.items():
        mine = bj_energy(el, c6tab, numbers, pos, a1, s8, a2)
        theirs = d3.dftd3(torch.tensor(numbers), torch.tensor(pos, dtype=torch.float64),
                          param).sum().item()
        rel = abs(mine - theirs) / abs(theirs)
        status = "ok" if rel < 1e-6 else "FAIL"
        ok &= rel < 1e-6
        print(f"{status}: {name}: file {mine:.12e}  tad {theirs:.12e}  rel {rel:.1e}")

    # Unit check for r0ab: tad's three-body (ATM) term uses these radii, so it is
    # reproduced from the file only if r0ab is read in the right unit (bohr).
    numbers, pos = [6, 8, 7], [[0.0, 0.0, 0.0], [4.0, 0.5, 0.0], [1.0, 3.7, 0.4]]
    atm_param = {k: torch.tensor(v, dtype=torch.float64)
                 for k, v in dict(s6=0.0, s8=0.0, a1=a1, a2=a2, s9=1.0).items()}
    mine = atm_energy(el, r0, c6tab, numbers, pos)
    theirs = d3.dftd3(torch.tensor(numbers), torch.tensor(pos, dtype=torch.float64),
                      atm_param).sum().item()
    rel = abs(mine - theirs) / abs(theirs)
    status = "ok" if rel < 1e-5 else "FAIL"
    ok &= rel < 1e-5
    print(f"{status}: ATM C O N: file {mine:.12e}  tad {theirs:.12e}  rel {rel:.1e}")
    return ok


def atm_energy(el, r0, c6tab, numbers, pos):
    """Axilrod-Teller-Muto energy of one triangle (Grimme et al. 2010), rs9 = 4/3."""
    assert len(numbers) == 3
    d = lambda a, b: math.dist(pos[a], pos[b])
    cn = [sum(1.0 / (1.0 + math.exp(-16.0 * ((el[numbers[a]]["rcov"] + el[numbers[b]]["rcov"])
                                               / d(a, b) - 1.0)))
              for b in range(3) if b != a) for a in range(3)]
    w = []
    for a in range(3):
        g = [math.exp(-4.0 * (cn[a] - c) ** 2) for c in el[numbers[a]]["cnref"]]
        w.append([x / sum(g) for x in g])

    def c6(a, b):
        return sum(w[a][i] * w[b][j] * c6tab[numbers[a], numbers[b], i, j]
                   for i in range(len(w[a])) for j in range(len(w[b])))

    rab, rbc, rca = d(0, 1), d(1, 2), d(2, 0)
    cos_a = (rab ** 2 + rca ** 2 - rbc ** 2) / (2 * rab * rca)
    cos_b = (rab ** 2 + rbc ** 2 - rca ** 2) / (2 * rab * rbc)
    cos_c = (rbc ** 2 + rca ** 2 - rab ** 2) / (2 * rbc * rca)
    c9 = math.sqrt(c6(0, 1) * c6(1, 2) * c6(2, 0))
    r0bar = (r0[numbers[0], numbers[1]] * r0[numbers[1], numbers[2]]
             * r0[numbers[2], numbers[0]]) ** (1 / 3)
    rbar = (rab * rbc * rca) ** (1 / 3)
    fdamp = 1.0 / (1.0 + 6.0 * (rbar / (4.0 / 3.0 * r0bar)) ** -16)
    return c9 * (3 * cos_a * cos_b * cos_c + 1) / (rab * rbc * rca) ** 3 * fdamp


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python -I tools/extract_d3_data.py OUTPUT_PATH")
    versions = {"tad_dftd3": d3.__version__, "tad_mctc": mctc.__version__}
    if versions != EXPECTED_VERSIONS:
        sys.exit(f"expected {EXPECTED_VERSIONS}, found {versions}")
    data = extract()
    write(sys.argv[1], *data)
    if not check(sys.argv[1], *data):
        sys.exit(1)


if __name__ == "__main__":
    main()

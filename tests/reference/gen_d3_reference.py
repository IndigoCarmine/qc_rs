"""Generate DFT-D3 reference values (energy, gradient, Hessian) with dftd3 (simple-dftd3).

Usage (from the repository root):
    .venv/Scripts/pip install dftd3==1.7.0                       # Windows
    .venv/Scripts/python tests/reference/gen_d3_reference.py
    .venv/bin/pip install dftd3==1.7.0                           # Linux / macOS
    .venv/bin/python tests/reference/gen_d3_reference.py

Writes tests/reference/d3_reference.json.  The dftd3 package (LGPL) is only run
here to produce numbers; it is not shipped with or linked into qc_rs.  It is used
as a black box through its documented Python API (dftd3.interface).

* The damping parameters are given as numbers, never loaded by functional name,
  so the reference does not depend on the parameter tables inside dftd3.
* The three-body (ATM) term is switched off (s9 = 0).  Zero damping uses
  alp = 14 and rs8 = 1, BJ damping alp = 14 (unused by the two-body term).
* The real-space cutoffs (two-body, three-body, coordination number) are raised
  to CUTOFF_BOHR with DispersionModel.set_realspace_cutoff, far beyond every
  interatomic distance, so the values are the untruncated sums.  The largest
  interatomic distance of each molecule is recorded and checked against it.
* The library runs single-threaded (OMP_NUM_THREADS=1, set before import) so
  that the summation order, and hence the JSON, is reproducible bit for bit.
* D3 depends only on the geometry; the molecular charge plays no role.

Molecules: the water dimer, every tests/molecules/*.xyz, and small molecules
that together contain every element H-Ar.  Units in the JSON: Hartree, Bohr
(gradients are dE/dR in Hartree/Bohr).  The 3N x 3N Hessian
(DispersionModel.get_hessian, Hartree/Bohr^2, index 3*atom + xyz) is stored only
for the built-in molecules (GEOMS_ANG, at most 6 atoms), not for the larger
tests/molecules/*.xyz, to keep the JSON small.
"""
import glob
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import dftd3  # noqa: E402
from dftd3.interface import DispersionModel, RationalDampingParam, ZeroDampingParam  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOL_DIR = os.path.join(ROOT, "tests", "molecules")
OUT = os.path.join(ROOT, "tests", "reference", "d3_reference.json")

ANG = 1.0 / 0.529177210903  # Angstrom -> Bohr (CODATA 2018, as gen_pyscf_reference.py)
CUTOFF_BOHR = 1000.0

ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]
Z = {s: i + 1 for i, s in enumerate(ELEMENTS)}

# Damping parameters, given explicitly (Grimme et al. 2010, 2011).
PARAMS = {
    "b3lyp-d3bj": ("bj", {"s6": 1.0, "a1": 0.3981, "s8": 1.9889, "a2": 4.4211, "s9": 0.0, "alp": 14.0}),
    "cam-b3lyp-d3bj": ("bj", {"s6": 1.0, "a1": 0.3708, "s8": 2.0674, "a2": 5.4743, "s9": 0.0, "alp": 14.0}),
    "b3lyp-d3zero": ("zero", {"s6": 1.0, "rs6": 1.261, "s8": 1.703, "rs8": 1.0, "s9": 0.0, "alp": 14.0}),
    "cam-b3lyp-d3zero": ("zero", {"s6": 1.0, "rs6": 1.378, "s8": 1.217, "rs8": 1.0, "s9": 0.0, "alp": 14.0}),
}

# Geometries in Angstrom (converted to Bohr below).  Several are slightly
# distorted so that the gradients are not fixed by symmetry.
GEOMS_ANG = {
    "water_dimer": [("O", (-1.551007, -0.114520, 0.000000)), ("H", (-1.934259, 0.762503, 0.000000)),
                    ("H", (-0.599677, 0.040712, 0.000000)), ("O", (1.350625, 0.111469, 0.000000)),
                    ("H", (1.680398, -0.373741, -0.758561)), ("H", (1.680398, -0.373741, 0.758561))],
    "he2": [("He", (0.0, 0.0, 0.0)), ("He", (0.0, 0.0, 3.00))],
    "ne2": [("Ne", (0.0, 0.0, 0.0)), ("Ne", (0.0, 0.0, 3.10))],
    "ar2": [("Ar", (0.0, 0.0, 0.0)), ("Ar", (0.0, 0.0, 3.76))],
    "lih": [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.60))],
    "beh2": [("H", (0.0, 0.0, -1.33)), ("Be", (0.0, 0.0, 0.0)), ("H", (0.05, 0.0, 1.35))],
    "bh3": [("B", (0.0, 0.0, 0.0)), ("H", (0.0, 1.19, 0.0)), ("H", (1.031, -0.595, 0.0)),
            ("H", (-1.05, -0.60, 0.04))],
    "ch4": [("C", (0.0, 0.0, 0.0)), ("H", (0.66, 0.61, 0.63)), ("H", (-0.63, -0.63, 0.63)),
            ("H", (-0.63, 0.63, -0.63)), ("H", (0.63, -0.63, -0.63))],
    "nh3": [("N", (0.0, 0.0, 0.10)), ("H", (0.0, 0.94, -0.27)), ("H", (0.81, -0.47, -0.27)),
            ("H", (-0.81, -0.47, -0.25))],
    "h2o": [("O", (0.0, 0.0, 0.117)), ("H", (0.0, 0.757, -0.469)), ("H", (0.02, -0.76, -0.47))],
    "hf": [("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.92))],
    "nah": [("Na", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.89))],
    "mgh2": [("H", (0.0, 0.0, -1.70)), ("Mg", (0.0, 0.0, 0.0)), ("H", (0.0, 0.08, 1.72))],
    "alh3": [("Al", (0.0, 0.0, 0.0)), ("H", (0.0, 1.58, 0.0)), ("H", (1.368, -0.79, 0.0)),
             ("H", (-1.368, -0.79, 0.05))],
    "sih4": [("Si", (0.0, 0.0, 0.0)), ("H", (0.8545, 0.8545, 0.8545)), ("H", (-0.8545, -0.8545, 0.8545)),
             ("H", (-0.8545, 0.8545, -0.8545)), ("H", (0.88, -0.83, -0.86))],
    "ph3": [("P", (0.0, 0.0, 0.0)), ("H", (0.0, 1.18, -0.78)), ("H", (1.022, -0.59, -0.78)),
            ("H", (-1.022, -0.59, -0.80))],
    "h2s": [("S", (0.0, 0.0, 0.0)), ("H", (0.0, 0.965, -0.93)), ("H", (0.0, -0.965, -0.93))],
    "hcl": [("H", (0.0, 0.0, 0.0)), ("Cl", (0.0, 0.0, 1.27))],
}


def read_xyz(path):
    """Element symbols and Bohr coordinates of an XYZ file (Angstrom); the comment line is ignored."""
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0].split()[0])
    atoms, coords = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        atoms.append(t[0])
        coords.append([float(x) * ANG for x in t[1:4]])
    if len(atoms) != n:
        raise ValueError(f"{path}: expected {n} atoms, found {len(atoms)}")
    return atoms, coords


def molecules():
    mols = {}
    for name, geom in GEOMS_ANG.items():
        mols[name] = ([s for s, _ in geom], [[float(x) * ANG for x in xyz] for _, xyz in geom],
                      "gen_d3_reference.py")
    for path in sorted(glob.glob(os.path.join(MOL_DIR, "*.xyz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in mols:
            raise ValueError(f"{path}: name clashes with a built-in molecule")
        atoms, coords = read_xyz(path)
        mols[name] = (atoms, coords, "tests/molecules/" + os.path.basename(path))
    return mols


def damping(kind, p):
    if kind == "bj":
        return RationalDampingParam(**p)
    return ZeroDampingParam(**p)


def compute(atoms, coords, with_hessian):
    numbers = np.array([Z[s] for s in atoms])
    pos = np.array(coords)
    dmax = max((float(np.linalg.norm(pos[i] - pos[j])) for i in range(len(atoms)) for j in range(i)), default=0.0)
    if dmax >= CUTOFF_BOHR:
        raise ValueError(f"largest interatomic distance {dmax} Bohr reaches the cutoff {CUTOFF_BOHR}")
    model = DispersionModel(numbers, pos)
    model.set_realspace_cutoff(CUTOFF_BOHR, CUTOFF_BOHR, CUTOFF_BOHR)
    results = {}
    for label, (kind, p) in PARAMS.items():
        r = model.get_dispersion(damping(kind, p), grad=True)
        results[label] = {"energy": float(r["energy"]),
                          "gradient": [[float(x) for x in row] for row in r["gradient"]]}
        if with_hessian:
            hess = np.asarray(model.get_hessian(damping(kind, p))["hessian"]).reshape(3 * len(atoms), 3 * len(atoms))
            results[label]["hessian"] = [[float(x) for x in row] for row in hess]
    return dmax, results


def main():
    out_mols = {}
    for name, (atoms, coords, source) in molecules().items():
        dmax, results = compute(atoms, coords, with_hessian=name in GEOMS_ANG)
        out_mols[name] = {"source": source, "atoms": atoms, "coords_bohr": coords,
                          "max_interatomic_distance_bohr": dmax, "results": results}
        e = results["b3lyp-d3bj"]["energy"]
        print(f"{name:12s} natm={len(atoms):3d}  dmax={dmax:7.3f} Bohr  E(B3LYP-D3(BJ))={e:.12f}", flush=True)
    out = {
        "generator": "dftd3 " + dftd3.__version__ + " (simple-dftd3 Python API, dftd3.interface)",
        "dftd3_version": dftd3.__version__,
        "units": {"energy": "Hartree", "length": "Bohr", "gradient": "Hartree/Bohr (dE/dR)",
                  "hessian": "Hartree/Bohr^2 (d2E/dR dR), row/column index 3*atom + xyz"},
        "angstrom_to_bohr": ANG,
        "cutoff": {
            "how": "DispersionModel.set_realspace_cutoff(disp2, disp3, cn)",
            "disp2_bohr": CUTOFF_BOHR, "disp3_bohr": CUTOFF_BOHR, "cn_bohr": CUTOFF_BOHR,
            "note": "larger than every interatomic distance, so no pair is cut off",
        },
        "three_body": "off (s9 = 0)",
        "threads": "OMP_NUM_THREADS=1",
        "parameters": {label: {"damping": kind, **p} for label, (kind, p) in PARAMS.items()},
        "notes": "Damping parameters passed as numbers (RationalDampingParam for bj, "
                 "ZeroDampingParam for zero), not loaded by functional name. "
                 "D3 depends only on the geometry.",
        "molecules": out_mols,
    }
    with open(OUT, "w", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

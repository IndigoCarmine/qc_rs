"""Generate PySCF reference values for every molecule in tests/molecules/*.xyz.

Usage (from the repository root):
    .venv/Scripts/python tests/reference/gen_molecule_reference.py            # Windows
    .venv/bin/python tests/reference/gen_molecule_reference.py                # Linux / macOS
    ... gen_molecule_reference.py --force        # recompute everything
    ... gen_molecule_reference.py h2o BarPh      # only these molecules (file stems)

Writes tests/reference/molecules_reference.json, which tests/molecules_pyscf.rs
compares against.  To add a test molecule: drop an XYZ file into tests/molecules/
and run this script; nothing else has to be edited.

Molecules whose geometry is unchanged since the last run are not recomputed, and
entries whose XYZ file has been deleted are dropped.

XYZ conventions: coordinates in Angstrom; the charge is taken from a `charge=N`
token in the comment (second) line and is 0 if there is none.  Only closed-shell
molecules (even electron count) of H-Ar are supported by qc_rs.

PySCF is used as a black box.  The basis sets are read from data/basis/*.gbs, so
both programs use exactly the same exponents and coefficients.
"""
import glob
import json
import os
import sys

import numpy as np
import pyscf
from pyscf import gto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_pyscf_reference import ANG, BASIS, ROOT, make_mf  # noqa: E402

# (method, basis) pairs computed for every molecule.  Methods are PySCF names:
# "rhf" or an xc string ("b3lypg" = B3LYP with VWN-RPA = qc_rs "b3lyp").
METHODS = [("rhf", "6-31g"), ("b3lypg", "6-31g")]
DFT_GRID = (99, 590)  # radial points, Lebedev points; unpruned

MOL_DIR = os.path.join(ROOT, "tests", "molecules")
OUT = os.path.join(ROOT, "tests", "reference", "molecules_reference.json")


def read_xyz(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0].split()[0])
    charge = 0
    for tok in lines[1].split():
        if tok.startswith("charge="):
            charge = int(tok[len("charge="):])
    atoms, coords = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        atoms.append(t[0])
        coords.append([float(x) * ANG for x in t[1:4]])
    if len(atoms) != n:
        raise ValueError(f"{path}: expected {n} atoms, found {len(atoms)}")
    return atoms, coords, charge


def unchanged(entry, atoms, coords, charge):
    if entry is None or entry["atoms"] != atoms or entry["charge"] != charge:
        return False
    if [[r["method"], r["basis"]] for r in entry["results"]] != [list(m) for m in METHODS]:
        return False
    return np.allclose(np.array(entry["coords_bohr"]), np.array(coords), rtol=0, atol=1e-10)


def compute(name, atoms, coords, charge):
    results = []
    for method, basis in METHODS:
        mol = gto.M(atom=[[s, xyz] for s, xyz in zip(atoms, coords)], unit="Bohr", charge=charge,
                    basis={s: BASIS[basis][s] for s in set(atoms)}, cart=False, symmetry=False, verbose=0)
        mf = make_mf(mol, method, grid=DFT_GRID)
        e = mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"{name} {method}/{basis}: PySCF SCF did not converge")
        g = mf.nuc_grad_method().kernel()
        rec = {"method": method, "basis": basis, "nao": int(mol.nao), "e_tot": float(e),
               "gradient": [[float(x) for x in row] for row in g]}
        if method != "rhf":
            rec["dft_grid"] = list(DFT_GRID)
        results.append(rec)
        print(f"{name:16s} {method:8s} {basis:7s} nao={mol.nao:4d}  E={e:.10f}  max|g|={abs(g).max():.2e}", flush=True)
    return {"atoms": atoms, "charge": charge, "coords_bohr": coords, "results": results}


def main():
    args = sys.argv[1:]
    force = "--force" in args
    only = [a for a in args if not a.startswith("--")]
    old = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            old = json.load(f).get("molecules", {})
    molecules = {}
    for path in sorted(glob.glob(os.path.join(MOL_DIR, "*.xyz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        atoms, coords, charge = read_xyz(path)
        entry = old.get(name)
        keep = unchanged(entry, atoms, coords, charge)
        if (only and name not in only and entry is not None) or (keep and not force and name not in only):
            if not keep:
                print(f"{name:16s} WARNING: geometry changed but not recomputed (not selected)")
            else:
                print(f"{name:16s} unchanged, kept")
            molecules[name] = entry
            continue
        molecules[name] = compute(name, atoms, coords, charge)
    for name in sorted(set(old) - set(molecules)):
        print(f"{name:16s} XYZ file is gone, entry dropped")
    out = {
        "generator": "PySCF " + pyscf.__version__,
        "notes": "Atomic units (Hartree, Bohr); gradients are dE/dR in Hartree/Bohr. "
                 "One entry per tests/molecules/<name>.xyz.",
        "molecules": molecules,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

"""Generate reference values with PySCF (used only as a black box).

Usage (from the repository root):
    .venv/Scripts/python tests/reference/gen_pyscf_reference.py

Writes tests/reference/pyscf_reference.json.

The basis sets are read from data/basis/*.gbs (Basis Set Exchange, Gaussian94
format) and handed to PySCF explicitly, so that both programs use exactly the
same exponents and contraction coefficients.  All coordinates are in Bohr.
"""
import json
import os
import sys

import numpy as np
import pyscf
from pyscf import gto, scf, dft
from pyscf.geomopt import geometric_solver

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANG = 1.0 / 0.529177210903  # Angstrom -> Bohr (CODATA 2018)
LSYM = {"S": 0, "P": 1, "D": 2, "F": 3}


def load_gbs(name):
    """Parse a Gaussian94 basis file into PySCF's internal basis format."""
    path = os.path.join(ROOT, "data", "basis", name + ".gbs")
    out = {}
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("!") or line == "****":
            i += 1
            continue
        tok = line.split()
        if len(tok) == 2 and tok[1] == "0":
            sym = tok[0]
            shells = []
            i += 1
            while lines[i].strip() != "****":
                kind, nprim, _scale = lines[i].split()
                nprim = int(nprim)
                rows = []
                for k in range(nprim):
                    rows.append([float(x.replace("D", "E")) for x in lines[i + 1 + k].split()])
                i += 1 + nprim
                if kind == "SP":
                    shells.append([0] + [[r[0], r[1]] for r in rows])
                    shells.append([1] + [[r[0], r[2]] for r in rows])
                else:
                    shells.append([LSYM[kind]] + [[r[0], r[1]] for r in rows])
            out[sym] = shells
        i += 1
    return out


BASIS = {n: load_gbs(f) for n, f in [("sto-3g", "sto-3g"), ("6-31g", "6-31g"),
                                     ("6-31g*", "6-31gs"), ("6-31g**", "6-31gss")]}

# Geometries in Angstrom (converted to Bohr below).  Deliberately not at
# equilibrium so that the gradients are non-trivial; no symmetry in "h2o_dist".
GEOMS_ANG = {
    "h2": [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))],
    "h2o": [("O", (0.0, 0.0, 0.1173)), ("H", (0.0, 0.7572, -0.4692)), ("H", (0.0, -0.7572, -0.4692))],
    "h2o_dist": [("O", (0.02, -0.01, 0.10)), ("H", (0.05, 0.80, -0.45)), ("H", (-0.10, -0.70, -0.52))],
    "nh3": [("N", (0.0, 0.0, 0.10)), ("H", (0.0, 0.95, -0.25)), ("H", (0.82, -0.47, -0.25)),
            ("H", (-0.82, -0.47, -0.25))],
    "ch4": [("C", (0.0, 0.0, 0.0)), ("H", (0.63, 0.63, 0.63)), ("H", (-0.63, -0.63, 0.63)),
            ("H", (-0.63, 0.63, -0.63)), ("H", (0.63, -0.63, -0.63))],
    "h2co": [("C", (0.0, 0.0, -0.53)), ("O", (0.0, 0.0, 0.68)), ("H", (0.0, 0.94, -1.11)),
             ("H", (0.0, -0.94, -1.11))],
    "hcl": [("H", (0.0, 0.0, 0.0)), ("Cl", (0.0, 0.0, 1.30))],
}
GEOMS = {k: [(s, tuple(float(x) * ANG for x in xyz)) for s, xyz in v] for k, v in GEOMS_ANG.items()}

FINE_GRID = (150, 1202)
OPT_GRID = (99, 590)


def make_mol(geom, basis, cart):
    syms = {s for s, _ in geom}
    return gto.M(atom=[[s, xyz] for s, xyz in geom], unit="Bohr",
                 basis={s: BASIS[basis][s] for s in syms}, cart=cart, symmetry=False, verbose=0)


def make_mf(mol, method, grid=FINE_GRID):
    if method == "rhf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = method
        mf.grids.atom_grid = grid
        mf.grids.prune = None
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    return mf


def single_point(geom_name, basis, method, cart=False, grad=True):
    mol = make_mol(GEOMS[geom_name], basis, cart)
    mf = make_mf(mol, method)
    e = mf.kernel()
    assert mf.converged
    s = mol.intor("int1e_ovlp")
    t = mol.intor("int1e_kin")
    v = mol.intor("int1e_nuc")
    dm = mf.make_rdm1()
    rec = {
        "name": geom_name, "basis": basis, "method": method, "cartesian": bool(cart),
        "atoms": [s_ for s_, _ in GEOMS[geom_name]],
        "coords_bohr": [list(xyz) for _, xyz in GEOMS[geom_name]],
        "nao": int(mol.nao),
        "e_tot": float(e),
        "e_nuc": float(mol.energy_nuc()),
        "e_kin": float(np.einsum("ij,ji", dm, t)),
        "e_nuc_attr": float(np.einsum("ij,ji", dm, v)),
        "mo_energy": [float(x) for x in mf.mo_energy],
        "ovlp_eigvals": [float(x) for x in np.linalg.eigvalsh(s)],
        "dipole_au": [float(x) for x in mf.dip_moment(unit="AU", verbose=0)],
        "scf_summary": {k: float(v_) for k, v_ in mf.scf_summary.items()},
    }
    if grad:
        g = mf.nuc_grad_method().kernel()
        rec["gradient"] = [[float(x) for x in row] for row in g]
    print(f"SP  {geom_name:9s} {method:16s} {basis:8s} cart={cart}  E={e:.10f}", flush=True)
    return rec


def optimize(geom_name, basis, method):
    mol = make_mol(GEOMS[geom_name], basis, False)
    mf = make_mf(mol, method, grid=OPT_GRID)
    params = {"convergence_energy": 1e-9, "convergence_grms": 3e-6, "convergence_gmax": 4.5e-6,
              "convergence_drms": 1.2e-5, "convergence_dmax": 1.8e-5}
    mol_eq = geometric_solver.optimize(mf, maxsteps=200, **params)
    mf2 = make_mf(mol_eq, method, grid=OPT_GRID)
    e = mf2.kernel()
    g = mf2.nuc_grad_method().kernel()
    c = mol_eq.atom_coords()
    n = len(c)
    dist = [[float(np.linalg.norm(c[i] - c[j])) for j in range(n)] for i in range(n)]
    print(f"OPT {geom_name:9s} {method:16s} {basis:8s} E={e:.10f} max|g|={abs(g).max():.2e}", flush=True)
    return {
        "name": geom_name, "basis": basis, "method": method, "cartesian": False,
        "atoms": [s for s, _ in GEOMS[geom_name]],
        "initial_coords_bohr": [list(xyz) for _, xyz in GEOMS[geom_name]],
        "dft_grid": list(OPT_GRID),
        "e_opt": float(e),
        "final_coords_bohr": [[float(x) for x in row] for row in c],
        "final_distances_bohr": dist,
        "final_max_abs_gradient": float(abs(g).max()),
    }


def main():
    sp = []
    # --- RHF ---
    sp.append(single_point("h2", "sto-3g", "rhf"))
    sp.append(single_point("h2o", "sto-3g", "rhf"))
    sp.append(single_point("h2o", "6-31g", "rhf"))
    sp.append(single_point("h2o", "6-31g*", "rhf"))
    sp.append(single_point("h2o", "6-31g*", "rhf", cart=True))
    sp.append(single_point("h2o_dist", "6-31g*", "rhf"))
    sp.append(single_point("h2o", "6-31g**", "rhf"))
    sp.append(single_point("nh3", "6-31g*", "rhf"))
    sp.append(single_point("ch4", "6-31g*", "rhf"))
    sp.append(single_point("h2co", "6-31g*", "rhf"))
    sp.append(single_point("hcl", "6-31g*", "rhf"))
    # --- RKS: individual functional components, then the hybrids ---
    for xc in ["slater,", "slater,vwn5", "slater,vwn_rpa", "b88,", "b88,lyp", "b3lyp5", "b3lypg"]:
        sp.append(single_point("h2o", "6-31g", xc))
    sp.append(single_point("h2o_dist", "6-31g", "b3lypg"))
    sp.append(single_point("h2o", "6-31g*", "b3lypg"))
    sp.append(single_point("nh3", "6-31g", "b3lypg"))
    sp.append(single_point("ch4", "6-31g", "b3lypg"))
    sp.append(single_point("h2co", "6-31g", "b3lypg"))
    # --- Range-separated hybrid ---
    for geom, basis in [("h2o", "6-31g"), ("h2o_dist", "6-31g"), ("h2o", "6-31g*"), ("nh3", "6-31g"),
                        ("ch4", "6-31g"), ("h2co", "6-31g")]:
        sp.append(single_point(geom, basis, "camb3lyp"))
    # What does the plain name "b3lyp" mean in this PySCF version?
    sp.append(single_point("h2o", "6-31g", "b3lyp", grad=False))

    opt = [
        optimize("h2o_dist", "6-31g*", "rhf"),
        optimize("nh3", "6-31g*", "rhf"),
        optimize("h2o_dist", "6-31g", "b3lypg"),
        optimize("nh3", "6-31g", "b3lypg"),
        optimize("h2o_dist", "6-31g", "camb3lyp"),
    ]

    out = {
        "generator": "PySCF " + pyscf.__version__,
        "python": sys.version.split()[0],
        "dft_grid_single_point": list(FINE_GRID),
        "notes": "All quantities in atomic units (Hartree, Bohr). Gradients are dE/dR in Hartree/Bohr.",
        "single_points": sp,
        "optimizations": opt,
    }
    path = os.path.join(ROOT, "tests", "reference", "pyscf_reference.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()

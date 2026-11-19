"""Generate reference values with PySCF (used only as a black box).

Usage (from the repository root):
    .venv/Scripts/python tests/reference/gen_pyscf_reference.py

Writes tests/reference/pyscf_reference.json.  With --hessians-only, only the
"hessians" section is (re)computed and the other sections of the existing JSON
are kept unchanged (a full run reproduces them only to about 1e-9).  With
--append-f, only the F_SINGLE_POINTS entries not yet present are computed and
appended to the end of "single_points"; everything else is kept unchanged.

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
from pyscf.hessian import thermo

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
                                     ("6-31g*", "6-31gs"), ("6-31g**", "6-31gss"),
                                     ("6-311g**", "6-311gss"), ("6-31g(2df,p)", "6-31g_2df_p"),
                                     ("6-311g(2df,2pd)", "6-311g_2df_2pd")]}

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
    # Linear molecule for the vibrational analysis (5 removed external modes).
    "hcn": [("H", (0.0, 0.0, -1.10)), ("C", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.12))],
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


HESS_STEP = 5e-3  # Bohr; the displacement h of eq:hess-fd, used here for the dipole derivatives
THERMO_CONDITIONS = [(298.15, 101325.0), (500.0, 2.0e5)]  # (T in K, p in Pa)
# eq:ir-intensity: A [km/mol] = N_A e^2 / (12 eps0 c^2 m_u) * 1e-3 * |dmu/dQ|^2 [e^2/u]  (CODATA 2018)
_NA, _E, _EPS0, _C, _MU = 6.02214076e23, 1.602176634e-19, 8.8541878128e-12, 299792458.0, 1.66053906660e-27
IR_KM_MOL = _NA * _E ** 2 / (12.0 * _EPS0 * _C ** 2 * _MU) * 1e-3


def _wavenumber(f):
    """PySCF returns imaginary frequencies as imaginary numbers; store them as negative wavenumbers."""
    f = complex(f)
    return f.real if f.imag == 0.0 else -abs(f.imag)


def _thermo_values(t):
    out = {}
    for k, (v, _unit) in t.items():
        if k.startswith("qrrho") or k in ("temperature", "pressure"):
            continue
        if k == "sym_number":
            out[k] = int(v)
        elif np.ndim(v) == 0:
            out[k] = float(v)
        else:  # rot_const: infinite for the zero moment of a linear molecule -> null
            out[k] = [float(x) if np.isfinite(x) else None for x in v]
    return out


def hessian_entry(geom_name, basis, method):
    """Optimise (geomeTRIC, same parameters as optimize()), then the analytic Hessian, harmonic analysis
    and thermochemistry from PySCF.  For DFT the fine grid is used for the optimisation as well.

    The IR intensities are not a PySCF result (this PySCF has no IR module): PySCF's dipole moments
    (mf.dip_moment) at +-h Cartesian displacements are central-differenced, contracted with the
    normal modes from PySCF's thermo.harmonic_analysis, and converted with *our* formula
    (docs/theory.typ eq:ir-intensity, km/mol).
    """
    grid = FINE_GRID
    mol = make_mol(GEOMS[geom_name], basis, False)
    mf = make_mf(mol, method, grid=grid)
    params = {"convergence_energy": 1e-9, "convergence_grms": 3e-6, "convergence_gmax": 4.5e-6,
              "convergence_drms": 1.2e-5, "convergence_dmax": 1.8e-5}
    mol_eq = geometric_solver.optimize(mf, maxsteps=200, **params)
    mf = make_mf(mol_eq, method, grid=grid)
    e = mf.kernel()
    assert mf.converged
    g = mf.nuc_grad_method().kernel()
    h4 = mf.Hessian().kernel()  # (natm, natm, 3, 3)
    n = mol_eq.natm
    hess = h4.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    masses = mol_eq.atom_mass_list(isotope_avg=True)
    ha = thermo.harmonic_analysis(mol_eq, h4)
    # norm_mode[k] is the Cartesian displacement l_k / sqrt(m) of the unit mass-weighted mode l_k.
    modes = ha["norm_mode"].reshape(-1, 3 * n)
    lnorm = np.linalg.norm(modes * np.sqrt(np.repeat(masses, 3)), axis=1)
    assert np.allclose(lnorm, 1.0, atol=1e-10), lnorm

    # Dipole derivatives by central differences of PySCF's dipole moment (tightly converged SCF).
    coords = mol_eq.atom_coords()
    syms = [mol_eq.atom_symbol(i) for i in range(n)]
    dm0 = mf.make_rdm1()
    dmu = np.zeros((3 * n, 3))
    for j in range(3 * n):
        mu = []
        for sgn in (1.0, -1.0):
            c = coords.copy()
            c[j // 3, j % 3] += sgn * HESS_STEP
            mf_d = make_mf(make_mol(list(zip(syms, map(tuple, c))), basis, False), method, grid=grid)
            mf_d.conv_tol = 1e-13
            mf_d.conv_tol_grad = 1e-10
            mf_d.kernel(dm0=dm0)
            assert mf_d.converged
            mu.append(mf_d.dip_moment(unit="AU", verbose=0))
        dmu[j] = (mu[0] - mu[1]) / (2.0 * HESS_STEP)
    dmu_dq = modes @ dmu  # (nmode, 3), e / sqrt(u)
    ir = IR_KM_MOL * np.sum(dmu_dq ** 2, axis=1)

    freq = [_wavenumber(f) for f in ha["freq_wavenumber"]]
    th = []
    for temp, pres in THERMO_CONDITIONS:
        rec = {"temperature_k": temp, "pressure_pa": pres}
        rec.update(_thermo_values(thermo.thermo(mf, ha["freq_au"], temp, pres)))
        th.append(rec)
    print(f"HES {geom_name:9s} {method:16s} {basis:8s} E={e:.10f} max|g|={abs(g).max():.2e} "
          f"freq={np.round(freq, 1)} IR={np.round(ir, 2)}", flush=True)
    return {
        "name": geom_name, "basis": basis, "method": method, "cartesian": False,
        "atoms": syms,
        "initial_coords_bohr": [list(xyz) for _, xyz in GEOMS[geom_name]],
        "dft_grid": list(grid),
        "coords_bohr": [[float(x) for x in row] for row in coords],
        "masses_u": [float(x) for x in masses],
        "e_tot": float(e),
        "max_abs_gradient": float(abs(g).max()),
        "hessian": [[float(x) for x in row] for row in hess],
        "freq_wavenumber": [float(x) for x in freq],
        "norm_mode_cartesian": [[float(x) for x in row] for row in modes],
        "dipole_derivative_au": [[float(x) for x in row] for row in dmu],
        "ir_displacement_bohr": HESS_STEP,
        "ir_intensity_km_mol": [float(x) for x in ir],
        "thermo": th,
    }


def hessians():
    return [
        hessian_entry("h2o_dist", "6-31g*", "rhf"),
        hessian_entry("nh3", "6-31g*", "rhf"),
        hessian_entry("hcn", "6-31g*", "rhf"),
        hessian_entry("h2o_dist", "6-31g", "b3lypg"),
        hessian_entry("nh3", "6-31g", "b3lypg"),
        hessian_entry("h2o_dist", "6-31g", "camb3lyp"),
    ]


HESSIANS_NOTES = (
    "Analytic Hessian (Eh/Bohr^2, 3N x 3N, row/column 3*atom+xyz), frequencies (cm^-1, imaginary as "
    "negative), normal modes (Cartesian, = mass-weighted unit vector / sqrt(m)) and thermochemistry "
    "(thermo.thermo, Eh and Eh/K; sym_number chosen by PySCF) from PySCF at the geomeTRIC-optimised "
    "structure (same convergence as 'optimizations', fine grid for DFT). Masses are PySCF's "
    "isotope-averaged defaults. ir_intensity_km_mol is NOT a PySCF result: PySCF's dipole moments "
    "at +-ir_displacement_bohr Cartesian displacements are central-differenced and contracted with "
    "PySCF's normal modes, then converted with docs/theory.typ eq:ir-intensity.")
OUT_PATH = os.path.join(ROOT, "tests", "reference", "pyscf_reference.json")


def main_hessians_only():
    """Add/replace only 'hessians' in the existing JSON.  A full regeneration reproduces the older
    sections only to ~1e-9 (SCF convergence), so they are kept exactly as committed."""
    with open(OUT_PATH) as f:
        out = json.load(f)
    out["hessians_notes"] = HESSIANS_NOTES
    out["hessians"] = hessians()
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT_PATH)


# Single points with f functions (and one 6-311G** case), added after the original list.  The
# 6-31G(2df,p) data (H/He with two P shells) is read from data/basis like every other set, so
# PySCF uses exactly our exponents and coefficients, not its own set of the same name.
F_SINGLE_POINTS = [
    ("h2o", "6-311g**", "rhf", False),
    ("h2o", "6-31g(2df,p)", "rhf", False),
    ("h2o", "6-31g(2df,p)", "rhf", True),
    ("h2o_dist", "6-31g(2df,p)", "rhf", False),
    ("hcl", "6-31g(2df,p)", "rhf", False),
    ("h2o", "6-311g(2df,2pd)", "rhf", False),
    ("h2o", "6-31g(2df,p)", "b3lypg", False),
    ("h2o", "6-31g(2df,p)", "camb3lyp", False),
]


def f_single_points():
    return [single_point(geom, basis, method, cart=cart) for geom, basis, method, cart in F_SINGLE_POINTS]


def _key(rec):
    return (rec["name"], rec["basis"], rec["method"], rec["cartesian"])


def main_append_f():
    """Append the F_SINGLE_POINTS entries missing from the existing JSON to the end of
    'single_points'.  A full regeneration reproduces the committed values only to ~1e-9 (SCF
    convergence), so everything already present is kept exactly as it is; this is checked below."""
    with open(OUT_PATH) as f:
        old_text = f.read()
    out = json.loads(old_text)
    old = json.loads(old_text)
    have = {_key(r) for r in out["single_points"]}
    n_old = len(out["single_points"])
    for geom, basis, method, cart in F_SINGLE_POINTS:
        if (geom, basis, method, cart) not in have:
            out["single_points"].append(single_point(geom, basis, method, cart=cart))
    # Existing content must be untouched: same values, and the serialised old prefix is identical.
    assert out["single_points"][:n_old] == old["single_points"]
    assert {k: v for k, v in out.items() if k != "single_points"} == \
        {k: v for k, v in old.items() if k != "single_points"}
    assert json.dumps(old, indent=1) == old_text.replace("\r\n", "\n"), "existing JSON is not in canonical form"
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT_PATH} ({len(out['single_points']) - n_old} single points appended)")


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
    sp.extend(f_single_points())

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
        "hessians_notes": HESSIANS_NOTES,
        "hessians": hessians(),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    if "--hessians-only" in sys.argv[1:]:
        main_hessians_only()
    elif "--append-f" in sys.argv[1:]:
        main_append_f()
    else:
        main()

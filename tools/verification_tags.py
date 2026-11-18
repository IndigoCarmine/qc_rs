"""Maintain the literature-verification tags in the source code.

Every place where the code cites an equation of docs/theory.typ ("eq:label") can
carry one of these tags at the end of the comment line:

    UNCHECKED(Key)          the formula has a literature source (docs/papers/README.md),
                            but nobody has compared the code with that paper yet
    VERIFIED(Key, where)    compared with the paper; "where" is the equation/table/page
    DERIVED                 no literature source: derived in docs/theory.typ and checked
                            numerically (see the tests)

Usage (from the repository root):
    python tools/verification_tags.py init              # add UNCHECKED/DERIVED where no tag exists yet
    python tools/verification_tags.py verify eq:b88 Becke1988 "Eq. (8)"
                                                        # UNCHECKED(Becke1988) -> VERIFIED(...) on lines citing eq:b88
    python tools/verification_tags.py status            # counts per key

Only the first citation of a label in each file is tagged, to keep the noise down.
"""
import glob
import re
import sys

# Spec label -> literature key(s). Keys are explained in docs/papers/README.md.
SOURCES = {
    "bohr": ["CODATA2018"],
    "vnn": ["Szabo1996"], "vnn-grad": ["Szabo1996"],
    "prim": ["Helgaker2000"], "prim-norm": ["Helgaker2000"], "contracted": ["Helgaker2000"],
    "contr-norm": ["Helgaker2000"], "cart-norm": ["Helgaker2000"], "sph-d": ["Schlegel1995"],
    "gpt": ["McMurchie1978"], "hermite-exp": ["McMurchie1978"], "E-rec": ["McMurchie1978"],
    "ovlp-1d": ["McMurchie1978"], "ovlp": ["McMurchie1978"], "kin-1d": ["McMurchie1978"],
    "kin": ["McMurchie1978"], "dip-1d": ["McMurchie1978"], "R-rec": ["McMurchie1978"],
    "nuc-prim": ["McMurchie1978"], "nuc": ["Szabo1996"], "eri": ["McMurchie1978"],
    "dipole": ["Szabo1996"],
    "boys": ["Helgaker2000"], "boys-series": ["Helgaker2000"], "boys-down": ["Helgaker2000"],
    "boys-up": ["Helgaker2000"],
    "eri-sym": ["Szabo1996"], "schwarz": ["Haser1989"],
    "density": ["Roothaan1951"], "jk": ["Roothaan1951"], "fock-rhf": ["Roothaan1951"],
    "roothaan": ["Roothaan1951"], "e-rhf": ["Roothaan1951"],
    "orth": ["Szabo1996"], "jacobi": ["Golub2013"],
    "sad": ["VanLenthe2006"], "sad-occ": ["VanLenthe2006"],
    "diis-err": ["Pulay1982"], "diis": ["Pulay1980"],
    "e-rks": ["Johnson1993", "Kohn1965"], "fock-rks": ["Johnson1993", "Kohn1965"], "rho": ["Johnson1993"],
    "exc": ["Johnson1993", "Kohn1965"], "vxc": ["Johnson1993"], "ao-deriv": ["Helgaker2000"],
    "radial": ["Becke1988grid"], "becke-mu": ["Becke1988grid"], "becke-a": ["Becke1988grid"],
    "becke-s": ["Becke1988grid"], "becke-w": ["Becke1988grid"],
    "angular": ["Press2007"], "gauss-legendre": ["Press2007"],
    "dual": ["Griewank2008"],
    "slater": ["Dirac1930", "Becke1988", "Kohn1965"], "b88": ["Becke1988"], "vwn": ["Vosko1980"],
    "lyp": ["Lee1988", "Miehlich1989"],
    "cam-split": ["Yanai2004"], "cam-k": ["Yanai2004"], "cam-fock": ["Yanai2004"],
    "cam-grad": [], "cam-b3lyp": ["Yanai2004"], "cam-b88": ["Yanai2004"],
    "cam-K": ["Iikura2001", "Becke1988"], "cam-a": ["Tawada2004"], "cam-att": ["Tawada2004"],
    "eri-erf": ["Adamson1999"],
    "cam-series": [],  # derived here
    "grad": ["Pulay1969"], "W": ["Pulay1969"], "center-deriv": ["Pulay1969"], "dS": ["Pulay1969"],
    "hf-force": ["Pulay1969"], "gamma": ["Pulay1969"], "xc-grad": ["Johnson1993"],
    "bfgs": ["Fletcher1970"],
}

FILES = sorted(glob.glob("src/*.rs") + glob.glob("src/functional/*.rs"))
LABEL = re.compile(r"eq:([A-Za-z0-9-]+)")
TAG = re.compile(r"(UNCHECKED|VERIFIED)\(([^(),]+)(?:,[^()]*(?:\([^()]*\)[^()]*)*)?\)|DERIVED")


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def split(line):
    body = line.rstrip("\r\n")
    return body, line[len(body):]


def init():
    added = 0
    for path in FILES:
        seen = set()
        out = []
        for line in read(path).splitlines(keepends=True):
            body, eol = split(line)
            if "//" in body:
                comment = body[body.index("//"):]
                tags = []
                for label in LABEL.findall(comment):
                    label = label.rstrip("-")
                    if label not in SOURCES or label in seen:
                        continue
                    seen.add(label)
                    keys = SOURCES[label]
                    for t in (["DERIVED"] if not keys else [f"UNCHECKED({k})" for k in keys]):
                        key = t if t == "DERIVED" else t[len("UNCHECKED("):-1]
                        already = any((m.group(0) == "DERIVED") if key == "DERIVED" else (m.group(2) == key)
                                      for m in TAG.finditer(comment))
                        if not already and t not in tags:
                            tags.append(t)
                if tags:
                    body = body + " " + " ".join(tags)
                    added += len(tags)
            out.append(body + eol)
        write(path, "".join(out))
    print(f"added {added} tags")


def verify(label, key, where):
    label = label.removeprefix("eq:")
    changed = 0
    for path in FILES:
        out = []
        for line in read(path).splitlines(keepends=True):
            body, eol = split(line)
            if f"UNCHECKED({key})" in body and re.search(rf"eq:{re.escape(label)}(?![A-Za-z0-9-])", body):
                body = body.replace(f"UNCHECKED({key})", f"VERIFIED({key}, {where})")
                changed += 1
            out.append(body + eol)
        write(path, "".join(out))
    print(f"eq:{label}: {changed} tag(s) set to VERIFIED({key}, {where})")
    if changed == 0:
        sys.exit(1)


def status():
    counts = {}
    for path in FILES:
        for m in TAG.finditer(read(path)):
            kind = "DERIVED" if m.group(0) == "DERIVED" else m.group(1)
            key = "-" if kind == "DERIVED" else m.group(2)
            counts[(key, kind)] = counts.get((key, kind), 0) + 1
    for (key, kind), n in sorted(counts.items()):
        print(f"{key:16s} {kind:10s} {n}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "init":
        init()
    elif cmd == "verify" and len(sys.argv) == 5:
        verify(*sys.argv[2:5])
    elif cmd == "status":
        status()
    else:
        print(__doc__)
        sys.exit(2)

"""Generate audited sparse Wick coefficient evaluation from frozen constants."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/k3_v29"
dest = root / "estimators/ablation_v30_009"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
spec = importlib.util.spec_from_file_location("wickconst", parent / "estimator.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
expressions = []
for const, exponent, p1, p2 in m.WICK_UNIFIED:

    def poly(coefs):
        terms = []
        for degree, coef in enumerate(coefs):
            if coef:
                base = "ones_n" if degree == 0 else "alpha" if degree == 1 else f"a{degree}"
                terms.append(base if coef == 1 else f"({coef!r} * {base})")
        return " + ".join(terms)

    pieces = []
    for coefs, density in [(p1, "phi"), (p2, "Phi")]:
        p = poly(coefs)
        if p:
            pieces.append(f"(({p}) * {density})")
    scale = (
        "ones_n"
        if exponent == 0
        else "sigma"
        if exponent == 1
        else f"s{exponent}"
        if exponent > 0
        else "inv"
        if exponent == -1
        else f"i{-exponent}"
    )
    expr = " + ".join(pieces) or "zeros_n"
    expressions.append(f"(({expr}) * {const!r}) * {scale}")
source = (parent / "estimator.py").read_text()
old = "            W_all = ((APOW @ C1) * phic + (APOW @ C2) * Phic) * (SB @ SELC)"
new = (
    "            W_all = fnp.stack([\n"
    + ",\n".join("                " + expr for expr in expressions)
    + "\n            ], axis=1)"
)
assert source.count(old) == 1
s = source.replace(old, new)
(dest / "estimator.py").write_text(s)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
    sha256=hashlib.sha256(s.encode()).hexdigest(),
    ablation_key="C09",
    ablation_version="v30.009",
    modifications=[
        "Generated sparse Wick polynomial expressions from original consts; legal flopscope only"
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")
(dest / "wick_expressions.json").write_text(json.dumps(expressions, indent=2) + "\n")

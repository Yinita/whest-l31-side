import hashlib
import json
import shutil
from pathlib import Path

root = Path.cwd()
parent = root / "estimators/k3_v29"
dest = root / "estimators/ablation_v30_006"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists(), "Candidate is immutable; use a new version"
s = (parent / "estimator.py").read_text()
old = """        MP = fnp.matmul(Z_st, fnp.swapaxes(L_st, -1, -2), out=bufs["mp"][:k])   # V27: BLAS path
        fnp.multiply(MP, P_st, out=MP)"""
new = """        if need_d21:
            MP = fnp.matmul(Z_st, fnp.swapaxes(L_st, -1, -2), out=bufs["mp"][:k])
            fnp.multiply(MP, P_st, out=MP)
        else:
            # C06 exact M-leg diagonal contraction; Sb/Eb keep their old path.
            PPL_diag = fnp.matmul(PP, L_st, out=bufs["ppl"][:k])
            M_diag = fnp.einsum("kiq,kiq->i", PPL_diag, Z_st)
            MP = fnp.zeros_like(PP)"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """              + fnp.einsum("kij,kij->i", MP, P_st))"""
new = (
    old
    + """
        if not need_d21:
            D3 = D3 + M_diag"""
)
assert s.count(old) == 1
s = s.replace(old, new)
(dest / "estimator.py").write_text(s)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256((parent / "estimator.py").read_bytes()).hexdigest(),
    sha256=hashlib.sha256(s.encode()).hexdigest(),
    ablation_key="C06",
    ablation_version="v30.006",
    modifications=[
        "Final-layer M leg exact diagonal contraction PP@L dot Z; Sb/Eb and feedback unchanged"
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

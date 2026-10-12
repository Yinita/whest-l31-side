import hashlib
import json
import shutil
from pathlib import Path

root = Path.cwd()
parent = root / "estimators/k3_v29"
dest = root / "estimators/ablation_v30_005"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists(), "Candidate is immutable; use a new version"
s = (parent / "estimator.py").read_text()
old = '        t1 = fnp.einsum("kiq,kcq->ic", PPL, Z_st)'
new = """        # C05: contract the same k,q axes using explicit flattened columns.
        left = fnp.reshape(fnp.swapaxes(PPL, 0, 1), (n, -1))
        right = fnp.reshape(fnp.swapaxes(Z_st, 0, 1), (n, -1))
        t1 = fnp.matmul(left, right.T)"""
assert s.count(old) == 1
(dest / "estimator.py").write_text(s.replace(old, new))
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256(s.encode()).hexdigest(),
    sha256=hashlib.sha256((dest / "estimator.py").read_bytes()).hexdigest(),
    ablation_key="C05",
    ablation_version="v30.005",
    modifications=["explicit swapaxes/reshape and matmul for PPL-Z contraction; copies counted"],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

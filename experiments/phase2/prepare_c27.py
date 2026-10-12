"""Drop historical scratch pool when the MLP shape changes."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/k3_v29"
dest = root / "estimators/ablation_v30_027"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
s = (parent / "estimator.py").read_text()
old = """        if pool is None or pool.dtype != f32:
            pool = _Pool(f32)
            self._pool = pool"""
new = """        if pool is None or pool.dtype != f32 or getattr(self, "_pool_shape", None) != (n, L):
            pool = _Pool(f32)
            self._pool = pool
            self._pool_shape = (n, L)"""
assert s.count(old) == 1
s = s.replace(old, new)
(dest / "estimator.py").write_text(s)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256((parent / "estimator.py").read_bytes()).hexdigest(),
    sha256=hashlib.sha256(s.encode()).hexdigest(),
    ablation_key="C27",
    ablation_version="v30.027",
    modifications=["Recreate pooled scratch on width/depth change; no GC manipulation"],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

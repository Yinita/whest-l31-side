"""Plain-only pooled Winograd; original hub and leaf_hub unchanged."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/k3_v29"
dest = root / "estimators/ablation_v30_002"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
s = (parent / "estimator.py").read_text()
start = s.index("    def mm(self, X, Y, out, lev):")
end = s.index("    def _leaf_mm", start)
new = """    def mm(self, X, Y, out, lev):
        bx, P, m, kd = X.shape
        by, w = Y.shape[0], Y.shape[3]
        if not self._ok(m, kd, w, lev):
            fnp.matmul(X, Y, out=out)
            return
        h, q, v = m // 2, kd // 2, w // 2
        a11, a12, a21, a22 = X[..., :h, :q], X[..., :h, q:], X[..., h:, :q], X[..., h:, q:]
        b11, b12, b21, b22 = Y[..., :q, :v], Y[..., :q, v:], Y[..., q:, :v], Y[..., q:, v:]
        left, left4 = self._buf2(("WL", P, h, q), (bx, 7, P, h, q))
        right, right4 = self._buf2(("WR", P, q, v), (by, 7, P, q, v))
        fnp.copyto(left[:, 0], a11)
        fnp.copyto(left[:, 1], a12)
        fnp.add(a21, a22, out=left[:, 4])
        fnp.subtract(left[:, 4], a11, out=left[:, 5])
        fnp.subtract(a11, a21, out=left[:, 6])
        fnp.subtract(a12, left[:, 5], out=left[:, 2])
        fnp.copyto(left[:, 3], a22)
        fnp.copyto(right[:, 0], b11)
        fnp.copyto(right[:, 1], b21)
        fnp.copyto(right[:, 2], b22)
        fnp.subtract(b12, b11, out=right[:, 4])
        fnp.subtract(b22, right[:, 4], out=right[:, 5])
        fnp.subtract(b22, b12, out=right[:, 6])
        fnp.subtract(right[:, 5], b21, out=right[:, 3])
        products, products4 = self._buf2(("WM", P, h, v), (by, 7, P, h, v))
        self.mm(left4, right4, products4, lev - 1)
        c11, c12 = out[..., :h, :v], out[..., :h, v:]
        c21, c22 = out[..., h:, :v], out[..., h:, v:]
        fnp.add(products[:, 0], products[:, 1], out=c11)
        u2 = self._buf(("WU2", P, h, v), (by, P, h, v))
        u3 = self._buf(("WU3", P, h, v), (by, P, h, v))
        fnp.add(products[:, 0], products[:, 5], out=u2)
        fnp.add(u2, products[:, 6], out=u3)
        fnp.add(u2, products[:, 4], out=c12)
        fnp.add(c12, products[:, 2], out=c12)
        fnp.subtract(u3, products[:, 3], out=c21)
        fnp.add(u3, products[:, 4], out=c22)

"""
s = s[:start] + new + s[end:]
(dest / "estimator.py").write_text(s)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256((parent / "estimator.py").read_bytes()).hexdigest(),
    sha256=hashlib.sha256(s.encode()).hexdigest(),
    ablation_key="C02",
    ablation_version="v30.002",
    modifications=[
        "Plain mm pooled Winograd15-add schedule; original hub retained; conservative separate products pool"
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

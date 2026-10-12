"""Combine final two thin feedback GEMMs, preserve GY/GX construction."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/k3_v29"
dest = root / "estimators/ablation_v30_004"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
s = (parent / "estimator.py").read_text()
old = """                fnp.matmul(GYR, fnp.swapaxes(F2, -1, -2), out=Xt)
                D21_fb = fnp.sum(Xt, axis=0, out=bufs["d21fb"])"""
assert s.count(old) == 1
s = s.replace(old, "")
old = """                fnp.matmul(GXR, fnp.swapaxes(F1, -1, -2), out=Xt)
                fnp.sum(Xt, axis=0, out=bufs["t1"])
                fnp.add(D21_fb, bufs["t1"], out=D21_fb)"""
new = """                capacity = bufs["gyr"].shape[0]
                thin_left = self._pool.get("c04_left", (capacity, n, 2 * rfb))[:k]
                thin_right = self._pool.get("c04_right", (capacity, n, 2 * rfb))[:k]
                fnp.copyto(thin_left[:, :, :rfb], GYR)
                fnp.copyto(thin_left[:, :, rfb:], GXR)
                fnp.copyto(thin_right[:, :, :rfb], F2)
                fnp.copyto(thin_right[:, :, rfb:], F1)
                fnp.matmul(thin_left, fnp.swapaxes(thin_right, -1, -2), out=Xt)
                D21_fb = fnp.sum(Xt, axis=0, out=bufs["d21fb"])"""
assert s.count(old) == 1
s = s.replace(old, new)
(dest / "estimator.py").write_text(s)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256((parent / "estimator.py").read_bytes()).hexdigest(),
    sha256=hashlib.sha256(s.encode()).hexdigest(),
    ablation_key="C04",
    ablation_version="v30.004",
    modifications=[
        "Pool combined GYR/GXR and F2/F1 columns; one final GEMM and k-sum; copies charged"
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

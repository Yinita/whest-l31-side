"""Preserve dense AB layout but avoid materialized d3/g4 factor matrices."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/ablation_v30_011"
dest = root / "estimators/ablation_v30_021"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
s = (parent / "estimator.py").read_text()
old = """                    d3col = fnp.multiply((D3)[:, None], ones_row, out=NN("d3col"))
                    g4col = fnp.multiply((g4row)[:, None], ones_row, out=NN("g4col"))"""
new = """                    d3col = D3[:, None]
                    g4col = g4row[:, None]"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """                        fnp.copyto(abbuf[t_], b2[b_])"""
new = """                        fnp.multiply(b2[b_], ones2, out=abbuf[t_])"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """                        fnp.copyto(abbuf[t_], b2[a_])"""
new = """                        fnp.multiply(b2[a_], ones2, out=abbuf[t_])"""
assert s.count(old) == 1
s = s.replace(old, new)
(dest / "estimator.py").write_text(s)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256((parent / "estimator.py").read_bytes()).hexdigest(),
    sha256=hashlib.sha256(s.encode()).hexdigest(),
    ablation_key="C21",
    ablation_version="v30.021",
    modifications=[
        "Vector d3/g4 broadcast; materialize only actual dense AB products; C11 grouped contractions"
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

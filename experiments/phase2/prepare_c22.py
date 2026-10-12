"""Unique one-dimensional factors with original term-order indexed gather."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/k3_v29"
dest = root / "estimators/ablation_v30_022"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
source = (parent / "estimator.py").read_text()
old = '            B1stack = fnp.stack([ab(p, b1) for p in prog["B1"]], axis=0)'
new = """            unique_pairs = list(dict.fromkeys(prog["B1"]))
            unique_values = fnp.stack([ab(p, b1) for p in unique_pairs], axis=0)
            pair_indices = [unique_pairs.index(p) for p in prog["B1"]]
            B1stack = unique_values[pair_indices]"""
assert source.count(old) == 1
candidate = source.replace(old, new)
(dest / "estimator.py").write_text(candidate)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
    sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    ablation_key="C22",
    ablation_version="v30.022",
    modifications=[
        "Unique one-dimensional pair stack followed by original-order gather; operations charged"
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

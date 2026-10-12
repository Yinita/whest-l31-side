"""Immutable duplicate AB2 product elimination on frozen C11 parent."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/ablation_v30_011"
dest = root / "estimators/ablation_v30_012"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
source = (parent / "estimator.py").read_text()
old = """                for t_, (a_, b_) in enumerate(ab2):
                    if a_ in ("ones2", "ones1"):"""
new = """                computed_pairs = {}
                for t_, (a_, b_) in enumerate(ab2):
                    pair_key = (a_, b_)
                    if pair_key in computed_pairs:
                        fnp.copyto(abbuf[t_], abbuf[computed_pairs[pair_key]])
                        continue
                    computed_pairs[pair_key] = t_
                    if a_ in ("ones2", "ones1"):"""
assert source.count(old) == 1
candidate = source.replace(old, new)
(dest / "estimator.py").write_text(candidate)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
    sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    ablation_key="C12",
    ablation_version="v30.012",
    modifications=[
        "Compute each unique AB2 product once; charged copy to repeated term slots; C11 group layout unchanged"
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

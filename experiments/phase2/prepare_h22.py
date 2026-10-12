"""H22 immutable fixed one-layer future weight/average gate sketch, rank16."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/k3_v29"
dest = root / "estimators/ablation_v30_052"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
source = (parent / "estimator.py").read_text()
old = """                fnp.copyto(Omf, w32[:, :rfb])"""
new = """                future_weight = mlp.weights[li + 1]
                future_mean = future_weight.T @ pk1v
                # Fixed diagonal variance proxy for the one future gate; no reference.
                future_var = (future_weight * future_weight).T @ K2v
                future_sigma = fnp.sqrt(fnp.maximum(future_var, 1e-10))
                average_future_gate = fnp.mean(flops.stats.norm.cdf(future_mean / future_sigma).astype(f32))
                fnp.multiply(future_weight[:, :rfb], average_future_gate, out=Omf)"""
assert source.count(old) == 1
candidate = source.replace(old, new)
(dest / "estimator.py").write_text(candidate)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
    sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    ablation_key="H22",
    ablation_version="v30.052",
    modifications=[
        "D21feedback sketch changes only to fixed one future weight and scalar legal average gate; same16rank and power iteration. Last trimmed layer does not enter birth. Scalar gate cannot change subspace in real arithmetic; attribution chiefly future weight."
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

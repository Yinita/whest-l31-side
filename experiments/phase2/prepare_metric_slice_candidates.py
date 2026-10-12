"""Separate immutable H19/H20 using explicitly transported frozen proxy core."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/k3_v29"
source = (parent / "estimator.py").read_text()
marker = """                    g4row = dG * METRIC_C"""
helper = """                    proxy_previous = fnp.multiply(C, lam_prev, out=NN("metric_proxy_previous"))
                    fnp.fill_diagonal(proxy_previous, g_prev)
                    proxy_pre = (W @ proxy_previous) @ W.T
                    actual_metric = W @ W.T
                    metric_diag = fnp.diag(actual_metric)
"""
for key, number in [("H19", 49), ("H20", 50)]:
    dest = root / f"estimators/ablation_v30_{number:03d}"
    dest.mkdir(exist_ok=True)
    if (dest / "estimator.py").exists():
        continue
    candidate = source.replace(marker, helper + marker)
    if key == "H19":
        old = '                    wk431 = None if trim else fnp.multiply(C_off, (0.5 * METRIC_C * lam_prev), out=NN("wk431"))'
        new = "                    wk431 = None if trim else _zero_diag((dG[:, None] * actual_metric + metric_diag[:, None] * proxy_pre) * 0.5)"
    else:
        old = '                    wk4m = _zero_diag(fnp.add(g22c, g22c.T, out=NN("wk4m")))'
        new = "                    wk4m = _zero_diag((dG[:, None] * metric_diag[None, :] + metric_diag[:, None] * dG[None, :] + 4.0 * proxy_pre * actual_metric) * (1.0 / 6.0))"
    assert candidate.count(old) >= 1
    candidate = candidate.replace(old, new, 1)
    (dest / "estimator.py").write_text(candidate)
    shutil.copy(parent / "LICENSE", dest / "LICENSE")
    p = json.loads((parent / "provenance.json").read_text())
    p.update(
        parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
        sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        ablation_key=key,
        ablation_version=f"v30.{number:03d}",
        modifications=[
            "Only named31 or22 slice replaced using explicit full transported frozenGproxy and actual Wmetric; other feed/diagonal channels unchanged"
        ],
    )
    (dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

"""Shared batched joiner Gram operator, rank/sketch/QR unchanged."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/k3_v29"
dest = root / "estimators/ablation_v30_019"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
s = (parent / "estimator.py").read_text()
old = """                        fnp.matmul(Aj.T, Om, out=Yq1)
                        fnp.multiply(dAj, Yq1, out=Yq1)
                        fnp.matmul(Aj, Yq1, out=Yq)
                        fnp.matmul(Pj.T, Om, out=Yq1)
                        fnp.multiply(dPj, Yq1, out=Yq1)
                        fnp.matmul(Pj, Yq1, out=Yq2)
                        fnp.add(Yq, Yq2, out=Yq)"""
new = """                        joined = pool.get("c19_joined", (2, n, n))
                        fnp.copyto(joined[0], Aj)
                        fnp.copyto(joined[1], Pj)
                        thin = pool.get("c19_thin", (2, n, r_old))
                        fnp.matmul(fnp.swapaxes(joined, -1, -2), Om[None], out=thin)
                        fnp.multiply(thin[0], dAj, out=thin[0])
                        fnp.multiply(thin[1], dPj, out=thin[1])
                        products = pool.get("c19_products", (2, n, r_old))
                        fnp.matmul(joined, thin, out=products)
                        fnp.add(products[0], products[1], out=Yq)"""
assert s.count(old) == 1
s = s.replace(old, new)
(dest / "estimator.py").write_text(s)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256((parent / "estimator.py").read_bytes()).hexdigest(),
    sha256=hashlib.sha256(s.encode()).hexdigest(),
    ablation_key="C19",
    ablation_version="v30.019",
    modifications=[
        "Two pooled batched GEMMs join Aj/Pj Gram operator; QPASS/rank/sketch unchanged"
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

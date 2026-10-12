"""Immutable C13: exact bidirectional contraction in upper output blocks."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/k3_v29"
dest = root / "estimators/ablation_v30_013"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
source = (parent / "estimator.py").read_text()
old = '                PK2 = fnp.einsum("tij,ti,tj,tg->gij", ABstack, WL2, WR2, fpm["IND2"])'
new = """                PK2 = fnp.empty((3, n, n), dtype=f32)
                fnp.copyto(PK2[1:2], fnp.einsum("tij,ti,tj,tg->gij", ABstack, WL2, WR2, fpm["IND2"][:, 1:2]))
                for output_group in (0, 2):
                    for start_i in range(0, n, 256):
                        stop_i = min(start_i + 256, n)
                        si = slice(start_i, stop_i)
                        for start_j in range(start_i, n, 256):
                            stop_j = min(start_j + 256, n)
                            sj = slice(start_j, stop_j)
                            forward = fnp.einsum("tij,ti,tj,tg->gij", ABstack[:, si, sj], WL2[:, si], WR2[:, sj], fpm["IND2"][:, output_group:output_group + 1])
                            reverse = fnp.einsum("tji,tj,ti,tg->gij", ABstack[:, sj, si], WL2[:, sj], WR2[:, si], fpm["IND2"][:, output_group:output_group + 1])
                            combined = (forward[0] + reverse[0]) * 0.5
                            fnp.copyto(PK2[output_group, si, sj], combined)
                            if start_i != start_j:
                                fnp.copyto(PK2[output_group, sj, si], combined.T)"""
assert source.count(old) == 1
candidate = source.replace(old, new)
(dest / "estimator.py").write_text(candidate)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
    sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    ablation_key="C13",
    ablation_version="v30.013",
    modifications=[
        "Exact upper PK11/PK22 bidirectional block contraction; full directional PK21 retained; no half-FLOP assumption"
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

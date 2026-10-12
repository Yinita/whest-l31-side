"""C16 r03: same symmetric core mechanism with 16-row block copy schedule."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/ablation_v30_016"
dest = root / "estimators/ablation_v30_016_r03"
dest.mkdir(exist_ok=True)
assert not (dest / "estimator.py").exists()
source = (parent / "estimator.py").read_text()
old = """        for row in range(n):
            fnp.matmul(X[row:row + 1], Y[:, row:], out=out[row:row + 1, row:])
            if row + 1 < n:
                fnp.copyto(out[row + 1:, row], out[row, row + 1:])"""
new = """        for row in range(0, n, 16):
            end = min(row + 16, n)
            fnp.matmul(X[row:end], Y[:, row:], out=out[row:end, row:])
            if end < n:
                fnp.copyto(out[end:, row:end], out[row:end, end:].T)"""
assert source.count(old) == 1
candidate = source.replace(old, new)
(dest / "estimator.py").write_text(candidate)
shutil.copy(parent / "LICENSE", dest / "LICENSE")
p = json.loads((parent / "provenance.json").read_text())
p.update(
    parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
    sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    revision="r03",
    modifications=[
        "C16 fixed16 row slab schedule; diagonal slabs computed full; cross slab mirror charged; retain r01/r02 failures"
    ],
)
(dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

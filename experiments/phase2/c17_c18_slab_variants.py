"""Reduce residual calls in C17/C18 only, immutable16-row slab revision."""

import hashlib
import json
import shutil
from pathlib import Path


def main():
    for version in ("017", "018"):
        parent = Path(f"estimators/ablation_v30_{version}")
        out = Path(f"estimators/ablation_v30_{version}_r02")
        out.mkdir(exist_ok=False)
        text = (parent / "estimator.py").read_text()
        start = text.index("    def _core_triangular(")
        end = text.index("    def _sym_product(", start)
        helper = """    def _core_triangular(self, X, Y, n):
        out = self._pool.get(("core_triangular_slab16", n), (n, n))
        for row in range(0, n, 16):
            stop = min(row + 16, n)
            fnp.matmul(X[row:stop], Y[:, row:], out=out[row:stop, row:])
            if stop < n:
                fnp.copyto(out[stop:, row:stop], out[row:stop, stop:].T)
        return out

"""
        text = text[:start] + helper + text[end:]
        (out / "estimator.py").write_text(text)
        shutil.copy(parent / "LICENSE", out / "LICENSE")
        provenance = json.loads((parent / "provenance.json").read_text())
        provenance.update(
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            direct_parent=str(parent / "estimator.py"),
            direct_parent_sha256=hashlib.sha256((parent / "estimator.py").read_bytes()).hexdigest(),
            revision="r02 fixed16-row slab; same triangular core mechanism, diagonal slabs computed dense",
        )
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Separate triangular-output variants for mathematically symmetric cores."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / "estimators/k3_v29"
original = (parent / "estimator.py").read_text()
helper = """    def _core_triangular(self, X, Y, n):
        # Compute actual upper rows, mirror only the mathematical symmetric output.
        out = self._pool.get(("core_triangular", n), (n, n))
        for row in range(n):
            fnp.matmul(X[row:row + 1], Y[:, row:], out=out[row:row + 1, row:])
            if row + 1 < n:
                fnp.copyto(out[row + 1:, row], out[row, row + 1:])
        return out

"""
variants = {
    "C16": (
        "                    Sj = (FAj @ (dAj * FAj.T)) + (FPj @ (dPj * FPj.T))",
        """                    left = fnp.concatenate([FAj, FPj], axis=1)
                    right = fnp.concatenate([dAj * FAj.T, dPj * FPj.T], axis=0)
                    Sj = self._core_triangular(left, right, FAj.shape[0]).copy()""",
    ),
    "C17": (
        "                            Sg = Tq @ (Sg @ Tq.T)",
        """                            Sg = self._core_triangular(Tq, Sg @ Tq.T, Tq.shape[0]).copy()""",
    ),
    "C18": (
        "                        S2 = Un.T @ (G2 @ Un)",
        """                        S2 = self._core_triangular(Un.T, G2 @ Un, Un.shape[1]).copy()""",
    ),
}
for key, (old, new) in variants.items():
    version = f"v30.{int(key[1:]):03d}"
    dest = root / f"estimators/ablation_{version.replace('.', '_')}"
    dest.mkdir(exist_ok=True)
    assert not (dest / "estimator.py").exists()
    assert original.count(old) == 1
    s = original.replace(old, new).replace(
        "    def _sym_product(self, X, Y, n, out, lev):",
        helper + "    def _sym_product(self, X, Y, n, out, lev):",
    )
    (dest / "estimator.py").write_text(s)
    shutil.copy(parent / "LICENSE", dest / "LICENSE")
    p = json.loads((parent / "provenance.json").read_text())
    p.update(
        parent_sha256=hashlib.sha256(original.encode()).hexdigest(),
        sha256=hashlib.sha256(s.encode()).hexdigest(),
        ablation_key=key,
        ablation_version=version,
        modifications=[
            "Only named symmetric core output upper rows calculated; charged copy mirror; no symmetry tag tricks"
        ],
    )
    (dest / "provenance.json").write_text(json.dumps(p, indent=2) + "\n")

"""Offline exact contract for terminal fresh-source diagonal correction.

Ordinary third cumulant missing at birth:
  dT = Sym(3 * E.T, I, I), E = Rres - Q @ Bm.
Transport P=W_next.T. Terminal diagonal:
  dD3 = 3 * rowsum((P @ E.T) * (P * P)).
This correction is safe as an isolated terminal-layer change because no future
birth subtracts corrected dslices from approximate source state.
"""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    output = Path("results/phase2/sprint_freshbirth_contract_r01.json")
    assert not output.exists()
    rows = []
    for n in (5, 9, 16):
        rng = np.random.default_rng(20261013 + n)
        residual = rng.normal(size=(n, n))
        q, _ = np.linalg.qr(rng.normal(size=(n, min(n, 3))))
        bm = q.T @ residual
        e = residual - q @ bm
        p = rng.normal(size=(n, n)) / np.sqrt(n)
        eye = np.eye(n)
        unsym = np.einsum("ir,jr,kr->ijk", 3 * e.T, eye, eye)
        dt = sum(unsym.transpose(perm) for perm in itertools.permutations(range(3))) / 6
        transported = np.einsum("ia,jb,kc,abc->ijk", p, p, p, dt)
        exact_diag = np.einsum("iii->i", transported)
        fast_diag = 3 * np.sum((p @ e.T) * (p * p), axis=1)
        rows.append({"width": n, "max_abs_error": float(abs(exact_diag - fast_diag).max())})
    assert all(row["max_abs_error"] < 1e-12 for row in rows)
    report = {
        "scope": "Synthetic arithmetic contract only; no benchmark score, no reference target",
        "rows": rows,
        "formula": "3*rowsum((Wnext.T@(Rres-Q@Bm).T)*(Wnext.T**2))",
        "safe_scope": "Terminal layer fresh source only; retain prior source state and no subsequent wick",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "extra_analytic_flops_n1024_r16": 2 * 1024**3 + 2 * 1024**2 * 16 + 6 * 1024**2,
        "complete": True,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

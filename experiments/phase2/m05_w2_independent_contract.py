"""Verify legal Gaussian-W2 on noncommuting PSD covariances via an independent SVD identity."""

# ruff: noqa: E402
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from m05_small_split_ut import gaussian_w2


def main():
    path = Path("results/phase2/m05_w2_independent_contract_r01.json")
    assert not path.exists()
    rng = np.random.default_rng(61013150)
    rows = []
    for n in (3, 5, 9):
        a, b = rng.normal(size=(2, n, n))
        c1, c2 = a @ a.T + .1 * np.eye(n), b @ b.T + .1 * np.eye(n)
        m1, m2 = rng.normal(size=(2, n))
        # Independent Bures metric: sqrt cross trace = nuclear norm of L1.T L2.
        l1, l2 = np.linalg.cholesky(c1), np.linalg.cholesky(c2)
        cross_trace = np.linalg.svd(l1.T @ l2, compute_uv=False).sum()
        expected = np.sqrt(max(np.sum((m1 - m2)**2) + np.trace(c1 + c2) - 2 * cross_trace, 0))
        with flops.BudgetContext(flop_budget=2**35, quiet=True) as ctx:
            fm1, fm2, fc1, fc2 = map(fnp.asarray, (m1, m2, c1, c2))
            legal = gaussian_w2(fm1, fc1, fm2, fc2)
            reversed_distance = gaussian_w2(fm2, fc2, fm1, fc1)
        got, reversed_got = float(legal.tolist()), float(reversed_distance.tolist())
        error = abs(got - expected)
        symmetry_error = abs(got - reversed_got)
        assert max(error, symmetry_error) < 1e-10
        rows.append({"width": n, "covariance_commutator_norm": float(np.linalg.norm(c1 @ c2 - c2 @ c1)),
                     "independent_svd_max_abs_error": error,
                     "symmetry_max_abs_error": symmetry_error,
                     "legal_flops": ctx.summary_dict()["flops_used"]})
    path.write_text(json.dumps({"scope": "Independent arithmetic W2 contract only; no score/reference or population convergence",
                                "rows": rows,
                                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "method_sha256": hashlib.sha256(Path("experiments/phase2/m05_small_split_ut.py").read_bytes()).hexdigest()}, indent=2) + "\n")
    print(rows)


if __name__ == "__main__":
    main()

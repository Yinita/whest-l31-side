"""H07/H10 exact V29 WK22 off-diagonal low-rank plus diagonal representation."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(40)
    rows = []
    for n in (4, 7, 19):
        g = rng.normal(size=n)
        c = 3.0
        p = g * c / 24
        matrix = p[:, None] + p[None, :]
        np.fill_diagonal(matrix, 0)
        left = np.stack([p, np.ones(n)], axis=1)
        right = np.stack([np.ones(n), p], axis=1)
        diagonal = -2 * p
        full = left @ right.T + np.diag(diagonal)
        assert np.allclose(full, matrix, atol=1e-12)
        # Right leg for C*WK22: w3_i w2_j WK22_ij.
        w3 = rng.normal(size=n)
        w2 = rng.normal(size=n)
        expected = w3[:, None] * matrix * w2[None, :]
        factored = (w3[:, None] * left) @ (w2[:, None] * right).T + np.diag(w3 * w2 * diagonal)
        error = float(np.max(abs(expected - factored)))
        assert error < 1e-12
        rows.append(
            dict(
                n=n,
                max_error=error,
                offdiagonal_rank=int(np.linalg.matrix_rank(matrix)),
                separable_rank=2,
                diagonal_rank=n,
            )
        )
    path = Path("results/phase2/ablation_h07_h10_wk22_separable_rank.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="V29 WK22/4 = p1^T+1p^T-diag(2p), p=dG*METRIC_C/24. Rank2 alone is incorrect due strict zero diagonal; diagonal support yields repeated birth coordinates whose subtraction must be propagated consistently. O(n²r) lowrank paths plausible but full birth/source subtraction and charged integration pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

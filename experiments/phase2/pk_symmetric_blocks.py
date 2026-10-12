"""C13 exact symmetric block formula, arbitrary directional source matrices."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(13)
    rows = []
    for n in (3, 8, 17):
        for terms in (1, 7, 23):
            a = rng.normal(size=(terms, n, n))
            left = rng.normal(size=(terms, n))
            right = rng.normal(size=(terms, n))
            indicator = rng.normal(size=(terms, 3))
            full = np.einsum("tij,ti,tj,tg->gij", a, left, right, indicator)
            expected = full.copy()
            for g in (0, 2):
                expected[g] = (full[g] + full[g].T) / 2
            actual = np.zeros_like(full)
            block = 4
            for i in range(0, n, block):
                si = slice(i, min(i + block, n))
                for j in range(i, n, block):
                    sj = slice(j, min(j + block, n))
                    forward = np.einsum(
                        "tij,ti,tj,tg->gij",
                        a[:, si, sj],
                        left[:, si],
                        right[:, sj],
                        indicator[:, [0, 2]],
                    )
                    reverse = np.einsum(
                        "tji,tj,ti,tg->gij",
                        a[:, sj, si],
                        left[:, sj],
                        right[:, si],
                        indicator[:, [0, 2]],
                    )
                    combined = (forward + reverse) / 2
                    actual[[0, 2], si, sj] = combined
                    actual[[0, 2], sj, si] = combined.transpose(0, 2, 1)
            actual[1] = full[1]
            for g in range(3):
                np.fill_diagonal(actual[g], 0)
                np.fill_diagonal(expected[g], 0)
            error = float(np.max(abs(actual - expected)))
            assert error < 1e-12
            rows.append(dict(n=n, terms=terms, max_error=error))
    path = Path("results/phase2/ablation_v30_013_symmetric_blocks.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Exact forward+reverse block formula for arbitrary asymmetric D21. Two contractions per upper block means approximately same element contraction work as full directional matrix; avoids full PK11/22 output but adds slicing/copy/residual. No half-FLOP claim. Legal charged candidate still pending.",
                adjusted_score=None,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

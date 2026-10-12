"""H15 CP cappedcontraction rank structure and diagonal exclusions."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(45)
    rows = []
    for n, rank in [(4, 2), (7, 3), (11, 4)]:
        factors = rng.normal(size=(3, n, rank))
        a, b, c = factors
        tensor = sum(np.einsum("ir,jr,kr->ijk", *p) for p in itertools.permutations(factors)) / 6
        u = rng.normal(size=n)
        full = np.einsum("i,ijk->jk", u, tensor)
        low = sum((y * (x.T @ u)[None, :]) @ z.T for x, y, z in [(a, b, c), (b, c, a), (c, a, b)])
        low = (low + low.T) / 6
        assert np.allclose(full, low, atol=1e-10)
        d21 = np.array([[tensor[i, i, j] for j in range(n)] for i in range(n)])
        capped = low - u[:, None] * d21 - u[None, :] * d21.T
        np.fill_diagonal(capped, 0)
        direct = np.array(
            [
                [
                    sum(u[i] * tensor[i, j, k] for i in range(n) if i not in (j, k))
                    if j != k
                    else 0
                    for k in range(n)
                ]
                for j in range(n)
            ]
        )
        error = float(np.max(abs(capped - direct)))
        assert error < 1e-10
        rows.append(
            dict(
                n=n,
                rank=rank,
                full_contraction_rank=int(np.linalg.matrix_rank(full)),
                naive_rank_bound=6 * rank,
                capped_rank=int(np.linalg.matrix_rank(capped)),
                max_error=error,
            )
        )
    path = Path("results/phase2/ablation_v30_045_source_cap_rank_bound.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="FullCP contraction rank<=6R; repeated exclusions require D21 factors and diagonal zeroing may make cappedmatrix fullrank. Lowrankcore compression is approximation unless diagonalcorrection separately represented. Actualfullproduction coefficients/oldbirthcancel and chargedcost pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

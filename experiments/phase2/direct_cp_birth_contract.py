"""H32 direct separable diagram CP construction without dense decomposition."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(62)
    rows = []
    for n, rank in [(4, 2), (7, 3), (11, 4)]:
        # Direct two-leg tree with rankR edges; middle coordinates give R²n CP columns.
        u, v = rng.normal(size=(2, n, rank))
        p, q = rng.normal(size=(2, n, rank))
        weight = rng.normal(size=n)
        left = u @ v.T
        right = p @ q.T
        direct = left[:, :, None] * right[None, :, :] * weight[None, :, None]
        factors = []
        for r, s, j in itertools.product(range(rank), range(rank), range(n)):
            middle = np.zeros(n)
            middle[j] = weight[j] * v[j, r] * p[j, s]
            factors.append((u[:, r], middle, q[:, s]))
        actual = sum(np.einsum("i,j,k->ijk", a, b, c) for a, b, c in factors)
        error = float(np.max(abs(actual - direct)))
        assert error < 1e-11
        rows.append(dict(n=n, edge_rank=rank, direct_cp_columns=len(factors), error=error))
    path = Path("results/phase2/ablation_v30_062_direct_cp_birth.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Direct tree CP factors can be formed from current edge factors with no dense tensor decomposition, but naivecolumn count nR² grows withwidth. This is not fixedsmallrank proof; historicalstate and residualcost gate remains unpassed.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

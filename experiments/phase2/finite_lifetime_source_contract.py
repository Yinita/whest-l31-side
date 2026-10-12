"""H13/H14/H30 finite-lifetime source contractions after one/two transports."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(431)
    rows = []
    for n, rank in [(4, 2), (7, 3), (11, 4)]:
        a, b, c = rng.normal(size=(3, n, rank))
        tensor = (
            sum(
                np.einsum("it,jt,kt->ijk", x, y, z) for x, y, z in itertools.permutations((a, b, c))
            )
            / 6
        )
        factors = [a, b, c]
        for step in (1, 2):
            w = rng.normal(size=(n, n)) / np.sqrt(n)
            gate = rng.uniform(0.1, 1, size=n)
            tensor = np.einsum("ai,bj,ck,ijk->abc", w, w, w, tensor)
            factors = [w @ f for f in factors]
            d3 = np.sum(factors[0] * factors[1] * factors[2], axis=1)
            assert np.allclose(d3, np.einsum("iii->i", tensor), atol=1e-10)
            aa, bb, cc = factors
            d21 = (
                np.einsum("ir,jr->ij", aa * bb, cc)
                + np.einsum("ir,jr->ij", aa * cc, bb)
                + np.einsum("ir,jr->ij", bb * cc, aa)
            ) / 3
            direct = np.array([[tensor[i, i, j] for j in range(n)] for i in range(n)])
            error = float(np.max(abs(d21 - direct)))
            assert error < 1e-10
            rows.append(
                dict(
                    n=n,
                    rank=rank,
                    step=step,
                    d21_error=error,
                    mean_only_d3_norm=float(np.linalg.norm(d3)),
                )
            )
            tensor *= gate[:, None, None] * gate[None, :, None] * gate[None, None, :]
            factors = [gate[:, None] * f for f in factors]
    path = Path("results/phase2/ablation_h13_h14_h30_lifetime_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="CPsource one/two linear Wick transports and D3/D21 contractions exact. Actual B2/mixeddiagram source coefficients, repeatedbirth compensation and eviction-state semantics still needed. Deleting after one layer is a model truncation, not exact nonlinear preservation.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

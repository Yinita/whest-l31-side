"""H23 PSD objective of fixedsource K21 slice as a compression proxy."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(523)
    rows = []
    for n, rank in [(4, 2), (7, 3), (11, 4)]:
        a, b, c = rng.normal(size=(3, n, rank))
        t = (
            sum(
                np.einsum("ir,jr,kr->ijk", x, y, z) for x, y, z in itertools.permutations((a, b, c))
            )
            / 6
        )
        k21 = np.array([[t[i, i, j] for j in range(n)] for i in range(n)])
        operator = k21 @ k21.T
        eig = np.linalg.eigvalsh(operator)
        assert eig.min() > -1e-10
        omega = rng.normal(size=(n, 2))
        fast = k21 @ (k21.T @ omega)
        assert np.allclose(fast, operator @ omega, atol=1e-10)
        rows.append(
            dict(
                n=n,
                rank=rank,
                min_eigenvalue=float(eig.min()),
                operator_error=float(np.max(abs(fast - operator @ omega))),
            )
        )
    path = Path("results/phase2/ablation_v30_053_k21_psd_sketch.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="K21K21ᵀ is validPSD compression proxy with matrix-free action. Fullsource K21 extraction, repeatedcompensation, oldbasis mapping and cost remain necessary; no claim that PSDenergy equals final mean sensitivity.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

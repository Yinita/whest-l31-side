"""H28/L27 nonlinear state projection must recompute repeated compensation."""

import itertools
import json
from pathlib import Path

import numpy as np


def tensor(f):
    return sum(np.einsum("ir,jr,kr->ijk", *p) for p in itertools.permutations(f)) / 6


def main():
    rng = np.random.default_rng(1287)
    rows = []
    for n, rank in [(5, 2), (9, 3), (13, 4)]:
        source = rng.normal(size=(3, n, 3))
        q1, _ = np.linalg.qr(rng.normal(size=(n, rank)))
        q2, _ = np.linalg.qr(rng.normal(size=(n, rank)))
        old = tensor([q1 @ (q1.T @ f) for f in source])
        new = tensor([q2 @ (q2.T @ f) for f in source])
        old21 = np.array([[old[i, i, j] for j in range(n)] for i in range(n)])
        new21 = np.array([[new[i, i, j] for j in range(n)] for i in range(n)])
        difference = new - old
        updated = old21 + np.array([[difference[i, i, j] for j in range(n)] for i in range(n)])
        error = float(np.max(abs(updated - new21)))
        assert error < 1e-11
        rows.append(
            dict(
                n=n,
                rank=rank,
                updated_repeat_error=error,
                stale_repeat_error=float(np.linalg.norm(old21 - new21)),
            )
        )
    path = Path("results/phase2/ablation_v30_122_basis_reselection_repeats.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Changing shared compression basis changes represented tensor repeats; delta-representation update required, stale repeats wrong. Recomputing repeats may be expensive; no live V29 replacement accepted.",
                scope="Projection/repeat state semantic contract",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""L27 compressed CP eviction: projection residual and repeated slices together."""

import itertools
import json
from pathlib import Path

import numpy as np


def full(factors):
    return sum(np.einsum("ir,jr,kr->ijk", *p) for p in itertools.permutations(factors)) / 6


def slices(t):
    n = len(t)
    return np.einsum("iii->i", t), np.array([[t[i, i, j] for j in range(n)] for i in range(n)])


def main():
    rng = np.random.default_rng(1273)
    rows = []
    for n, r in [(5, 2), (9, 3), (13, 4)]:
        q, _ = np.linalg.qr(rng.normal(size=(n, r)))
        sources = [rng.normal(size=(3, n, 2)) for _ in range(3)]
        projected = [[q @ (q.T @ f) for f in source] for source in sources]
        ts = [full(source) for source in projected]
        total = sum(ts)
        # Evict one projected source, and preserve exactly the projected-state semantics.
        retained = ts[0] + ts[2]
        d3, d21 = slices(total)
        gone3, gone21 = slices(ts[1])
        got3, got21 = slices(retained)
        err = max(float(np.max(abs(d3 - gone3 - got3))), float(np.max(abs(d21 - gone21 - got21))))
        assert err < 1e-11
        # Shared Q Gram must be rebuilt or subtract its source contribution.
        grams = [sum((q.T @ f) @ (q.T @ f).T for f in source) for source in sources]
        rebuilt = grams[0] + grams[2]
        subtraction = sum(grams) - grams[1]
        assert np.allclose(rebuilt, subtraction)
        rows.append(
            dict(
                n=n,
                basis_rank=r,
                coherent_slice_error=err,
                removed_gram_error=float(np.max(abs(rebuilt - subtraction))),
                projected_vs_original_source_error=float(np.linalg.norm(ts[1] - full(sources[1]))),
            )
        )
    path = Path("results/phase2/ablation_v30_122_projected_eviction_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Eviction of projected source and its projected repeats is coherent; shared Gram subtraction exact. Basis reselection changes all retained sources, so old repeat compensation cannot survive reselection unchanged. Full V29 hub weights/dense young semantics remain pending.",
                scope="Synthetic compressed-state subset; not final production candidate",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

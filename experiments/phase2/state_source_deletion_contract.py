"""L27 delete a complete CPsource and its own repeated compensation coherently."""

import itertools
import json
from pathlib import Path

import numpy as np


def tensor(factors):
    return sum(np.einsum("ir,jr,kr->ijk", *p) for p in itertools.permutations(factors)) / 6


def repeated(t):
    n = len(t)
    return np.array([[t[i, i, j] for j in range(n)] for i in range(n)])


def main():
    rng = np.random.default_rng(127)
    rows = []
    for n in (4, 7, 11):
        sources = [tensor(rng.normal(size=(3, n, 2))) for _ in range(3)]
        total = sum(sources)
        saved_repeats = repeated(total)
        removed = total - sources[1]
        adjusted_repeats = saved_repeats - repeated(sources[1])
        error = float(np.max(abs(adjusted_repeats - repeated(removed))))
        assert error < 1e-12
        # Leaving old subtraction produces exactly the removed-source repeated mismatch.
        stale_error = float(np.linalg.norm(saved_repeats - repeated(removed)))
        rows.append(dict(n=n, coherent_error=error, stale_repeat_error=stale_error))
    path = Path("results/phase2/ablation_v30_122_source_deletion_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Whole-source deletion must also delete its exact repeated compensation; stale subtraction creates nonzero mismatch. This tests additiveCPstate only, not V29 compressedhub/rebuild eviction or nonlinear impliedmoments. Real deletion negativecontrol and score pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""O03 joint old+birth K111 and complete K211 conversion, offline synthetic."""

import itertools
import json
from pathlib import Path

import numpy as np
from birth_source_derivation import symhub
from probe_k211_conversion import cumulant


def main():
    rng = np.random.default_rng(65)
    rows = []
    for n in (4, 7):
        x = rng.normal(size=(n, 73)) ** 2 + rng.normal(size=(n, 1))
        weights = rng.uniform(size=73)
        weights /= weights.sum()
        mu = x @ weights
        centered = x - mu[:, None]
        covariance = (centered * weights) @ centered.T
        k111 = np.einsum("is,js,ks,s->ijk", centered, centered, centered, weights)
        birth = symhub(rng.normal(size=(n, n)), rng.normal(size=(n, n)))
        old = k111 - birth
        actual = np.zeros((n, n))
        converted = np.zeros_like(actual)
        for a, b, c in itertools.product(range(n), repeat=3):
            if len({a, b, c}) != 3:
                continue
            pk = cumulant([x[a] ** 2, x[b], x[c]], weights)
            actual[b, c] += cumulant([x[a], x[a], x[b], x[c]], weights)
            converted[b, c] += (
                pk
                - 2 * mu[a] * (old[a, b, c] + birth[a, b, c])
                - 2 * covariance[a, b] * covariance[a, c]
            )
        error = float(np.max(abs(actual - converted)))
        assert error < 1e-10
        rows.append(
            dict(
                n=n,
                joint_cap_error=error,
                old_norm=float(np.linalg.norm(old)),
                birth_norm=float(np.linalg.norm(birth)),
            )
        )
    path = Path("results/phase2/ablation_v30_065_joint_source_conversion.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Joint complete finite-distribution conversion with arbitraryold/birthsplit exact; no isolated trimming of cancellationterms. This is bookkeeping algebra, not official truncateddiagram numericalproduction parity; actual matched-budgetproduction and approximationerror remain pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

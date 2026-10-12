"""Independent full symmetric hub tensor versus analytical repeated content."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(57)
    rows = []
    for n, rank in [(4, 2), (7, 3), (11, 4)]:
        x, y = rng.normal(size=(2, n, n))
        p = np.eye(n)
        tensor = (
            sum(
                np.einsum("it,jt,kt->ijk", a, b, c) for a, b, c in itertools.permutations((x, p, y))
            )
            / 6
        )
        d3 = np.diag(x) * np.diag(y)
        d21 = (np.diag(x)[:, None] * y.T + x * y + np.diag(y)[:, None] * x.T) / 3
        extracted = np.array([[tensor[i, i, j] for j in range(n)] for i in range(n)])
        errors = [
            float(abs(d21 - extracted).max()),
            float(abs(d3 - np.einsum("iii->i", tensor)).max()),
        ]
        a, b, c = rng.normal(size=(3, n, rank))
        old = (
            sum(
                np.einsum("it,jt,kt->ijk", aa, bb, cc)
                for aa, bb, cc in itertools.permutations((a, b, c))
            )
            / 6
        )
        w = rng.normal(size=n)
        wicked = old * w[:, None, None] * w[None, :, None] * w[None, None, :]
        mixed = wicked + tensor
        full21 = np.array([[mixed[i, i, j] for j in range(n)] for i in range(n)])
        bookkeeping = (
            np.array([[old[i, i, j] * w[i] ** 2 * w[j] for j in range(n)] for i in range(n)]) + d21
        )
        errors.append(float(abs(full21 - bookkeeping).max()))
        assert max(errors) < 1e-11
        rows.append(
            dict(
                n=n,
                rank=rank,
                birth_d21_error=errors[0],
                birth_d3_error=errors[1],
                old_plus_birth_repeated_error=errors[2],
            )
        )
    path = Path("results/phase2/ablation_v30_057_bookkeeping.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                scope="Exact identity-born arbitrary hub and arbitrary oldCP Wick repeated-content; does not prove truncated nonlinear production exact",
                rows=rows,
                conclusion="No source bookkeeping mismatch found in tested formulas; approximation and bookkeeping kept separate",
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

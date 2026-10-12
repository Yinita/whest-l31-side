"""H21 official three K211 feed factorizations versus full symmetric tensor."""

import itertools
import json
from pathlib import Path

import numpy as np
from metric_slice_derivation import lift


def symcp(a, b, c):
    return (
        sum(np.einsum("it,jt,kt->ijk", x, y, z) for x, y, z in itertools.permutations((a, b, c)))
        / 6
    )


def main():
    rng = np.random.default_rng(51)
    rows = []
    for n in (4, 7):
        g = rng.normal(size=(n, n))
        g = (g + g.T) / 2
        m = rng.normal(size=(n, n))
        m = (m + m.T) / 2
        w1, w2 = rng.normal(size=(2, n))
        identity = np.eye(n)
        ones = np.ones(n)
        first = symcp(
            w2[:, None] * np.diag(g)[:, None] * ones[None, :],
            w1[:, None] * m,
            w1[:, None] * identity / 4,
        )
        second = symcp(w1[:, None] * g, w2[:, None] * identity, w1[:, None] * m)
        third = symcp(
            w1[:, None] * g,
            w1[:, None] * identity,
            w2[:, None] * np.diag(m)[:, None] * ones[None, :] / 4,
        )
        full = lift(g, m)
        direct = np.zeros((n, n, n))
        for i, j, k in itertools.product(range(n), repeat=3):
            direct[i, j, k] = 0.5 * (
                w2[i] * w1[j] * w1[k] * full[i, i, j, k]
                + w1[i] * w2[j] * w1[k] * full[i, j, j, k]
                + w1[i] * w1[j] * w2[k] * full[i, j, k, k]
            )
        represented = first + second + third
        mask = np.array([len(set(i)) == 3 for i in itertools.product(range(n), repeat=3)]).reshape(
            n, n, n
        )
        error = float(np.max(abs((represented - direct)[mask])))
        assert error < 1e-11
        rows.append(
            dict(
                n=n,
                distinct_feed_error=error,
                repeated_difference=float(np.max(abs(represented - direct))),
            )
        )
    path = Path("results/phase2/ablation_v30_051_nondiagonal_feed_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Official three-factor coefficients1/4,1,1/4 match direct K211 Wick feed on distinct coordinates for arbitrary symmetric G/M. Repeated coordinate handling separate; full metric bridge adds dense legs unless factored. Legal bounded-rank candidate and charged cost pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

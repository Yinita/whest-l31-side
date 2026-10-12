"""H21 exact repeats for official represented feed, distinct then transported."""

import itertools
import json
from pathlib import Path

import numpy as np
from metric_slice_derivation import lift
from non_diagonal_feed_contract import symcp


def main():
    rng = np.random.default_rng(513)
    rows = []
    for n in (4, 7, 11):
        g = rng.normal(size=(n, n))
        g = (g + g.T) / 2
        m = rng.normal(size=(n, n))
        m = (m + m.T) / 2
        w1, w2 = rng.normal(size=(2, n))
        identity = np.eye(n)
        ones = np.ones(n)
        factors = [
            (
                w2[:, None] * np.diag(g)[:, None] * ones[None, :],
                w1[:, None] * m,
                w1[:, None] * identity / 4,
            ),
            (w1[:, None] * g, w2[:, None] * identity, w1[:, None] * m),
            (
                w1[:, None] * g,
                w1[:, None] * identity,
                w2[:, None] * np.diag(m)[:, None] * ones[None, :] / 4,
            ),
        ]
        full = lift(g, m)
        direct = np.zeros((n, n, n))
        for i, j, k in itertools.product(range(n), repeat=3):
            direct[i, j, k] = 0.5 * (
                w2[i] * w1[j] * w1[k] * full[i, i, j, k]
                + w1[i] * w2[j] * w1[k] * full[i, j, j, k]
                + w1[i] * w1[j] * w2[k] * full[i, j, k, k]
            )
        represented = sum(symcp(*f) for f in factors)
        mask = np.fromfunction(lambda i, j, k: (i != j) & (i != k) & (j != k), (n, n, n))
        repeat = represented * (~mask)
        transport = rng.normal(size=(n, n)) / np.sqrt(n)
        expected = np.einsum(
            "ai,bj,ck,ijk->abc", transport, transport, transport, direct * mask, optimize=True
        )
        alternate = sum(symcp(*(transport @ f for f in fs)) for fs in factors) - np.einsum(
            "ai,bj,ck,ijk->abc", transport, transport, transport, repeat, optimize=True
        )
        error = float(np.max(abs(expected - alternate)))
        assert error < 1e-10
        rows.append(
            dict(
                n=n,
                transport_max_error=error,
                cp_columns=3 * n,
                repeated_content_norm=float(np.linalg.norm(repeat)),
            )
        )
    p = Path("results/phase2/ablation_v30_051_feed_repeat_transport.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Three official feed factorizations plus own repeated subtraction transported exactly; dense fullmetric representation3n CP columns. Existing V29 feed must be removed consistently and lowrank metric/core approximations charged; not integrated candidate.",
                scope="Full represented feed correctness subset",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

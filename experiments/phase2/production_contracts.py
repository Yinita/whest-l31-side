"""Independent output-axis and distinct-index cap contracts."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(93)
    rows = []
    for n in [2, 3, 4, 7, 8]:
        t = rng.normal(size=(n, n, n))
        t = (t + t.swapaxes(1, 2)) / 2
        direct = np.zeros((n, n))
        for b, c in itertools.product(range(n), repeat=2):
            if b != c:
                direct[b, c] = sum(t[a, b, c] for a in range(n) if a != b and a != c)
        cap = t.sum(axis=0)
        for b, c in itertools.product(range(n), repeat=2):
            cap[b, c] -= t[b, b, c] + t[c, b, c]
        np.fill_diagonal(cap, 0)
        err = float(abs(cap - direct).max())
        assert err < 1e-12
        x = rng.normal(size=(19, n)) + rng.normal(size=n)
        centered = x - x.mean(axis=0)
        pk31 = (centered**3).T @ centered / len(x)
        asym = float(abs(pk31 - pk31.T).max())
        rows.append(
            dict(
                n=n,
                cap_error=err,
                power31_asymmetry=asym,
                zero_diagonal=bool(np.all(np.diag(cap) == 0)),
            )
        )
    path = Path("results/phase2/ablation_v30_093_094_production_contracts.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                scope="Direct index exclusions and finite distribution row3/col1 convention; official diagram parity separate",
                rows=rows,
                contract="pK31 generally asymmetric: never average transpose; cap requires output b!=c after a=b,c removal",
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

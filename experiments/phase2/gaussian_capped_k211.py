"""H17 exact Gaussian covariance-path capped K211 production diagnostic."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(47)
    rows = []
    for n in (4, 7, 11):
        c = rng.normal(size=(n, n))
        c = (c + c.T) / 2
        np.fill_diagonal(c, 0)
        # Arbitrary Wick coefficients w(k,p) are shared across routes.
        w11, w12, w21, w22 = rng.normal(size=(4, n))
        direct = np.zeros((n, n))
        for b, cc in itertools.product(range(n), repeat=2):
            if b == cc:
                continue
            for a in range(n):
                if a in (b, cc):
                    continue
                direct[b, cc] += c[a, b] * c[a, cc] * w22[a] * w11[b] * w11[cc]
                direct[b, cc] += c[a, b] * c[b, cc] * w12[a] * w21[b] * w11[cc]
                direct[b, cc] += c[a, cc] * c[b, cc] * w12[a] * w11[b] * w21[cc]
        # First covariance path: zero diagonal C excludes a=b,c automatically.
        core = c.T @ (w22[:, None] * c)
        formula = core * w11[:, None] * w11[None, :]
        weighted_sum = w12 @ c
        # subtract a=c for a-b-c path and a=b for a-c-b path.
        formula += c * (weighted_sum[:, None] - w12[None, :] * c) * w21[:, None] * w11[None, :]
        formula += c * (weighted_sum[None, :] - w12[:, None] * c) * w11[:, None] * w21[None, :]
        np.fill_diagonal(formula, 0)
        error = float(np.max(abs(formula - direct)))
        assert error < 1e-11
        rows.append(dict(n=n, max_error=error))
    path = Path("results/phase2/ablation_v30_047_gaussian_capped_k211.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Three Gaussian two-edge capped K211 terms exact with a!=b,c and b!=c. This is pK production subset, not full pK->K cumulant conversion or harmonic core residual; existingproxy overlap/projection and charged candidate pending.",
                cost="One weighted dense covariance Gram O(n³), two weighted sums O(n²), elementwise output. Raw diagram inventory is not score.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

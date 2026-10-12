"""Independent exactPrice Gaussian pair check against conditional integration."""

import json
import math
from pathlib import Path

import numpy as np
from nonzero_gaussian_pair import pair, relu_mean


def main():
    rows = []
    nodes, weights = np.polynomial.legendre.leggauss(8)
    for a, b, r in [
        (0, 0, -0.6),
        (0, 0, 0.6),
        (-1, 0.5, 0.3),
        (1, -2, -0.5),
        (2, 1, 0.6),
        (-3, -2, 0.1),
    ]:
        t = r * (nodes + 1) / 2
        density = np.exp(-(a * a + b * b - 2 * t * a * b) / (2 * (1 - t * t))) / (
            2 * math.pi * np.sqrt(1 - t * t)
        )
        def gate(x):
            return 0.5 * math.erfc(-x / math.sqrt(2))
        price = r * gate(a) * gate(b) + r / 2 * np.sum(weights * (r - t) * density)
        exact = pair(a, b, r, 256) - relu_mean(a, 1) * relu_mean(b, 1)
        rows.append(dict(a=a, b=b, rho=r, abs_error=float(abs(price - exact))))
    assert max(x["abs_error"] for x in rows) < 2e-8
    out = Path("results/phase2/discussion_price8_contract.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="8nodePrice covariance only; pK21/22 remain quadratic, rho clipping excludes singular endpoints",
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

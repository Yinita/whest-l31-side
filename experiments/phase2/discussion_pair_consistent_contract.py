"""Complete consistent Gaussian pair Hermite tails for all11/21/22 channels."""

import json
import math
from pathlib import Path

import numpy as np
from h04_actual_pair_gap import ndtr


def coefficients(a, degree):
    phi = math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    p = float(ndtr(a))
    m = phi + a * p
    he = [1.0, a]
    for k in range(2, degree):
        he.append(a * he[-1] - (k - 1) * he[-2])
    one = [m, p] + [phi * (-1) ** k * he[k - 2] for k in range(2, degree + 1)]
    two = [(a * a + 1) * p + a * phi, 2 * m, 2 * p] + [
        2 * phi * (-1) ** (k - 3) * he[k - 3] for k in range(3, degree + 1)
    ]
    return one, two


def main():
    nodes, weights = np.polynomial.legendre.leggauss(256)
    rows = []
    for a, b, r in [(0, 0, 0.6), (-1, 0.5, 0.3), (1, -2, -0.5), (2, 1, 0.6), (-3, -2, 0.1)]:
        lo = max(-a, -12)
        x = (nodes + 1) * (12 - lo) / 2 + lo
        sd = math.sqrt(1 - r * r)
        z = b + r * x
        p = ndtr(z / sd)
        f = np.exp(-((z / sd) ** 2) / 2) / math.sqrt(2 * math.pi)
        right = [sd * f + z * p, (z * z + sd * sd) * p + z * sd * f]
        density = np.exp(-x * x / 2) / math.sqrt(2 * math.pi) * weights * (12 - lo) / 2
        aa = coefficients(a, 12)
        bb = coefficients(b, 12)
        for i, j in [(0, 0), (1, 0), (1, 1)]:
            exact = float(np.sum((x + a) ** (i + 1) * right[j] * density))
            series = sum(r**k * aa[i][k] * bb[j][k] / math.factorial(k) for k in range(13))
            rows.append(dict(a=a, b=b, rho=r, powers=[i + 1, j + 1], abs_error=abs(exact - series)))
    out = Path("results/phase2/discussion_pair12_contract.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                rows=rows,
                max_error=max(x["abs_error"] for x in rows),
                scope="Degree12 Gaussian11/21/22; truncation remains, not exact near rho1",
            ),
            indent=2,
        )
        + "\n"
    )
    print(max(x["abs_error"] for x in rows))


if __name__ == "__main__":
    main()

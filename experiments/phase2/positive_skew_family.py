"""L29 positive normalized skew-normal family matching mean/variance/skew."""

import json
import math
from pathlib import Path

import numpy as np


def quad(function, lower, upper, epsabs=None):
    nodes, weights = np.polynomial.legendre.leggauss(256)
    lower = -14 if not np.isfinite(lower) else lower
    upper = 14 if not np.isfinite(upper) else upper
    points = (nodes + 1) * (upper - lower) / 2 + lower
    return sum(w * function(z) for w, z in zip(weights, points)) * (upper - lower) / 2, None


def brentq(function, lower, upper):
    for _ in range(100):
        middle = (lower + upper) / 2
        if function(middle) > 0:
            upper = middle
        else:
            lower = middle
    return (lower + upper) / 2


class norm:
    @staticmethod
    def pdf(x):
        return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)

    @staticmethod
    def cdf(x):
        return 0.5 * math.erfc(-x / math.sqrt(2))


def main():
    c = np.sqrt(2 / np.pi)

    def skew(delta):
        return ((4 - np.pi) / 2) * (delta * c) ** 3 / (1 - (delta * c) ** 2) ** 1.5

    limit = skew(1 - 1e-10)
    rows = []
    for gamma in (-0.8, -0.3, 0, 0.3, 0.8):
        delta = brentq(lambda d: skew(d) - gamma, -1 + 1e-10, 1 - 1e-10)
        shape = delta / np.sqrt(1 - delta * delta)
        scale = 1 / np.sqrt(1 - delta * delta * c * c)
        location = 0.2 - scale * delta * c

        def density(z):
            t = (z - location) / scale
            return 2 * norm.pdf(t) * norm.cdf(shape * t) / scale

        moments = [
            quad(lambda z, k=k: z**k * density(z), -np.inf, np.inf, epsabs=1e-10)[0]
            for k in range(4)
        ]
        variance = moments[2] - moments[1] ** 2
        central3 = moments[3] - 3 * moments[1] * moments[2] + 2 * moments[1] ** 3
        relu = quad(lambda z: z * density(z), 0, np.inf, epsabs=1e-10)[0]
        assert abs(moments[0] - 1) < 1e-8 and abs(moments[1] - 0.2) < 1e-8
        assert abs(variance - 1) < 1e-8 and abs(central3 - gamma) < 1e-8
        rows.append(
            dict(
                skew=gamma,
                shape=shape,
                location=location,
                scale=scale,
                normalization=moments[0],
                mean=moments[1],
                variance=variance,
                central3=central3,
                relu_mean=relu,
            )
        )
    path = Path("results/phase2/ablation_v30_123_positive_skew_family.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                max_representable_abs_skew=limit,
                conclusion="Skew-normal is positive normalized and matches mu/var/skew within bounded skew range. Choosing it is an approximation assumption, not cumulant-determined truth; outside range must reject, not silently clip. Actual final state and network raw errors pending.",
                adjusted_score=None,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

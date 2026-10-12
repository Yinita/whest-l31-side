"""L29 positive skew-normal moment family on captured development finalstate."""

import json
import math
from pathlib import Path

import numpy as np
from positive_skew_family import brentq, norm


def mean_relu(mu, sigma, gamma, nodes, weights):
    c = math.sqrt(2 / math.pi)

    def skew(delta):
        return ((4 - math.pi) / 2) * (delta * c) ** 3 / (1 - (delta * c) ** 2) ** 1.5

    if abs(gamma) >= skew(1 - 1e-10):
        return None
    delta = brentq(lambda d: skew(d) - gamma, -1 + 1e-10, 1 - 1e-10)
    shape = delta / math.sqrt(1 - delta * delta)
    scale = sigma / math.sqrt(1 - delta * delta * c * c)
    location = mu - scale * delta * c
    lower = max(-location / scale, -12)
    upper = 12
    if lower >= upper:
        return 0
    z = (nodes + 1) * (upper - lower) / 2 + lower
    return (
        sum(
            weight * (location + scale * t) * 2 * norm.pdf(t) * norm.cdf(shape * t)
            for weight, t in zip(weights, z)
        )
        * (upper - lower)
        / 2
    )


def main():
    nodes, weights = np.polynomial.legendre.leggauss(96)
    rows = []
    for i in range(8):
        d = dict(np.load(f"benchmark/data/v29_features_v30_096/network_{i}.npz"))
        f = d["features"][-1]
        mu = f[:, 1]
        sigma = f[:, 3]
        gamma = f[:, 9] / sigma**3
        pred = np.array(
            [mean_relu(m, s, g, nodes, weights) for m, s, g in zip(mu, sigma, gamma)], dtype=object
        )
        invalid = sum(v is None for v in pred)
        if invalid:
            rows.append(dict(network_id=i, invalid_skews=invalid))
            continue
        pred = pred.astype(float)
        reference = d["reference"][-1]
        before = float(np.mean((d["predictions"][-1] - reference) ** 2))
        after = float(np.mean((pred - reference) ** 2))
        rows.append(
            dict(
                network_id=i,
                before_raw=before,
                after_raw=after,
                ratio=after / before,
                invalid_skews=0,
            )
        )
    path = Path("results/phase2/ablation_v30_123_positive_family_network_screen.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="Offline fixedskew-normal from capturedmu/var/D3, originalreference read only to score; no fitting",
                conclusion="Family choice is an approximation, not distribution recovery; rejects out-of-range skew. Charged quadrature unavailable and K4 ignored; offline screen only.",
                adjusted_score=None,
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

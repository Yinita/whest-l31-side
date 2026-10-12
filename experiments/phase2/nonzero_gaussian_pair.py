"""H04 nonzero-mean bivariate Gaussian ReLU covariance quadrature contract."""

import json
import math
from pathlib import Path

import numpy as np


def relu_mean(mu, sigma):
    alpha = mu / sigma
    return sigma * math.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi) + mu * 0.5 * math.erfc(
        -alpha / math.sqrt(2)
    )


def pair(mu1, mu2, rho, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    lower = max(-mu1, -12)
    upper = 12
    if lower >= upper:
        return 0
    x = (nodes + 1) * (upper - lower) / 2 + lower
    conditional_sigma = math.sqrt(max(1 - rho * rho, 1e-30))
    return (
        sum(
            weight
            * (z + mu1)
            * relu_mean(mu2 + rho * z, conditional_sigma)
            * math.exp(-z * z / 2)
            / math.sqrt(2 * math.pi)
            for weight, z in zip(weights, x)
        )
        * (upper - lower)
        / 2
    )


def main():
    rows = []
    for a, b, rho in [
        (0, 0, -0.8),
        (0, 0, 0.8),
        (-1, 0.5, 0.3),
        (1, -2, -0.5),
        (2, 1, 0.95),
        (-3, -2, 0.1),
    ]:
        x = pair(a, b, rho, 128)
        y = pair(a, b, rho, 256)
        assert abs(x - y) < 1e-9
        covariance = y - relu_mean(a, 1) * relu_mean(b, 1)
        if a == 0 and b == 0:
            exact = (math.sqrt(1 - rho * rho) + (math.pi - math.acos(rho)) * rho - 1) / (
                2 * math.pi
            )
            assert abs(covariance - exact) < 1e-10
        rows.append(
            dict(mu1=a, mu2=b, rho=rho, covariance=covariance, quadrature_order_delta=abs(x - y))
        )
    path = Path("results/phase2/ablation_v30_034_nonzero_pair_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="Integrate Xrelu * E[Yrelu|Z1=z] against standard normal; conditional variance1-rho²",
                conclusion="Nonzero mean pair integral stable on fixed grid and zero-mean arccos parity. Offline quadrature is not lawful production implementation; actual covariance distribution, highorder subtraction and costs pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

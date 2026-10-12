"""Charged Gaussian ReLU marginal cumulants through degree6, offline quadrature."""

import json
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.radial_k6_nonlinear_production import cumulants


def main():
    path = Path("results/phase2/ablation_v30_083_r02_k6_charged_marginal.json")
    assert not path.exists()
    mu = np.array([-1, 0, 0.5, 2], dtype="float32")
    sigma = np.array([0.7, 1, 1.3, 0.6], dtype="float32")
    nodes, weights = np.polynomial.legendre.leggauss(320)
    expected = []
    for m, s in zip(mu, sigma):
        upper = max(float(m + 12 * s), float(12 * s))
        x = (nodes + 1) * upper / 2
        density = np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        raw = np.array([np.sum(weights * density * x**degree) * upper / 2 for degree in range(7)])
        expected.append(cumulants(raw)[1:])
    with flops.BudgetContext(flop_budget=2**30) as ctx:
        m = fnp.array(mu)
        s = fnp.array(sigma)
        a = m / s
        cdf = flops.stats.norm.cdf(a).astype(fnp.float32)
        phi = flops.stats.norm.pdf(a).astype(fnp.float32)
        truncated = [cdf, s * phi + m * cdf]
        for degree in range(2, 7):
            truncated.append(m * truncated[-1] + (degree - 1) * s * s * truncated[-2])
        raw = [fnp.ones(len(mu), dtype=fnp.float32)] + truncated[1:]
        k = [fnp.zeros(len(mu), dtype=fnp.float32)]
        for degree in range(1, 7):
            correction = fnp.zeros(len(mu), dtype=fnp.float32)
            for j in range(1, degree):
                correction = correction + math.comb(degree - 1, j - 1) * k[j] * raw[degree - j]
            k.append(raw[degree] - correction)
    actual = np.stack([np.asarray(x) for x in k[1:]], axis=1)
    relative = float(np.max(abs(actual - np.array(expected)) / (1 + abs(np.array(expected)))))
    assert relative < 5e-4
    path.write_text(
        json.dumps(
            dict(
                mu=mu.tolist(),
                sigma=sigma.tolist(),
                actual=actual.tolist(),
                quadrature_expected=expected,
                maximum_scaled_error=relative,
                flops=ctx.summary_dict()["flops_used"],
                scope="Legally billed Gaussian marginal moment/cumulant production through6, no tensor radial trace or real network score",
                limitation="Float32 cancellation can dominate small highorder cumulants; mixed-coordinate and nonGaussian interactions not modeled",
            ),
            indent=2,
            default=lambda a: a.tolist(),
        )
        + "\n"
    )
    print(relative, ctx.summary_dict()["flops_used"])


if __name__ == "__main__":
    main()

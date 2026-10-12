"""O19 exact Gaussian ReLU K5 marginal production, not full core closure."""

import json
from pathlib import Path

import numpy as np
from radial_k6_nonlinear_production import cumulants, moments


def main():
    nodes, weights = np.polynomial.legendre.leggauss(256)
    rows = []
    for mu, sigma in [(-1.2, 0.8), (0, 1), (0.8, 1.2), (2, 0.7)]:
        raw = moments(mu, sigma)
        upper = max(mu + 12 * sigma, 12 * sigma)
        x = (nodes + 1) * upper / 2
        density = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        numeric = np.array([np.sum(weights * density * x**r) * upper / 2 for r in range(7)])
        analytic = cumulants(raw)[5]
        direct = cumulants(numeric)[5]
        error = float(abs(analytic - direct))
        assert error < 1e-8
        rows.append(
            dict(
                mu=mu,
                sigma=sigma,
                kappa5=float(analytic),
                quadrature_kappa5=float(direct),
                error=error,
            )
        )
    p = Path("results/phase2/ablation_v30_081_k5_gaussian_marginal.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Gaussian ReLU fifth marginal cumulant exactly produced and quadrature-checked; nonGaussian input cumulant terms, full r1 core and official K4-AUG diagram selection remain absent. Cannot propagate scalar K5 as complete tensor.",
                scope="Marginal production subset only; no score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

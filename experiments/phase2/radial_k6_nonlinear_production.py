"""O21/O22 Gaussian ReLU marginal sixth cumulant and radial closure contract."""

import json
import math
from pathlib import Path

import numpy as np


def cumulants(raw):
    k = np.zeros(7)
    for n in range(1, 7):
        k[n] = raw[n] - sum(math.comb(n - 1, j - 1) * k[j] * raw[n - j] for j in range(1, n))
    return k


def moments(mu, sigma):
    alpha = mu / sigma
    m = np.zeros(7)
    m[0] = (1 + math.erf(alpha / math.sqrt(2))) / 2
    phi = math.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi)
    m[1] = mu * m[0] + sigma * phi
    for r in range(2, 7):
        m[r] = mu * m[r - 1] + (r - 1) * sigma * sigma * m[r - 2]
    return m


def main():
    nodes, weights = np.polynomial.legendre.leggauss(256)
    rows = []
    for mu, sigma in [(-1, 0.7), (0, 1), (0.5, 1.3), (2, 0.6)]:
        raw = moments(mu, sigma)
        # Integrate positive half line, [-10sigma,10sigma] around mu.
        upper = max(mu + 12 * sigma, 12 * sigma)
        x = (nodes + 1) * upper / 2
        density = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        numeric = np.array([np.sum(weights * density * x**r) * upper / 2 for r in range(7)])
        relative = float(np.max(abs(numeric - raw) / (1 + abs(raw))))
        assert relative < 1e-10
        k = cumulants(raw)
        rows.append(
            dict(mu=mu, sigma=sigma, kappa6=float(k[6]), quadrature_relative_error=relative)
        )
    # Radial isotropic least-squares marginal fit q*diag(M)^3.
    metric_diag = np.array([0.5, 1, 2, 3.0])
    k6 = np.array([r["kappa6"] for r in rows])
    feature = metric_diag**3
    q = float(feature @ k6 / (feature @ feature))
    p = Path("results/phase2/ablation_v30_083_084_k6_production_marginals.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                radial_scalar_ls=q,
                marginal_fit_residual=float(np.linalg.norm(k6 - q * feature)),
                conclusion="Exact Gaussian preactivation ReLU marginal kappa6 production checked independently. Marginal fit alone cannot identify full radial trace; mixed-coordinate production and existing-state cumulant interactions remain missing.",
                scope="Synthetic nonlinear production subset; no candidate or adjusted score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

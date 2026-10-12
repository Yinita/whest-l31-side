"""H16 Gaussian K31 first-order production and identity-metric core projection."""

import json
import math
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(461)
    rows = []
    for n in (4, 7, 11):
        mu = rng.normal(size=n)
        sigma = rng.uniform(0.5, 1.5, size=n)
        alpha = mu / sigma
        gate = (1 + np.vectorize(math.erf)(alpha / np.sqrt(2))) / 2
        phi = np.exp(-alpha * alpha / 2) / np.sqrt(2 * np.pi)
        mean = sigma * phi + mu * gate
        second = (mu * mu + sigma * sigma) * gate + mu * sigma * phi
        coefficient = 3 * (1 - gate) * (second - 2 * mean * mean)
        raw = rng.normal(size=(n, n))
        cov = (raw + raw.T) / 2
        np.fill_diagonal(cov, 0)
        k31 = coefficient[:, None] * gate[None, :] * cov
        sym = (k31 + k31.T) / 2
        core = 12 * sym / (n + 4)
        assert np.allclose(core, core.T)
        rows.append(
            dict(
                n=n,
                k31_asymmetry=float(np.linalg.norm(k31 - k31.T)),
                projected_core_norm=float(np.linalg.norm(core)),
                core_symmetry_error=float(np.max(abs(core - core.T))),
            )
        )
    path = Path("results/phase2/ablation_v30_046_gaussian_core_projection.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                rule="K31_ij=3(1-Phi_i)(EX_i²-2EX_i²_mean)Phi_j Cij first-order Gaussian; core=12Sym(K31)/(n+4)",
                conclusion="Production coefficient retains asymmetric K31 until core projection. It is only Gaussian first-order contribution; subtracting V29 lambdaC proxy and inserting this alone may double count other production. Full nonlinear contribution map still pending.",
                scope="Lawful-state synthetic production/projection subset; no score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

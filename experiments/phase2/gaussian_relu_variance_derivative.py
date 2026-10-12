"""L25 independent two-variable derivative gate for coupled variance tangent."""

import json
from pathlib import Path

import numpy as np
from gaussian_state_jacobian_contract import forward


def main():
    rows = []
    for mu in (-2, -1, 0, 1, 2):
        for var in (0.25, 1, 4):
            m = np.array([mu], dtype=float)
            v = np.array([var])
            w = np.ones((1, 1))
            eps = 1e-6
            plus = forward(m, v + eps, w)
            minus = forward(m, v - eps, w)
            rows.append(
                dict(
                    mu=mu,
                    variance=var,
                    mean_variance_derivative=float((plus[0] - minus[0])[0] / (2 * eps)),
                    postvariance_variance_derivative=float((plus[1] - minus[1])[0] / (2 * eps)),
                )
            )
    path = Path("results/phase2/ablation_v30_120_variance_derivative_grid.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="Fixed Gaussian boundary grid, no labels",
                conclusion="Variance feedback derivatives measured including alpha0 and low/high sigma; full covariance and cumulant state derivatives remain absent.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

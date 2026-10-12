"""L28 covariance/control-variate efficiency gate derivation, offline only."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(124)
    rows = []
    for rho in (0, 0.3, 0.7, 0.9):
        x = rng.normal(size=100000)
        y = rho * x + np.sqrt(1 - rho * rho) * rng.normal(size=len(x))
        # Independent training beta; held samples never choose mixing coefficient.
        beta = np.cov(x[:50000], y[:50000], ddof=0)[0, 1] / np.var(y[:50000])
        ratio = float(np.var(x[50000:] - beta * y[50000:]) / np.var(x[50000:]))
        rows.append(
            dict(
                target_correlation=rho,
                train_beta=float(beta),
                held_variance_ratio=ratio,
                theoretical_ratio=1 - rho * rho,
            )
        )
    path = Path("results/phase2/ablation_v30_122_control_variate_budget.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="VarCV=VarY*(1-rho²); paidper-sample costmultiplierk requires rho²>1−1/k to improve samebudgetvariance. Known proxy expectation required.",
                conclusion="Synthetic train/held fixed control-variate contract only. Need a lawful network proxy with known expectation and paired sample/reference errors; no observed network correlation or score gain.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

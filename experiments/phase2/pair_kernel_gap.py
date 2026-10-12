"""Gaussian pair ReLU covariance gap diagnosis, fixed correlation grid."""

import json
import math
from pathlib import Path


def main():
    rows = []
    # Zero mean unit variance exact covariance; known arc-cosine kernel.
    for rho in [-0.95, -0.8, -0.5, -0.2, -0.1, -0.01, 0, 0.01, 0.1, 0.2, 0.5, 0.8, 0.95]:
        exact = (math.sqrt(1 - rho * rho) + (math.pi - math.acos(rho)) * rho - 1) / (2 * math.pi)
        truncated = rho / 4 + rho * rho / (4 * math.pi)
        rows.append(
            dict(
                rho=rho,
                exact_covariance=exact,
                quadratic_covariance=truncated,
                absolute_gap=exact - truncated,
                relative_gap=(exact - truncated) / exact if exact else 0,
            )
        )
    p = Path("results/phase2/ablation_h04_pair_gap.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                scope="Exact zero-mean Gaussian ReLU covariance only; correlationgrid not competition distribution",
                rows=rows,
                source="ChoSaul2009 arc-cosine kernel / ReLU bivariateGaussian integral",
                conclusion="Gap grows at strong correlation; actual V29 preactivation correlation mass must be measured before full-kernel replacement",
                limits=[
                    "Nonzero means/highorder state not covered",
                    "No candidate score or rational resummation claim",
                ],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

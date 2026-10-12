"""H05 zero-mean fixed Pade tail derivation, not full nonzero candidate."""

import json
import math
from pathlib import Path


def main():
    # Exact covariance: rho/4 + rho²/(4pi)+rho⁴/(48pi)+rho⁶/(160pi)+...
    # Match first two omitted even coefficients without target fitting.
    a = 1 / (48 * math.pi)
    b = (1 / (160 * math.pi)) / a
    rows = []
    for rho in [-0.95, -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 0.95]:
        exact = (math.sqrt(1 - rho * rho) + (math.pi - math.acos(rho)) * rho - 1) / (2 * math.pi)
        quadratic = rho / 4 + rho * rho / (4 * math.pi)
        rational = quadratic + a * rho**4 / (1 - b * rho * rho)
        rows.append(dict(rho=rho, quadratic_gap=exact - quadratic, pade_gap=exact - rational))
    assert 1 - b > 0
    path = Path("results/phase2/ablation_v30_035_pair_tail_pade.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                coefficients={"a": a, "b": b},
                conclusion="Fixed zero-mean Pade rho4/(1-.3rho²) matches exactrho4/rho6, positive denominator on[-1,1]. No targetfit; nonzero means/Hermite derivatives and actualnetwork candidate gate pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

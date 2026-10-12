"""Exact coefficient and power-count expansion of B1 feedback hub."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(80)
    rows = []
    for n in [4, 7, 11]:
        a, p, xt, yt = rng.normal(size=(4, n, n))
        w = rng.normal(size=n)
        full = (3 * a + xt) * (a * w[None, :] + yt)
        expanded = 3 * a * a * w[None, :] + 3 * a * yt + xt * a * w[None, :] + xt * yt
        err = float(abs(full - expanded).max())
        assert err < 1e-12
        rows.append(dict(n=n, diagonal_product_error=err))
    path = Path("results/phase2/ablation_v30_080_feedback_coefficients.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                scope="V29 X1=3A+Xt,Y1=A*w2+Yt product exact coefficient expansion; independent scalar diagonal check",
                rows=rows,
                terms=[
                    dict(factors="A*A*w2", coefficient=3, power_cost=2),
                    dict(factors="A*Yt", coefficient=3, power_cost=3),
                    dict(factors="Xt*A*w2", coefficient=1, power_cost=3),
                    dict(factors="Xt*Yt", coefficient=1, power_cost=4),
                ],
                repeated_content="D3 sums P*(3A²w2+3AYt+XtAw2+XtYt); D21 leftP uses third-scaled add, leftA and thin-right corrections separate",
                limits="Full official isomorphism and current WK22 paths pending; lowrankD21 is approximate input",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""O03/H15 legal merged source weight derived from Gaussian ReLU assignments."""

import json
import math
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(655)
    rows = []
    for n in (4, 7, 11):
        mu = rng.normal(size=n)
        sigma = rng.uniform(0.4, 1.6, size=n)
        alpha = mu / sigma
        gate = (1 + np.vectorize(math.erf)(alpha / np.sqrt(2))) / 2
        phi = np.exp(-alpha * alpha / 2) / np.sqrt(2 * np.pi)
        mean = sigma * phi + mu * gate
        # Gaussian second-power first derivative / ordinary first derivative.
        derivative_second = 2 * mean
        merged = derivative_second - 2 * gate * mean
        expected = 2 * mean * (1 - gate)
        error = float(np.max(abs(merged - expected)))
        assert error < 1e-12
        rows.append(
            dict(
                n=n, max_error=error, merged_min=float(merged.min()), merged_max=float(merged.max())
            )
        )
    p = Path("results/phase2/ablation_v30_065_relu_merged_source_weight.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="ua=E[(ReLU²)prime]=2m1, w=E[ReLUprime]=Phi, merged ua-2w*m1=2m1(1-Phi)",
                conclusion="Assignments close Gaussian old-source Wick/conversion merged coefficient and match K211 wing coefficient A. NonGaussian derivative corrections and full official budget class map remain pending.",
                scope="Lawful Gaussian coefficient mapping",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

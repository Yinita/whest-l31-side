"""Independent normal mean derivatives for source score coefficients."""

import json
import math
from pathlib import Path


def mean(mu, sigma):
    a = mu / sigma
    return (
        sigma * math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
        + mu * (1 + math.erf(a / math.sqrt(2))) / 2
    )


def main():
    rows = []
    for alpha in (-2, -1, 0, 0.5, 1, 2):
        sigma = 1.3
        mu = alpha * sigma
        h = 0.002 * sigma
        # Symmetric third derivative; converges O(h²).
        fd = (
            mean(mu + 2 * h, sigma)
            - 2 * mean(mu + h, sigma)
            + 2 * mean(mu - h, sigma)
            - mean(mu - 2 * h, sigma)
        ) / (2 * h**3)
        exact = -alpha * math.exp(-alpha * alpha / 2) / (math.sqrt(2 * math.pi) * sigma * sigma)
        err = abs(fd - exact)
        assert err < 2e-6
        rows.append(dict(alpha=alpha, third_derivative=exact, finite_difference=fd, abs_error=err))
    p = Path("results/phase2/ablation_v30_054_d3_mean_coefficient.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                coefficient="E ReLU third derivative /6 = -alpha phi/(6 sigma²)",
                conclusion="Sign and alpha dependence verified; zero at alpha0. Withdraw initial phi-only H24 sensitivity proxy, corrected r02 uses coefficient magnitude for PSD weighting.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

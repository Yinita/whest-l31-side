"""H25 exact known D3-column covariance term in K21, no residual fit."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(55)
    rows = []
    for n in (4, 7, 11):
        covariance = rng.normal(size=(n, n))
        covariance = (covariance + covariance.T) / 2
        np.fill_diagonal(covariance, 0)
        d3 = rng.normal(size=n)
        w11, w41, w12, w42, mean = rng.normal(size=(5, n))
        # TERM_SPECS pair: d3col*c_off p21 1/6 and sym p11 d3row*c_off1/3.
        # p11 symmetrization creates column term coefficient1/6.
        p21 = covariance * (d3 * w42)[:, None] * w11[None, :] / 6
        p11 = (
            covariance * (d3 * w41)[:, None] * w11[None, :]
            + covariance * w11[:, None] * (d3 * w41)[None, :]
        ) / 6
        target = p21 - 2 * mean[:, None] * p11
        leading = covariance * (d3 * (w42 - 2 * mean * w41))[:, None] * w11[None, :] / 6
        other = -covariance * (2 * mean * w11)[:, None] * (d3 * w41)[None, :] / 6
        error = float(np.max(abs(target - leading - other)))
        assert error < 1e-11
        rows.append(dict(n=n, max_error=error))
    path = Path("results/phase2/ablation_v30_055_second_separable_term.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Known D3column covariance contribution to p21−2mu*p11 factors into two diag-C-diag terms with1/6 coefficient. FullS21 includes birthrepeated subtraction and otherterms; cannot extract this as new exactleadingstate without fullcoverage/cancellation proof. No candidate score.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""Independent Gaussian K211 mixed derivative for stars centered on wing j."""

import json
import math
from pathlib import Path

import numpy as np


def relu_mean(mu, sigma):
    a = mu / sigma
    return (
        sigma * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
        + mu * (1 + np.vectorize(math.erf)(a / np.sqrt(2))) / 2
    )


def relu_second(mu, sigma):
    a = mu / sigma
    gate = (1 + np.vectorize(math.erf)(a / np.sqrt(2))) / 2
    return (mu * mu + sigma * sigma) * gate + mu * sigma * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)


def main():
    nodes, weights = np.polynomial.legendre.leggauss(256)
    rows = []
    mui, si, muj, sj, muk, sk = 0.4, 1.1, 0.2, 0.8, -0.3, 1.2
    mi = relu_mean(mui, si)
    mj = relu_mean(muj, sj)
    mk = relu_mean(muk, sk)
    vi = relu_second(mui, si) - mi * mi

    def k211(cij, cjk):
        # Conditional independence i,k given standardized j; induced Cik=cij*cjk/sj².
        s_i = np.sqrt(si * si - (cij / sj) ** 2)
        s_k = np.sqrt(sk * sk - (cjk / sj) ** 2)
        result = np.zeros(4)
        cutoff = -muj / sj
        for lo, hi in [(-12, cutoff), (cutoff, 12)]:
            z = lo + (nodes + 1) * (hi - lo) / 2
            measure = weights * (hi - lo) / 2 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
            j = np.maximum(muj + sj * z, 0) - mj
            im = relu_mean(mui + cij / sj * z, s_i) - mi
            isecond = relu_second(mui + cij / sj * z, s_i) - 2 * mi * (im + mi) + mi * mi
            km = relu_mean(muk + cjk / sj * z, s_k) - mk
            result += np.array(
                [
                    measure @ (isecond * j * km),
                    measure @ (j * km),
                    measure @ (im * j),
                    measure @ (im * km),
                ]
            )
        return result[0] - vi * result[1] - 2 * result[2] * result[3]

    for eps in (0.001, 0.0005):
        numeric = sum(a * b * k211(a * eps, b * eps) for a in (-1, 1) for b in (-1, 1)) / (
            4 * eps * eps
        )
        phi_j = np.exp(-0.5 * (muj / sj) ** 2) / np.sqrt(2 * np.pi)
        phi_i = (1 + math.erf(mui / si / np.sqrt(2))) / 2
        phi_k = (1 + math.erf(muk / sk / np.sqrt(2))) / 2
        expected = 2 * mi * (1 - phi_i) * (phi_j / sj) * phi_k
        error = float(abs(numeric - expected))
        assert error < 2e-6
        rows.append(
            dict(epsilon=eps, numeric=float(numeric), expected=float(expected), error=error)
        )
    p = Path("results/phase2/k211_wing_quadrature.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Wing-centered Gaussian K211 coefficient independently checked by conditional1D integration, includes induced Cik order2; validates omitted independent wing quadrature gate only, not nonGaussian production.",
                scope="Offline quadrature coefficient contract",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

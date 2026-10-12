"""Legal p3 Wick table contract against independent Gaussian quadrature."""

import json
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    path = Path("results/phase2/ablation_v30_067_r06_p3_wick_charged.json")
    assert not path.exists()
    mu = np.array([-3, -1, 0, 0.5, 2], dtype="float32")
    sigma = np.array([0.7, 1.3, 0.8, 2, 0.6], dtype="float32")
    nodes, weights = np.polynomial.legendre.leggauss(320)
    expected = []
    for mean, sd in zip(mu, sigma):
        lo = -float(mean / sd)
        hi = max(12.0, lo + 12.0)
        z = lo + (nodes + 1) * (hi - lo) / 2
        density = np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        expected.append(
            [
                np.sum(
                    weights
                    * (float(mean) + float(sd) * z) ** (3 - k)
                    * math.prod(range(4 - k, 4))
                    * density
                )
                * (hi - lo)
                / 2
                for k in (1, 2)
            ]
        )
    with flops.BudgetContext(flop_budget=2**30) as ctx:
        m = fnp.array(mu)
        s = fnp.array(sigma)
        a = m / s
        phi = flops.stats.norm.pdf(a).astype(fnp.float32)
        cdf = flops.stats.norm.cdf(a).astype(fnp.float32)
        w13 = 3 * ((m * m + s * s) * cdf + m * s * phi)
        w23 = 6 * (s * phi + m * cdf)
        w33 = 6 * cdf
        w43 = 6 * phi / s
        w53 = -6 * a * phi / (s * s)
    actual = np.stack([np.asarray(w13), np.asarray(w23)], axis=1)
    absolute = float(np.max(abs(actual - np.array(expected))))
    assert np.allclose(actual, expected, rtol=2e-5, atol=3e-6)
    path.write_text(
        json.dumps(
            dict(
                mu=mu.tolist(),
                sigma=sigma.tolist(),
                before="Offline supplied Wick arrays",
                after="p3 k1..5 constructed only with legal flopscope operations",
                quadrature_expected=expected,
                actual=actual.tolist(),
                maximum_absolute_error=absolute,
                flops=ctx.summary_dict()["flops_used"],
                higher_derivatives=[np.asarray(x).tolist() for x in (w33, w43, w53)],
                scope="p3 k1,k2 independent quadrature; k3..5 analytic derivatives only, not full integration or benchmark",
            ),
            indent=2,
        )
        + "\n"
    )
    print("p3 Wick", absolute, ctx.summary_dict()["flops_used"])


if __name__ == "__main__":
    main()

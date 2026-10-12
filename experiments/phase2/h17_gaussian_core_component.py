"""H17 legal complete Gaussian second-order capped matrix component."""

import json
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(471)
    rows = []
    for n in (8, 16, 32):
        mu = rng.normal(size=n)
        sigma = rng.uniform(0.5, 1.5, size=n)
        alpha = mu / sigma
        gate = (1 + np.vectorize(math.erf)(alpha / np.sqrt(2))) / 2
        phi = np.exp(-alpha * alpha / 2) / np.sqrt(2 * np.pi)
        mean = sigma * phi + mu * gate
        aa = 2 * mean * (1 - gate)
        bb = 2 * (gate * (1 - gate) - mean * phi / sigma)
        w2 = phi / sigma
        raw = rng.normal(size=(n, n))
        cov = (raw + raw.T) / 2
        np.fill_diagonal(cov, 0)
        expected = gate[:, None] * gate[None, :] * ((cov * bb[None, :]) @ cov)
        v = cov @ aa
        expected += cov * (
            w2[:, None] * gate[None, :] * (v[:, None] - cov * aa[None, :])
            + w2[None, :] * gate[:, None] * (v[None, :] - cov * aa[:, None])
        )
        np.fill_diagonal(expected, 0)
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            c = fnp.array(cov.astype("float32"))
            a = fnp.array(aa.astype("float32"))
            b = fnp.array(bb.astype("float32"))
            p = fnp.array(gate.astype("float32"))
            q = fnp.array(w2.astype("float32"))
            t = c @ a
            cap = p[:, None] * p[None, :] * ((c * b[None, :]) @ c)
            cap = cap + c * (
                q[:, None] * p[None, :] * (t[:, None] - c * a[None, :])
                + q[None, :] * p[:, None] * (t[None, :] - c * a[:, None])
            )
            fnp.fill_diagonal(cap, 0)
        error = float(np.max(abs(np.asarray(cap) - expected)))
        assert error < 2e-5
        rows.append(dict(n=n, max_float32_error=error, flops=ctx.summary_dict()["flops_used"]))
    p = Path("results/phase2/ablation_v30_047_gaussian_core_component.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Complete three Gaussian second-order capped211 paths implemented with legal operations; one cubic matmul dominates. Existing V29 proxy deletion and nonlinear nonGaussian contribution map absent, so no full candidate score.",
                scope="Charged synthetic core component",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

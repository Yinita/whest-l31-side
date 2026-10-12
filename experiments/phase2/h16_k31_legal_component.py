"""H16 asymmetric K31 Gaussian production and symmetric core legal component."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(462)
    rows = []
    for n in (8, 16, 32):
        coefficient = rng.normal(size=n).astype("float32")
        gate = rng.uniform(size=n).astype("float32")
        raw = rng.normal(size=(n, n)).astype("float32")
        cov = (raw + raw.T) * 0.5
        np.fill_diagonal(cov, 0)
        expected = coefficient[:, None] * gate[None, :] * cov
        expected_core = (expected + expected.T) * (6 / (n + 4))
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            c = fnp.array(cov)
            a = fnp.array(coefficient)
            p = fnp.array(gate)
            k31 = a[:, None] * p[None, :] * c
            core = (k31 + k31.T) * (6 / (n + 4))
        err = float(np.max(abs(np.asarray(core) - expected_core)))
        assert err < 1e-6
        rows.append(dict(n=n, max_core_error=err, flops=ctx.summary_dict()["flops_used"]))
    path = Path("results/phase2/ablation_v30_046_k31_legal_component.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Gaussian first-order K31 and core projection legal operations checked. Coefficient construction, proxy removal and full nonlinear core still needed; no full candidate score.",
                scope="Charged component only",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

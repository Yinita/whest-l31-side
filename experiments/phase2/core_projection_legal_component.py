"""O04/H28 harmonic trace-core inverse charged component including isotropic subtraction."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(584)
    rows = []
    for n in (8, 16, 32):
        raw = rng.normal(size=(n, n)).astype("float32")
        trace = (raw + raw.T) * 0.5
        metric = 2
        expected = 6 * trace / (metric * (n + 4)) - 3 * np.trace(trace) * np.eye(n) / (
            metric * (n + 2) * (n + 4)
        )
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            h = fnp.array(trace)
            core = h * (6 / (metric * (n + 4)))
            diagonal = fnp.diag(core) - fnp.sum(fnp.diag(h)) * (3 / (metric * (n + 2) * (n + 4)))
            # Legal diagonal assignment without in-place tag manipulation.
            fnp.fill_diagonal(core, diagonal)
        error = float(np.max(abs(np.asarray(core) - expected)))
        assert error < 1e-6
        rows.append(dict(n=n, error=error, flops=ctx.summary_dict()["flops_used"]))
    p = Path("results/phase2/ablation_v30_058_core_projection_component.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Full diagonal/offdiag harmonic inverse and isotropic trace subtraction legally checked. Does not supply nonlinear H from full official production or decide trace convention replacement; candidate integration pending.",
                scope="Charged synthetic projection component",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

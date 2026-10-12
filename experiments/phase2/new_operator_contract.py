"""O30 small charged operation contracts for new diagonal/core/block paths."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(30)
    cases = []
    for dtype in (np.float32, np.float64):
        a = rng.normal(size=(7, 7)).astype(dtype)
        weight = rng.normal(size=7).astype(dtype)
        weight_squared = weight * weight
        expected = (a * weight) @ a.T
        with flops.BudgetContext(flop_budget=2**30) as ctx:
            x = fnp.asarray(a)
            w = fnp.asarray(weight)
            target = fnp.empty((7, 7), dtype=dtype)
            fnp.matmul(x * w, x.T, out=target)
            fnp.copyto(target[3:, 0], target[0, 3:])
            fnp.multiply(w, w, out=w)
        numerical = np.asarray(target)
        expected[3:, 0] = expected[0, 3:]
        error = float(np.max(abs(numerical - expected)))
        assert np.allclose(numerical, expected, rtol=2e-6, atol=2e-6)
        assert np.allclose(np.asarray(w), weight_squared)
        summary = ctx.summary_dict()
        assert summary["flops_used"] > 0
        cases.append(
            dict(dtype=str(dtype), max_error=error, flops=summary["flops_used"], summary=summary)
        )
    path = Path("results/phase2/ablation_v30_092_new_operator_contract.json")
    # O30 is v30.092; no production/rpc blanket acceptance.
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                cases=cases,
                conclusion="Small local actual fnp broadcast/matmul/out/copy/same-object contract; complete candidates retain per-operator reports. No unmetered NumPy candidate computation. Remote parity and all high-order candidate operations remain pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

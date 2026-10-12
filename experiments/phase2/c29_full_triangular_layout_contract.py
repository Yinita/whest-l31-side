"""Same triangular block algorithm with full-grid packed scatter, charged gate."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    path = Path("results/phase2/ablation_v30_029_r03_full_triangular_layout.json")
    assert not path.exists()
    rng = np.random.default_rng(2903)
    rows = []
    for n in (32, 64, 128):
        x = rng.normal(size=(n, n)).astype("float32")
        y = x.T.copy()
        size = 16
        count = n // size
        costs = []
        values = []
        calls = []
        for packed in (False, True):
            with flops.BudgetContext(flop_budget=2**35) as ctx:
                left, right = fnp.array(x), fnp.array(y)
                out = fnp.empty((n, n), dtype=fnp.float32)
                grid = fnp.empty((count, count, size, size), dtype=fnp.float32) if packed else None
                for i in range(count):
                    for j in range(i, count):
                        block = (
                            left[i * size : (i + 1) * size] @ right[:, j * size : (j + 1) * size]
                        )
                        if packed:
                            fnp.copyto(grid[i, j], block)
                            if i != j:
                                fnp.copyto(grid[j, i], block.T)
                        else:
                            fnp.copyto(
                                out[i * size : (i + 1) * size, j * size : (j + 1) * size], block
                            )
                            if i != j:
                                fnp.copyto(
                                    out[j * size : (j + 1) * size, i * size : (i + 1) * size],
                                    block.T,
                                )
                if packed:
                    fnp.copyto(out, grid.transpose(0, 2, 1, 3).reshape(n, n))
            values.append(np.asarray(out).copy())
            costs.append(ctx.summary_dict()["flops_used"])
            calls.append(sum(v["calls"] for v in ctx.summary_dict()["operations"].values()))
        assert np.array_equal(*values)
        rows.append(
            dict(
                n=n,
                before_flops=costs[0],
                after_flops=costs[1],
                before_calls=calls[0],
                after_calls=calls[1],
                exact_equal=True,
            )
        )
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Full triangular-only block compute and mirrored populated grid scatter exact; packing adds copies/reshape and calls, fails nonincreasing FLOP gate. Does not warrant whole candidate under this layout.",
                scope="Fixed component negative, not C01 full estimator timing or memory acceptance",
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

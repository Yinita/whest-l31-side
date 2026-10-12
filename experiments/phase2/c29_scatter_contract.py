"""C29 full symmetric block scatter with legal APIs and measured cost."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(293)
    rows = []
    for n, size in [(64, 16), (128, 16), (128, 32)]:
        count = n // size
        seed = rng.normal(size=(n, n)).astype("float32")
        matrix = (seed + seed.T) * 0.5
        blocks = matrix.reshape(count, size, count, size).transpose(0, 2, 1, 3).copy()
        values = []
        costs = []
        for packed in (False, True):
            with flops.BudgetContext(flop_budget=2**35) as ctx:
                data = fnp.array(blocks)
                out = fnp.empty((n, n), dtype=fnp.float32)
                if packed:
                    layout = data.transpose(0, 2, 1, 3).reshape(n, n)
                    fnp.copyto(out, layout)
                else:
                    for i in range(count):
                        for j in range(i, count):
                            fnp.copyto(
                                out[i * size : (i + 1) * size, j * size : (j + 1) * size],
                                data[i, j],
                            )
                            if i != j:
                                fnp.copyto(
                                    out[j * size : (j + 1) * size, i * size : (i + 1) * size],
                                    data[i, j].T,
                                )
            values.append(np.asarray(out).copy())
            costs.append(ctx.summary_dict()["flops_used"])
        assert all(np.array_equal(value, matrix) for value in values)
        rows.append(
            dict(
                n=n,
                size=size,
                individual_cost=costs[0],
                packed_cost=costs[1],
                ratio=costs[1] / costs[0],
            )
        )
    p = Path("results/phase2/ablation_v30_029_scatter_contract.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Full populated-grid reshape scatter exact and legally charged. Packed approach requires both triangles already populated; does not implement triangular-only compute plus mirror without additional calls. Combined sparse-grid compute/resource acceptance pending.",
                scope="Component contract, no score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""C29 legal reshape layout packing vs individual copies, offline component."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(29)
    rows = []
    for n, size in [(64, 32), (64, 16), (128, 16)]:
        count = n // size
        x0 = rng.normal(size=(n, n)).astype("float32")
        y0 = rng.normal(size=(n, n)).astype("float32")
        costs = []
        values = []
        for packed in (False, True):
            with flops.BudgetContext(flop_budget=2**35) as ctx:
                x = fnp.array(x0)
                y = fnp.array(y0)
                xb = fnp.empty((count, 1, size, n), dtype=fnp.float32)
                yb = fnp.empty((count, 1, n, size), dtype=fnp.float32)
                if packed:
                    fnp.copyto(xb, x.reshape(count, 1, size, n))
                    layout = y.reshape(n, count, size).transpose(1, 0, 2)
                    fnp.copyto(yb, layout.reshape(count, 1, n, size))
                else:
                    for block in range(count):
                        fnp.copyto(xb[block, 0], x[block * size : (block + 1) * size])
                        fnp.copyto(yb[block, 0], y[:, block * size : (block + 1) * size])
            costs.append(ctx.summary_dict()["flops_used"])
            values.append((np.asarray(xb).copy(), np.asarray(yb).copy()))
        assert all(np.array_equal(a, b) for a, b in zip(*values))
        rows.append(
            dict(
                n=n,
                size=size,
                individual_flops=costs[0],
                packed_flops=costs[1],
                ratio=costs[1] / costs[0],
            )
        )
    path = Path("results/phase2/ablation_v30_029_layout_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Legal reshape/transpose/copy packing is exactly equal but charged reshape may increase FLOPs. Full triangular scatter layout/cost and residual gates pending; do not silently use NumPy strided views.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

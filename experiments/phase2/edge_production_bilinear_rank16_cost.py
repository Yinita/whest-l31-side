# ruff: noqa: E402
"""Exact edge-times-CP output D3 with distinct exclusions and legal operations."""

import json
import os
import time
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    path = Path("results/phase2/ablation_v30_074_r14_production_bilinear_rank16_cost.json")
    assert not path.exists()
    rng = np.random.default_rng(7402)
    rows = []
    for n, r in ((1024, 16),):
        legs = rng.normal(size=(3, n, r)).astype("float32")
        u, v = rng.normal(size=(2, n)).astype("float32")
        w = rng.normal(size=(n, n)).astype("float32")
        cov = rng.normal(size=(n, n)).astype("float32")
        cov = (cov + cov.T) / 2
        started = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41) as ctx:
            a, b, c = [fnp.array(x) for x in legs]
            uu, vv = fnp.array(u), fnp.array(v)
            ww = fnp.array(w)
            cc = fnp.array(cov)
            mask = fnp.ones((n, n), dtype=fnp.float32) - fnp.eye(n, dtype=fnp.float32)
            cc = cc * mask
            outputs = []
            slab = 64
            for start in range(0, n, slab):
                matrix = ww[start : min(start + slab, n)]
                total = fnp.zeros(matrix.shape[0], dtype=fnp.float32)
                for rank in range(r):
                    for orientation in range(3):
                        factors = [matrix * a[:, rank], matrix * b[:, rank], matrix * c[:, rank]]
                        others = [axis for axis in range(3) if axis != orientation]
                        x = factors[orientation] * uu
                        y = factors[others[0]] * vv
                        z = factors[others[1]] * vv
                        cz = z @ cc.T
                        cy = y @ cc
                        total = (
                            total
                            + fnp.sum(x, axis=1) * fnp.sum(y * cz, axis=1)
                            - fnp.sum(x * y * cz, axis=1)
                            - fnp.sum(x * z * cy, axis=1)
                        )
                outputs.append(total)
            fnp.concatenate(outputs)
        rows.append(
            dict(
                n=n,
                rank=r,
                elapsed_s=time.monotonic() - started,
                flops=ctx.summary_dict()["flops_used"],
                max_intermediate_elements=n * n,
                budget_summary=ctx.summary_dict(),
            )
        )
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="Production1024 rank1 exact component cost diagnostic; no dense oracle at this scale, previous bounded oracle contracts separate, no actual network score",
                cost="3Rn matrix elementwise contractions; O(Rn3) plus O(Rn2), many API calls; full network timing not established",
                score=None,
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

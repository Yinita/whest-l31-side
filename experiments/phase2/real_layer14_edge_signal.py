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
    path = Path("results/phase2/ablation_v30_059_r06_real_layer14_edge_signal.json")
    assert not path.exists()
    data = dict(np.load("benchmark/data/v29_real_source_v30_059_r05/network_0.npz"))
    # Actual public next layer matrix read separately below; independent weights never substituted.
    from whestbench import load_dataset, mlp_at

    public = load_dataset("benchmark/data/mini_first8", split="mini").select_columns(
        ["mlp_id", "mlp_name", "mlp_seed", "weights"]
    )
    actual_network = mlp_at(public, 0)
    w = np.asarray(actual_network.weights[15]).T.astype("float32")
    rows = []
    for n, r in ((1024, 16),):
        legs = np.stack([data["a"], data["p"], data["c"]]) * data["w1"][None, :, None]
        u, v = data["w2"], data["w1"]
        cov = data["covariance"]
        assert cov.shape == (n, n) and np.isfinite(cov).all()
        started = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41) as ctx:
            a, b, c = [fnp.array(x) for x in legs]
            uu, vv = fnp.array(u), fnp.array(v)
            ww = fnp.array(w)
            cc = fnp.array(cov)
            mask = fnp.ones((n, n), dtype=fnp.float32) - fnp.eye(n, dtype=fnp.float32)
            cc = cc * mask
            outputs = []
            slab = 1024
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
            actual = fnp.concatenate(outputs)
        rows.append(
            dict(
                n=n,
                rank=r,
                elapsed_s=time.monotonic() - started,
                flops=ctx.summary_dict()["flops_used"],
                max_intermediate_elements=n * n,
                budget_summary=ctx.summary_dict(),
                signal_rms=float(np.sqrt(np.mean(np.asarray(actual, dtype=np.float64) ** 2))),
                signal_maxabs=float(np.max(abs(np.asarray(actual)))),
            )
        )
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="Actual publicnetwork0 layer14 fixed16 youngB1column edge-source signal transported to next layer; no target accessed, excludes other sources/columns and feedback/feed/M; subset signal only",
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

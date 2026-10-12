"""L03 negative CV model, legal inference operator parity and charge."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    cache = Path("benchmark/data/v29_features_v30_096")
    data = [dict(np.load(cache / f"network_{i}.npz")) for i in range(8)]
    rows = []
    for held in range(8):
        train = [i for i in range(8) if i != held]
        x = np.concatenate([data[i]["features"][-1, :, :13] for i in train]).astype(np.float64)
        y = np.concatenate(
            [data[i]["reference"][-1] - data[i]["predictions"][-1] for i in train]
        ).astype(np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        constant = std < 1e-12
        mean[constant] = 0
        std[constant] = 1
        z = (x - mean) / std
        beta = np.linalg.solve(z.T @ z + 1e-3 * np.eye(13), z.T @ y)
        # Affine coefficients frozen before held target read; constants represented in intercept.
        coef = (beta / std).astype(np.float32)
        intercept = np.float32(-mean @ (beta / std))
        features = data[held]["features"][-1, :, :13].astype(np.float32)
        expected = features @ coef + intercept
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            remote = fnp.array(features)
            b = fnp.array(coef)
            delta = fnp.matmul(remote, b) + float(intercept)
        got = np.asarray(delta)
        assert np.allclose(got, expected, atol=1e-10)
        error = (data[held]["reference"][-1] - data[held]["predictions"][-1]).astype(np.float64)
        rows.append(
            dict(
                network_id=held,
                flops=ctx.summary_dict()["flops_used"],
                before_raw=float(np.mean(error**2)),
                after_raw=float(np.mean((error - got) ** 2)),
                max_inference_delta=float(np.max(abs(got - expected))),
            )
        )
    p = Path("results/phase2/ablation_v30_098_charged_readout_component.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                before_raw=float(np.mean([r["before_raw"] for r in rows])),
                after_raw=float(np.mean([r["after_raw"] for r in rows])),
                conclusion="Legal float32 inference affine model and FLOP charge verified, all targets excluded in held training. Input features cached offline; whole estimator feature construction cost and timing absent. Original negative CV retained; no official adjusted score.",
                scope="Component plus network-LOO diagnostic; not full acceptance",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""L15 actual fixedfold trained tanh model inference legal parity; no full score."""

# ruff: noqa: E402 -- thread freeze before numpy
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from nonlinear_readout_screen import train


def main():
    data = [dict(np.load(f"benchmark/data/v29_features_v30_096/network_{i}.npz")) for i in range(8)]
    x = np.concatenate([data[i]["features"][-1] for i in range(1, 8)]).astype(np.float64)
    y = np.concatenate(
        [data[i]["reference"][-1] - data[i]["predictions"][-1] for i in range(1, 8)]
    ).astype(np.float64)
    mean, scale, w, b, v, c, yscale = train(x, y)
    test = data[0]["features"][-1].astype(np.float32)
    expected = (np.tanh((test.astype(np.float64) - mean) / scale @ w + b) @ v + c[0]) * yscale
    with flops.BudgetContext(flop_budget=2**35) as ctx:
        f = fnp.array(test)
        normalized = (f - fnp.array(mean.astype("float32"))) / fnp.array(scale.astype("float32"))
        hidden = fnp.tanh(
            normalized @ fnp.array(w.astype("float32")) + fnp.array(b.astype("float32"))
        )
        delta = (hidden @ fnp.array(v.astype("float32")) + float(np.float32(c[0]))) * float(
            np.float32(yscale)
        )
    actual = np.asarray(delta)
    error = float(np.max(abs(actual - expected)))
    assert error < 1e-8
    p = Path("results/phase2/ablation_v30_110_tanh_legal_inference.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                train_ids=list(range(1, 8)),
                held_id=0,
                max_float32_inference_error=error,
                flops=ctx.summary_dict()["flops_used"],
                conclusion="Actual frozen500step tanh fold0 inference legal ops parity checked; cached features and full estimator timing/resource absent. Originalfull8fold negativeCV retained.",
                scope="Charged actualmodel inference component",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

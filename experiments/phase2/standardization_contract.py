"""Freeze a development-fold linear model and verify legal inference parity."""

import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    data = [
        dict(np.load(root / f"benchmark/data/v29_features_v30_096/network_{i}.npz"))
        for i in range(8)
    ]
    train_ids = list(range(1, 8))
    x = np.concatenate([data[i]["features"][-1] for i in train_ids]).astype(np.float64)
    y = np.concatenate(
        [data[i]["reference"][-1] - data[i]["predictions"][-1] for i in train_ids]
    ).astype(np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    mask = scale < 1e-12
    mean[mask] = 0
    scale[mask] = 1
    z = (x - mean) / scale
    coef = np.linalg.solve(z.T @ z + 1e-3 * np.eye(13), z.T @ y)
    model = dict(
        train_ids=train_ids,
        held_id=0,
        mean=mean.tolist(),
        scale=scale.tolist(),
        coef=coef.tolist(),
        source="developmentfoldonly",
        ridge=1e-3,
    )
    blob = json.dumps(model, sort_keys=True).encode()
    test = data[0]["features"][-1]
    expected = ((test.astype(np.float64) - mean) / scale) @ coef
    with flops.BudgetContext(flop_budget=2**41) as ctx:
        values = (
            (fnp.array(test) - fnp.array(mean.astype("float32")))
            / fnp.array(scale.astype("float32"))
        ) @ fnp.array(coef.astype("float32"))
    actual = np.asarray(values)
    error = float(abs(actual - expected).max())
    assert np.isfinite(actual).all()
    p = root / "results/phase2/ablation_v30_128_standardization_contract.json"
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                scope="Frozenfold model inference using legalfloat32ops vs offlinefloat64; no officialcandidate score",
                model=model,
                model_sha256=hashlib.sha256(blob).hexdigest(),
                maxerror=error,
                flops=ctx.summary_dict()["flops_used"],
                feature_order=json.loads(
                    (root / "results/phase2/ablation_v30_096_features_manifest.json").read_text()
                )["feature_order"],
                limitation="Full endtoend estimator integration not done; contract only",
            ),
            indent=2,
        )
        + "\n"
    )
    print("maxerror", error)


if __name__ == "__main__":
    main()

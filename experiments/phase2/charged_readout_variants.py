"""L04-06/L10-13 affine inference components, retain negative LOO results."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def fit(x, y, equal=False):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    constant = std < 1e-12
    mean[constant] = 0
    std[constant] = 1
    z = (x - mean) / std
    factor = 1 / 1024 if equal else 1
    beta = np.linalg.solve(factor * (z.T @ z) + 1e-3 * np.eye(x.shape[1]), factor * (z.T @ y))
    return (beta / std).astype(np.float32), np.float32(-mean @ (beta / std))


def main():
    data = [dict(np.load(f"benchmark/data/v29_features_v30_096/network_{i}.npz")) for i in range(8)]
    for item in data:
        f = item["features"]
        a = f[:, :, 4]
        phi = f[:, :, 6]
        skew = f[:, :, 9] / f[:, :, 3] ** 3
        extra = np.stack(
            [
                a * a * phi,
                a**3 * phi,
                a**4 * phi,
                skew,
                skew * phi,
                np.broadcast_to(f[:, :, 1].mean(axis=1)[:, None], a.shape),
                np.broadcast_to(f[:, :, 2].mean(axis=1)[:, None], a.shape),
                np.broadcast_to(f[:, :, 9].mean(axis=1)[:, None], a.shape),
            ],
            axis=2,
        )
        item["features"] = np.concatenate([f, extra], axis=2)
    output = {}
    variants = {
        "L04": list(range(1, 13)),
        "L05": list(range(16)),
        "L06": list(range(13)) + [16, 17],
        "L10": list(range(13)) + [18, 19, 20],
        "L11": list(range(13)),
        "L12": list(range(13)),
        "L13": list(range(13)),
    }
    for key, columns in variants.items():
        rows = []
        for held in range(8):
            train = [i for i in range(8) if i != held]
            features = [d["features"][-1][:, columns].astype(np.float64) for d in data]
            errors = [(d["reference"][-1] - d["predictions"][-1]).astype(np.float64) for d in data]
            scale = [
                np.maximum(abs(d["predictions"][-1]), 0.1) if key == "L11" else np.ones(1024)
                for d in data
            ]
            x = np.concatenate([features[i] for i in train])
            y = np.concatenate([errors[i] / scale[i] for i in train])
            if key == "L13":
                x = np.concatenate(
                    [data[i]["features"][:, :, :13].reshape(-1, 13) for i in train]
                ).astype(np.float64)
                y = np.concatenate(
                    [(data[i]["reference"] - data[i]["predictions"]).reshape(-1) for i in train]
                ).astype(np.float64)
            coef, intercept = fit(x, y, key == "L12")
            test = features[held].astype(np.float32)
            with flops.BudgetContext(flop_budget=2**35) as ctx:
                result = (fnp.array(test) @ fnp.array(coef) + float(intercept)) * fnp.array(
                    scale[held].astype(np.float32)
                )
            delta = np.asarray(result)
            expected = (test @ coef + intercept) * scale[held]
            assert np.allclose(delta, expected, atol=1e-9)
            rows.append(
                dict(
                    network_id=held,
                    before_raw=float(np.mean(errors[held] ** 2)),
                    after_raw=float(np.mean((errors[held] - delta) ** 2)),
                    flops=ctx.summary_dict()["flops_used"],
                )
            )
        output[key] = dict(
            rows=rows,
            before_raw=float(np.mean([r["before_raw"] for r in rows])),
            after_raw=float(np.mean([r["after_raw"] for r in rows])),
        )
    p = Path("results/phase2/ablation_learning_charged_components_r02.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                variants=output,
                scope="Legal inference components on offline cached features; held-network labels excluded from training",
                limitation="Extra feature construction not charged here; full candidates/timing absent, adjusted unavailable. Negative original screens retained.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

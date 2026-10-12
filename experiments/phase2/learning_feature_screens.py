"""Network-held-out offline residual correction screens, no final submission."""

import json
from pathlib import Path

import numpy as np


def fit_predict(train_x, train_y, test_x, network_equal=False):
    x = np.concatenate(train_x).astype(np.float64)
    y = np.concatenate(train_y).astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    constant = std < 1e-12
    mean[constant] = 0
    std[constant] = 1
    x = (x - mean) / std
    test = (test_x - mean) / std
    # Fixed small ridge chosen before targets; no scalar parameter sweep.
    ridge = 1e-3
    weights = np.ones(len(y))
    if network_equal:
        offset = 0
        for arr in train_y:
            weights[offset : offset + len(arr)] = 1 / len(arr)
            offset += len(arr)
    beta = np.linalg.solve(
        x.T @ (x * weights[:, None]) + ridge * np.eye(x.shape[1]), x.T @ (y * weights)
    )
    return test @ beta


def evaluate(data, columns, equal=False, normalize=False, shared=False):
    rows = []
    for held in range(8):
        train = [i for i in range(8) if i != held]
        features = [d["features"][-1, :, columns].T for d in data]
        errors = [d["reference"][-1] - d["predictions"][-1] for d in data]
        if normalize:
            scales = [np.maximum(abs(d["predictions"][-1]), 0.1) for d in data]
            targets = [y / s for y, s in zip(errors, scales)]
        else:
            scales = [np.ones(1024) for _ in data]
            targets = errors
        training_features = [features[i] for i in train]
        training_targets = [targets[i] for i in train]
        if shared:
            training_features = [
                data[i]["features"][:, :, columns].reshape(-1, len(columns)) for i in train
            ]
            training_targets = [
                (data[i]["reference"] - data[i]["predictions"]).reshape(-1) for i in train
            ]
        delta = (
            fit_predict(training_features, training_targets, features[held], equal) * scales[held]
        )
        before = float(np.mean(errors[held].astype(np.float64) ** 2))
        after = float(np.mean((errors[held] - delta) ** 2))
        rows.append(
            dict(network_id=held, before_raw=before, after_raw=after, raw_ratio=after / before)
        )
    return dict(
        scope="Development leave-one-network-out predictions; held network target excluded from training; not independent final validation",
        columns=columns,
        per_network=rows,
        before_raw=float(np.mean([r["before_raw"] for r in rows])),
        after_raw=float(np.mean([r["after_raw"] for r in rows])),
    )


def main():
    root = Path(__file__).resolve().parents[2]
    data = [
        dict(np.load(root / f"benchmark/data/v29_features_v30_096/network_{i}.npz"))
        for i in range(8)
    ]
    variants = {
        "L03": list(range(13)),
        "L04": list(range(1, 13)),
        "L05": list(range(16)),
        "L06": list(range(13)) + [16, 17],
        "L10": list(range(13)) + [18, 19, 20],
        "L11": list(range(13)),
        "L12": list(range(13)),
        "L13": list(range(13)),
    }
    for item in data:
        f = item["features"]
        alpha = f[:, :, 4]
        phi = f[:, :, 6]
        skew = f[:, :, 9] / f[:, :, 3] ** 3
        extra = np.stack(
            [
                alpha**2 * phi,
                alpha**3 * phi,
                alpha**4 * phi,
                skew,
                skew * phi,
                np.broadcast_to(f[:, :, 1].mean(axis=1)[:, None], alpha.shape),
                np.broadcast_to(f[:, :, 2].mean(axis=1)[:, None], alpha.shape),
                np.broadcast_to(f[:, :, 9].mean(axis=1)[:, None], alpha.shape),
            ],
            axis=2,
        )
        item["features"] = np.concatenate([f, extra], axis=2)
    out = {}
    for key, columns in variants.items():
        result = evaluate(
            data, columns, equal=key == "L12", normalize=key == "L11", shared=key == "L13"
        )
        result.update(
            adjusted_score=None,
            reason="Offline learned readout not yet implemented in charged flopscope estimator; cannot claim official adjusted score",
        )
        out[key] = result
        print(key, result["after_raw"] / result["before_raw"], flush=True)
    path = root / "results/phase2/ablation_learning_loocv_r01.json"
    assert not path.exists()
    path.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

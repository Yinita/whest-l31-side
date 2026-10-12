"""Frozen 13-8-1 tanh, fullbatchAdam500 steps, networkLOO, offline only."""

import json
from pathlib import Path

import numpy as np


def train(x, y):
    rng = np.random.default_rng(0)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    constant = scale < 1e-12
    mean[constant] = 0
    scale[constant] = 1
    x = (x - mean) / scale
    yscale = max(float(y.std()), 1e-12)
    target = y / yscale
    w = rng.normal(size=(13, 8)) / np.sqrt(13)
    b = np.zeros(8)
    v = np.zeros(8)
    c = np.zeros(1)
    parameters = [w, b, v, c]
    moments = [np.zeros_like(p) for p in parameters]
    variances = [np.zeros_like(p) for p in parameters]
    for step in range(1, 501):
        h = np.tanh(x @ w + b)
        error = h @ v + c[0] - target
        dy = 2 * error / len(x)
        hidden = dy[:, None] * v[None, :] * (1 - h * h)
        gradients = [x.T @ hidden, hidden.sum(axis=0), h.T @ dy, np.array([dy.sum()])]
        for i, (p, g) in enumerate(zip(parameters, gradients)):
            moments[i] *= 0.9
            moments[i] += 0.1 * g
            variances[i] *= 0.999
            variances[i] += 0.001 * g * g
            p -= (
                1e-3
                * (moments[i] / (1 - 0.9**step))
                / (np.sqrt(variances[i] / (1 - 0.999**step)) + 1e-8)
            )
    return mean, scale, w, b, v, c, yscale


def main():
    root = Path(__file__).resolve().parents[2]
    data = [
        dict(np.load(root / f"benchmark/data/v29_features_v30_096/network_{i}.npz"))
        for i in range(8)
    ]
    rows = []
    for held in range(8):
        train_ids = [i for i in range(8) if i != held]
        x = np.concatenate([data[i]["features"][-1] for i in train_ids]).astype(np.float64)
        y = np.concatenate(
            [data[i]["reference"][-1] - data[i]["predictions"][-1] for i in train_ids]
        ).astype(np.float64)
        mean, scale, w, b, v, c, yscale = train(x, y)
        test = data[held]["features"][-1].astype(np.float64)
        delta = (np.tanh((test - mean) / scale @ w + b) @ v + c[0]) * yscale
        residual = data[held]["reference"][-1] - data[held]["predictions"][-1]
        before = float(np.mean(residual.astype(np.float64) ** 2))
        after = float(np.mean((residual - delta) ** 2))
        rows.append(
            dict(
                held=held,
                train_ids=train_ids,
                before_raw=before,
                after_raw=after,
                ratio=after / before,
            )
        )
        print("fold", held, after / before, flush=True)
    path = root / "results/phase2/ablation_v30_110_tanh_cv.json"
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                scope="Offline networkLOO tanhreadout, frozen500stepAdamseed0; no chargedcandidate",
                rows=rows,
                ratio=float(
                    np.mean([r["after_raw"] for r in rows])
                    / np.mean([r["before_raw"] for r in rows])
                ),
                adjusted_score=None,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""L03 charged final-only candidate, train IDs1..7 and evaluate network0 only."""

import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    data = [
        dict(np.load(f"benchmark/data/v29_features_v30_096/network_{i}.npz")) for i in range(1, 8)
    ]
    x = np.concatenate([d["features"][-1] for d in data]).astype(np.float64)
    y = np.concatenate([d["reference"][-1] - d["predictions"][-1] for d in data]).astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    constant = std < 1e-12
    mean[constant] = 0
    std[constant] = 1
    z = (x - mean) / std
    beta = np.linalg.solve(z.T @ z + 1e-3 * np.eye(13), z.T @ y)
    coefficient = (beta / std).astype(np.float32).tolist()
    intercept = float(np.float32(-mean @ (beta / std)))
    parent = Path("estimators/k3_v29/estimator.py")
    text = parent.read_text()
    needle = """            if last:
                rows.append(pk1v if delta is None else pk1v + delta)
                break"""
    assert text.count(needle) == 1
    replacement = """            if last:
                # L03 final-only frozen fold0; every feature is charged, no labels in predict.
                d3_feature = D3 if mode == 1 else zeros_n
                d21_feature = fnp.sum(fnp.abs(D21), axis=1) if mode == 1 and D21 is not None else zeros_n
                k4_feature = ones_n * float(K4_sigma) if mode == 1 else zeros_n
                readout_features = fnp.stack([ones_n, mu, var, sigma, alpha, fnp.abs(alpha),
                    phi, Phi, pk1v, d3_feature, d21_feature, k4_feature, sigma * phi], axis=1)
                learned_delta = readout_features @ fnp.asarray(FOLD0_COEFFICIENT, dtype=f32) + FOLD0_INTERCEPT
                rows.append((pk1v if delta is None else pk1v + delta) + learned_delta)
                break"""
    text = text.replace(needle, replacement)
    text += (
        "\nFOLD0_COEFFICIENT = "
        + repr(coefficient)
        + "\nFOLD0_INTERCEPT = "
        + repr(intercept)
        + "\n"
    )
    out = Path("estimators/ablation_v30_098_fold0")
    out.mkdir(exist_ok=False)
    candidate = out / "estimator.py"
    candidate.write_text(text)
    provenance = dict(
        parent=str(parent),
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        train_ids=list(range(1, 8)),
        allowed_eval_ids=[0],
        mechanism="Final-only13feature frozen affine correction, no interlayer feedback",
        rule="Never report Mini8 aggregate on this fold-specific candidate; training network evaluation is in-sample",
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

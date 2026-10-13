"""M04 single-mechanism followup: learn the linear first-ReLU proxy on pilot only.

Known expectation of every first-ReLU feature keeps a fixed pilot regression
as a valid centered control on a distinct evaluation stream. No target labels.
This is an offline method/cost gate, not allowed-operator submission evidence.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def samples(weights, count, seed):
    rng = np.random.default_rng(seed)
    ys, features = [], []
    for start in range(0, count, 256):
        x = rng.standard_normal((min(256, count - start), weights.shape[1]), dtype=np.float32)
        h = np.maximum(x @ weights[0], 0)
        features.append(h)
        for weight in weights[1:]:
            h = np.maximum(h @ weight, 0)
        ys.append(h)
    return np.concatenate(ys), np.concatenate(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path, output = Path(args.weights), Path(args.output)
    assert not output.exists()
    weights = np.load(path)["weights"]
    depth, n, _ = weights.shape
    mean0 = np.sqrt(np.sum(weights[0].astype(np.float64) ** 2, axis=0)) / math.sqrt(2 * math.pi)
    count_train, count_held = 4096, 8192
    # Freeze ridge factor and seeds before any result; no parameter search.
    ridge_factor = 1e-3
    raw_sample_flops = 2 * depth * n**2 + depth * n
    cv_sample_flops = raw_sample_flops + 2 * n**2 + 2 * n
    training_flops = count_train * raw_sample_flops + 4 * count_train * n**2 + 4 * n**3
    budget = 2**41
    raw_count = budget // raw_sample_flops
    cv_count = (budget - training_flops) // cv_sample_flops
    rows = []
    for repeat in range(2):
        train_seed = 61013000 + 2 * repeat
        held_seed = train_seed + 1
        start = time.monotonic()
        train_y, train_h = samples(weights, count_train, train_seed)
        centered_h = train_h - train_h.mean(axis=0)
        centered_y = train_y - train_y.mean(axis=0)
        gram = centered_h.T @ centered_h
        ridge = ridge_factor * np.trace(gram) / n
        gram.flat[::n + 1] += ridge
        coefficients = np.linalg.solve(gram, centered_h.T @ centered_y)
        held_y, held_h = samples(weights, count_held, held_seed)
        control = (held_h.astype(np.float64) - mean0) @ coefficients.astype(np.float64)
        residual = held_y.astype(np.float64) - control
        raw_variance = held_y.astype(np.float64).var(axis=0, ddof=1)
        residual_variance = residual.var(axis=0, ddof=1)
        rows.append({"train_seed": train_seed, "held_seed": held_seed,
                     "raw_final_variance_mean": float(raw_variance.mean()),
                     "residual_final_variance_mean": float(residual_variance.mean()),
                     "variance_ratio": float(residual_variance.mean() / raw_variance.mean()),
                     "equal_budget_mse_ratio": float(residual_variance.mean() / cv_count
                                                   / (raw_variance.mean() / raw_count)),
                     "projected_raw_mse": float(raw_variance.mean() / raw_count),
                     "projected_cv_mse": float(residual_variance.mean() / cv_count),
                     "elapsed_s": time.monotonic() - start, "ridge": float(ridge),
                     "per_neuron_raw_variance": raw_variance.tolist(),
                     "per_neuron_residual_variance": residual_variance.tolist()})
    report = {"version": "M04-first-ReLU-regression-r02", "complete": True,
              "single_change": "replace fixed tangent proxy coefficients with pilot-only ridge regression; same firstReLUfeatures/knownmean/seeds/counts",
              "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "train_count": count_train, "held_count": count_held,
              "ridge_factor": ridge_factor, "rows": rows,
              "analytic_cost_projection": {"budget": budget, "training_flops": training_flops,
                                           "raw_sample_flops": raw_sample_flops,
                                           "cv_sample_flops": cv_sample_flops,
                                           "raw_samples": int(raw_count), "cv_samples": int(cv_count)},
              "scope": "Offline method/variance gate; no benchmark reference or score",
              "limitations": ["one existing research network, two independent sampling repeats",
                              "ordinary NumPy solve and cost model; no flopscope/resource acceptance",
                              "float32 first-layer evaluation may have tiny centering bias",
                              "do not use held stream to refit coefficients or ridge"]}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if not k.startswith('per_neuron')}
                      for row in rows], indent=2))


if __name__ == "__main__":
    main()

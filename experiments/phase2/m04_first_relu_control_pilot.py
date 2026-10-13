"""Offline M04 pilot: exactly centered first-ReLU tangent control variate.

For Gaussian x, E[ReLU(x W0)] = ||W0[:, j]|| / sqrt(2*pi).
Freeze a tangent transport J through later layers at this analytic mean.
g(x) = (ReLU(x W0) - analytic_mean) @ J has known expectation zero.
Fit one beta per output on a disjoint Gaussian stream; held samples never fit it.
Ordinary NumPy is diagnostic only, never a submission or charged benchmark.
"""

# ruff: noqa: E402 -- freeze threads before NumPy import
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


def samples(weights, transport, mean0, count, seed):
    rng = np.random.default_rng(seed)
    ys, gs = [], []
    for start in range(0, count, 256):
        x = rng.standard_normal((min(256, count - start), weights.shape[1]), dtype=np.float32)
        h = np.maximum(x @ weights[0], 0)
        g = (h - mean0) @ transport
        for weight in weights[1:]:
            h = np.maximum(h @ weight, 0)
        ys.append(h)
        gs.append(g)
    return np.concatenate(ys).astype(np.float64), np.concatenate(gs).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=4096)
    parser.add_argument("--held", type=int, default=8192)
    args = parser.parse_args()
    output, path = Path(args.output), Path(args.weights)
    assert not output.exists() and args.train > 1 and args.held > 1
    weights = np.load(path)["weights"]
    depth, n, _ = weights.shape
    start = time.monotonic()
    mean0 = np.sqrt(np.sum(weights[0].astype(np.float64) ** 2, axis=0)) / math.sqrt(2 * math.pi)
    # Gated linear transport around the known first-layer expectation.
    mean = mean0.astype(np.float32)
    transport = np.eye(n, dtype=np.float32)
    for weight in weights[1:]:
        pre = mean @ weight
        gated = weight * (pre > 0)[None, :]
        transport = transport @ gated
        mean = np.maximum(pre, 0)
    build_seconds = time.monotonic() - start
    # Analytic budget projection includes fitting and proxy evaluations; not flopscope evidence.
    build_flops = 2 * (depth - 1) * n**3 + 4 * depth * n**2
    raw_sample_flops = 2 * depth * n**2 + depth * n
    cv_sample_flops = raw_sample_flops + 2 * n**2 + n
    budget = 2**41
    raw_count = budget // raw_sample_flops
    cv_held_count = max(0, (budget - build_flops) // cv_sample_flops - args.train)
    rows = []
    for repeat in range(2):
        train_seed = 61013000 + 2 * repeat
        held_seed = train_seed + 1
        start = time.monotonic()
        train_y, train_g = samples(weights, transport, mean0.astype(np.float32), args.train, train_seed)
        yc, gc = train_y - train_y.mean(axis=0), train_g - train_g.mean(axis=0)
        var_g = np.mean(gc * gc, axis=0)
        beta = np.divide(np.mean(yc * gc, axis=0), var_g,
                         out=np.zeros(n), where=var_g > 1e-20)
        held_y, held_g = samples(weights, transport, mean0.astype(np.float32), args.held, held_seed)
        residual = held_y - beta * held_g
        raw_variance = held_y.var(axis=0, ddof=1)
        residual_variance = residual.var(axis=0, ddof=1)
        rows.append({"train_seed": train_seed, "held_seed": held_seed,
                     "train_count": args.train, "held_count": args.held,
                     "elapsed_s": time.monotonic() - start,
                     "raw_final_variance_mean": float(raw_variance.mean()),
                     "residual_final_variance_mean": float(residual_variance.mean()),
                     "variance_ratio": float(residual_variance.mean() / raw_variance.mean()),
                     "beta_mean": float(beta.mean()), "beta_min": float(beta.min()),
                     "beta_max": float(beta.max()),
                     "known_zero_proxy_held_mean_squared": float(np.mean(held_g.mean(axis=0)**2)),
                     "projected_full_budget_raw_mse": float(raw_variance.mean() / raw_count),
                     "projected_full_budget_cv_mse": float(residual_variance.mean() / cv_held_count),
                     "equal_budget_mse_ratio": float(residual_variance.mean() / cv_held_count
                                                   / (raw_variance.mean() / raw_count)),
                     "per_neuron_raw_variance": raw_variance.tolist(),
                     "per_neuron_residual_variance": residual_variance.tolist()})
    report = {"version": "M04-first-ReLU-tangent-r01", "complete": True,
              "single_mechanism": "Known-expectation first-ReLU tangent control; beta on independent pilot",
              "weights": str(path.resolve()), "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source": "https://artowen.su.domains/mc/Ch-var-basic.pdf section8.9",
              "build_seconds": build_seconds, "rows": rows,
              "analytic_cost_projection": {"budget": budget, "build_flops": build_flops,
                                           "raw_sample_flops": raw_sample_flops,
                                           "cv_sample_flops": cv_sample_flops,
                                           "raw_samples": int(raw_count),
                                           "cv_held_samples_after_pilot": int(cv_held_count)},
              "scope": "Offline real-network variance/cost pilot; no benchmark target or score",
              "limitations": ["float32 evaluation can incur numerical centering bias",
                              "two seeds on one existing research network; not unseen holdout",
                              "cost projection is analytic, not measured allowed-operator flopscope execution",
                              "sampling variance estimates are not independent strong reference"]}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": [{k: v for k, v in row.items() if not k.startswith('per_neuron')}
                               for row in rows], "analytic_cost_projection": report['analytic_cost_projection']}, indent=2))


if __name__ == "__main__":
    main()

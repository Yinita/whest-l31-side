"""Offline paired antithetic reference variance pilot; never benchmark labels."""

# ruff: noqa: E402
import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import json
import time
from pathlib import Path

import numpy as np


def main():
    output = Path("results/phase2/ablation_v30_114_r02_antithetic_pilot.json")
    assert not output.exists()
    weights = np.load("benchmark/data/independent_v30_115/network_0.npz")["weights"]
    rows = []
    means = []
    for seed in (42026001, 42026002):
        rng = np.random.default_rng(seed)
        sums = np.zeros((16, 1024), dtype=np.float64)
        squares = np.zeros_like(sums)
        single_sums = np.zeros_like(sums)
        single_squares = np.zeros_like(sums)
        cross = np.zeros_like(sums)
        start = time.monotonic()
        for _ in range(16):
            x = rng.standard_normal((256, 1024), dtype=np.float32)
            paired = np.concatenate((x, -x))
            for layer, w in enumerate(weights):
                paired = np.maximum(paired @ w, 0)
                positive, negative = (
                    paired[:256].astype(np.float64),
                    paired[256:].astype(np.float64),
                )
                average = (positive + negative) / 2
                sums[layer] += average.sum(axis=0)
                squares[layer] += np.square(average).sum(axis=0)
                single_sums[layer] += positive.sum(axis=0) + negative.sum(axis=0)
                single_squares[layer] += np.square(positive).sum(axis=0) + np.square(negative).sum(
                    axis=0
                )
                cross[layer] += (positive * negative).sum(axis=0)
        elapsed = time.monotonic() - start
        mean = sums / 4096
        pair_variance = (squares - sums * mean) / 4095
        single_mean = single_sums / 8192
        marginal_variance = (single_squares - single_sums * single_mean) / 8191
        means.append(mean)
        rows.append(
            dict(
                seed=seed,
                independent_pairs=4096,
                forward_evaluations=8192,
                elapsed_s=elapsed,
                final_pair_variance=float(pair_variance[-1].mean()),
                final_marginal_variance=float(marginal_variance[-1].mean()),
                per_layer_equal_forward_variance_ratio=(
                    2 * pair_variance.mean(axis=1) / marginal_variance.mean(axis=1)
                ).tolist(),
                final_cross_covariance=float((cross[-1] / 4096 - mean[-1] ** 2).mean()),
            )
        )
    result = dict(
        rows=rows,
        seed_difference_final_mse=float(np.mean((means[0][-1] - means[1][-1]) ** 2)),
        mechanism="Only x,-x coupling changes reference sampling; same Gaussian marginals, all16 layers",
        scope="Offline weak pilot, no candidate or benchmark labels; empirical ratio has sampling uncertainty",
        limitation="4096 independent pairs per seed; no strong reference or stage acceptance",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

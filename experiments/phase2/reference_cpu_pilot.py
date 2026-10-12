"""Measured reference throughput and two-seed noise; never a strong reference."""

# ruff: noqa: E402 -- freeze BLAS threads before importing numerical libraries
import json
import os
import time
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np


def main():
    report = Path("results/phase2/ablation_v30_114_reference_cpu_pilot.json")
    assert not report.exists()
    protocol = json.loads(
        Path("results/phase2/ablation_v30_115_independent_protocol.json").read_text()
    )
    weights = np.load("benchmark/data/independent_v30_115/network_0.npz")["weights"]
    rows = []
    means = []
    for seed in protocol["reference_rng_seeds"][0]:
        rng = np.random.default_rng(seed)
        sums = np.zeros((16, 1024), dtype=np.float64)
        squares = np.zeros_like(sums)
        start = time.monotonic()
        for _ in range(16):
            x = rng.standard_normal((256, 1024), dtype=np.float32)
            for layer, w in enumerate(weights):
                x = np.maximum(x @ w, 0)
                sums[layer] += x.sum(axis=0, dtype=np.float64)
                squares[layer] += np.square(x.astype(np.float64)).sum(axis=0)
        elapsed = time.monotonic() - start
        mean = sums / 4096
        variance = (squares - sums * mean) / 4095
        means.append(mean)
        rows.append(
            dict(
                seed=seed,
                samples=4096,
                elapsed_s=elapsed,
                samples_per_s=4096 / elapsed,
                projected_seconds_for_1e9=elapsed / 4096 * 1e9,
                final_neuron_variance_mean=float(variance[-1].mean()),
                final_mean_sampling_variance_mean=float(variance[-1].mean() / 4096),
            )
        )
    result = dict(
        rows=rows,
        independent_mean_difference_mse=float(np.mean((means[0][-1] - means[1][-1]) ** 2)),
        threads=1,
        input_dtype="float32",
        accumulator_dtype="float64",
        network="independent_v30_115/network_0",
        status="weak throughput/noise diagnostic; not benchmark labels",
        limitation="4096 samples per seed; extrapolation is a throughput projection, not achieved uncertainty; no stage acceptance",
    )
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

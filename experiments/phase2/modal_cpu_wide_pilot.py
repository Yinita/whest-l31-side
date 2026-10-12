"""Bounded CPU throughput/noise pilot; no competition estimator or strong labels."""
import json
from pathlib import Path

import modal

app = modal.App("whest-reference-cpu-wide-pilot")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==2.2.6").env(
    {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
).add_local_dir("benchmark/data/independent_v30_115", "/data/networks")


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=900,
              retries=0, max_containers=64, scaledown_window=2)
def pilot(job: tuple):
    network_id, seed = job
    import hashlib
    import time

    import numpy as np

    network_path = Path(f"/data/networks/network_{network_id}.npz")
    weights = np.load(network_path)["weights"]
    digest = hashlib.sha256(network_path.read_bytes()).hexdigest()
    rng = np.random.default_rng(seed)
    sums = np.zeros((16, 1024), dtype=np.float64)
    squares = np.zeros_like(sums)
    started = time.monotonic()
    samples = 1048576
    for _ in range(samples // 256):
        x = rng.standard_normal((256, 1024), dtype=np.float32)
        for layer, w in enumerate(weights):
            x = np.maximum(x @ w, 0)
            sums[layer] += x.sum(axis=0, dtype=np.float64)
            squares[layer] += np.square(x.astype(np.float64)).sum(axis=0)
    elapsed = time.monotonic() - started
    mean = sums / samples
    variance = (squares - sums * mean) / (samples - 1)
    return dict(network_id=network_id, seed=seed, samples=samples, elapsed_s=elapsed,
                samples_per_s=samples / elapsed, network_sha256=digest,
                final_mean=mean[-1].tolist(), final_variance=variance[-1].tolist(),
                numpy_version=np.__version__, status="weak pilot; not benchmark reference")


@app.local_entrypoint()
def main():
    out = Path("results/phase2/modal_cpu_wide_pilot_r01.json")
    assert not out.exists()
    rows = list(pilot.map([(n, 52026001 + n * 8 + shard)
                           for n in range(8) for shard in range(8)]))
    out.write_text(json.dumps(dict(rows=rows, cpu_limit=1, memory_limit_mib=1024,
                                  max_tasks=64, task_timeout_s=900, retries=0,
                                  conservative_run_reservation_usd=1.1,
                                  user_budget_cap_usd=30), indent=2) + "\n")
    print(json.dumps([dict(seed=r["seed"], samples_per_s=r["samples_per_s"])
                      for r in rows]))

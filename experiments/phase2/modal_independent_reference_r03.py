"""Bounded CPU throughput/noise pilot; no competition estimator or strong labels."""
import json
from pathlib import Path

import modal

app = modal.App("whest-independent-reference-r03")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==2.2.6").env(
    {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
).add_local_dir("benchmark/data/independent_v30_115", "/data/networks")


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=180,
              retries=0, max_containers=96, scaledown_window=2)
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
    samples = 65536
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
    import os
    import time

    out = Path("results/phase2/modal_independent_reference_r03.json")
    journal = out.with_suffix(".jsonl")
    assert not out.exists() and not journal.exists()
    started = time.monotonic()
    jobs = [(n, 72026001 + n * 128 + shard)
            for n in range(8) for shard in range(128)]
    rows, errors = [], []
    with journal.open("x") as stream:
        for result in pilot.map(jobs, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                record = {"error": type(result).__name__, "detail": str(result)}
                errors.append(record)
            else:
                record = result
                rows.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if (len(rows) + len(errors)) % 32 == 0:
                print(f"saved successes={len(rows)} failures={len(errors)}", flush=True)
    out.write_text(json.dumps(dict(rows=rows, errors=errors,
                                  elapsed_s=time.monotonic() - started,
                                  scope="weak independent pilot; not strong reference",
                                  samples_per_task=65536, max_tasks=1024,
                                  hard_cpu=1, hard_memory_mib=1024,
                                  task_timeout_s=180, retries=0), indent=2) + "\n")

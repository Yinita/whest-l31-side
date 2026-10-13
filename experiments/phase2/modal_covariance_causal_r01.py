"""Offline one-public-network variance attribution; never a submitted estimator."""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

WEIGHT_PATH = Path("benchmark/data/rpc_inputs/mlp_0.npz")
WEIGHT_SHA = "7fb2e3f519b4a90f2c3de9e64d45ad89b7f20ffd254d887e4ba4fa5c1fdf0754"
OUT = Path("results/phase2/modal_covariance_causal_r01")
SAMPLES = 262144
SHARDS = 128
STREAMS = 2
app = modal.App("whest-covariance-causal-r01")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_file(str(WEIGHT_PATH), "/data/network_0.npz")
)


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=300,
              retries=0, max_containers=64, scaledown_window=2)
def moments(job: tuple):
    import numpy as np

    stream, shard, seed = job
    started = time.monotonic()
    try:
        path = Path("/data/network_0.npz")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == WEIGHT_SHA
        weights = np.load(path)["weights"]
        assert weights.shape == (16, 1024, 1024) and weights.dtype == np.float32
        rng = np.random.default_rng(seed)
        # Parallel central-moment merge (Pébay formulas) avoids raw-power
        # cancellation. Axis0: pre/post; axis1: all16 layers.
        mean = np.zeros((2, 16, 1024), dtype=np.float64)
        m2, m3, m4 = np.zeros_like(mean), np.zeros_like(mean), np.zeros_like(mean)
        count = 0

        def update(stage, layer, values, prior):
            y = values.astype(np.float64)
            mb = y.mean(axis=0)
            centered = y-mb
            square = centered*centered
            b2 = square.sum(axis=0)
            b3 = (square*centered).sum(axis=0)
            b4 = (square*square).sum(axis=0)
            a = prior
            b = y.shape[0]
            n = a+b
            delta = mb-mean[stage, layer]
            old2 = m2[stage, layer].copy()
            old3 = m3[stage, layer].copy()
            m4[stage, layer] += (b4+delta**4*a*b*(a*a-a*b+b*b)/n**3
                +6*delta**2*(a*a*b2+b*b*old2)/n**2
                +4*delta*(a*b3-b*old3)/n)
            m3[stage, layer] += (b3+delta**3*a*b*(a-b)/n**2
                +3*delta*(a*b2-b*old2)/n)
            m2[stage, layer] += b2+delta*delta*a*b/n
            mean[stage, layer] += delta*b/n

        for _ in range(SAMPLES//256):
            x = rng.standard_normal((256, 1024), dtype=np.float32)
            for layer, w in enumerate(weights):
                pre = x@w
                update(0, layer, pre, count)
                x = np.maximum(pre, 0)
                update(1, layer, x, count)
            count += x.shape[0]
        buffer = io.BytesIO()
        np.savez_compressed(buffer, mean=mean, m2=m2, m3=m3, m4=m4,
                            count=np.asarray(count, dtype=np.int64))
        return {"stream": stream, "shard": shard, "seed": seed,
                "samples": count, "elapsed_s": time.monotonic()-started,
                "weights_sha256": digest, "numpy_version": np.__version__,
                "statistics": buffer.getvalue(), "complete": True}
    except Exception as exc:
        return {"stream": stream, "shard": shard, "seed": seed,
                "complete": False, "error_type": type(exc).__name__,
                "error": str(exc), "elapsed_s": time.monotonic()-started}


@app.local_entrypoint()
def main():
    assert hashlib.sha256(WEIGHT_PATH.read_bytes()).hexdigest() == WEIGHT_SHA
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    jobs = [(stream, shard, 913000001+stream*10000+shard)
            for stream in range(STREAMS) for shard in range(SHARDS)]
    manifest = {"version": "covariance.causal.r01", "network_id": 0,
                "mlp_name": "logan-fitzgerald", "mlp_seed": 3909487726,
                "weights": str(WEIGHT_PATH), "weights_sha256": WEIGHT_SHA,
                "samples_per_shard": SAMPLES, "shards_per_stream": SHARDS,
                "independent_streams": STREAMS, "jobs": jobs,
                "cpu_per_worker": 1, "max_workers": 64, "memory_mib": 1024,
                "timeout_s": 300, "retries": 0, "pid": os.getpid(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Offline variance attribution; float32 forward/float64 stable central moments; no candidate or stage acceptance"}
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    successful, failures = [], []
    with (OUT/"journal.jsonl").open("x") as journal:
        for result in moments.map(jobs, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                record = {"complete": False, "error_type": type(result).__name__,
                          "error": str(result), "scope": "infrastructure map failure; unknown job"}
            else:
                record = dict(result)
                data = record.pop("statistics", None)
                if data is not None:
                    path = OUT/f"stream{record['stream']}_shard{record['shard']:03d}.npz"
                    assert not path.exists()
                    path.write_bytes(data)
                    record.update(path=str(path), sha256=hashlib.sha256(data).hexdigest())
            (successful if record["complete"] else failures).append(record)
            journal.write(json.dumps(record, allow_nan=False)+"\n")
            journal.flush()
            os.fsync(journal.fileno())
            completed = len(successful)+len(failures)
            if completed % 8 == 0 or not record["complete"]:
                print(f"saved {completed}/{len(jobs)} successes={len(successful)} failures={len(failures)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    final = {**manifest, "successes": successful, "failures": failures,
             "elapsed_s": time.monotonic()-started,
             "complete": len(successful) == len(jobs) and not failures}
    (OUT/"completion.json").write_text(json.dumps(final, indent=2, allow_nan=False)+"\n")
    print(f"completed success={len(successful)} fail={len(failures)}", flush=True)

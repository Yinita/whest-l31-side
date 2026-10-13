"""Offline train-CV learned innovation replay with exact regenerated weights.

This is a frozen first-order mean tangent model, not nonlinear candidate rollout.
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

INPUT = Path("results/phase2/causal_mean_innovation_cv_r44")
ANCHOR = Path("results/phase2/modal_mean_innovation_targets_r43")
OUT = Path("results/phase2/modal_mean_innovation_replay_r45")
IDS = list(range(0, 384, 16))
app = modal.App("whest-mean-innovation-replay-r45")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.2.6")
         .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
         .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}))


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=120,
              retries=0, max_containers=24, scaledown_window=2)
def replay(job):
    import numpy as np
    import torch

    idx, weight_sha, payload = job
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
        g = torch.Generator(device="cpu").manual_seed(770000+idx)
        weights = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32)
                   * (2.0/1024)**.5).numpy()
        actual = hashlib.sha256(weights.tobytes()).hexdigest()
        assert actual == weight_sha
        arrays = np.load(io.BytesIO(payload), allow_pickle=False)
        phi = arrays["phi"]
        outputs = {}
        for name in ["local_hat", "total_hat"]:
            local = arrays[name]
            assert local.shape == phi.shape == (15, 1024)
            assert np.array_equal(local[0], np.zeros(1024))
            current = np.zeros(1024, dtype=np.float64)
            values = []
            for layer in range(15):
                current = (np.zeros_like(current) if layer == 0 else
                           phi[layer]*(weights[layer].astype(np.float64).T@current))+local[layer]
                values.append(current.copy())
            outputs[name] = np.stack(values)
        out = io.BytesIO()
        np.savez_compressed(out, **outputs)
        return dict(complete=True, idx=idx, weights_sha256=actual,
                    elapsed_s=time.monotonic()-started, payload=out.getvalue())
    except Exception as error:
        return dict(complete=False, idx=idx, error_type=type(error).__name__, error=str(error))


@app.local_entrypoint()
def main():
    import numpy as np

    OUT.mkdir(exist_ok=False)
    anchor = json.loads((ANCHOR/"completion.json").read_text())
    weight_hashes = {row["idx"]: row["weights_sha256"] for row in anchor["rows"]}
    jobs, evidence = [], []
    for idx in IDS:
        path = INPUT/f"cv_predictions_idx{idx:04d}.npz"
        payload = path.read_bytes()
        evidence.append(dict(idx=idx, path=str(path), sha256=hashlib.sha256(payload).hexdigest()))
        jobs.append((idx, weight_hashes[idx], payload))
    manifest = dict(pid=os.getpid(), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    inputs=evidence, train_cv_indices=IDS, cpu_per_worker=1,
                    memory_mib=2048, max_workers=24, timeout_s=120, retries=0,
                    scope="Train24 held-network mean tangent replay15layers only; MC labels used solelyoffline localscore; not nonlinearcandidate/rawscore improvement")
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    with (OUT/"journal.jsonl").open("x") as journal:
        for row in replay.map(jobs, order_outputs=False, return_exceptions=True):
            if isinstance(row, Exception):
                row = dict(complete=False, error_type=type(row).__name__, error=str(row))
            if row.get("complete"):
                idx = row["idx"]
                payload = row.pop("payload")
                path = OUT/f"replay_idx{idx:04d}.npz"
                path.write_bytes(payload)
                row.update(path=str(path), payload_sha256=hashlib.sha256(payload).hexdigest())
                original = np.load(INPUT/f"cv_predictions_idx{idx:04d}.npz", allow_pickle=False)
                output = np.load(io.BytesIO(payload), allow_pickle=False)
                error = original["total_error"]
                row["before_layer_mse"] = np.mean(error**2, axis=1).tolist()
                for name in ["local_hat", "total_hat"]:
                    row[name+"_replay_layer_mse"] = np.mean((error-output[name])**2, axis=1).tolist()
            rows.append(row)
            journal.write(json.dumps(row)+"\n")
            journal.flush()
            print("replay", row.get("idx"), row.get("complete"), flush=True)
    good = [row for row in rows if row.get("complete")]
    avg_before = np.mean([row["before_layer_mse"] for row in good], axis=0)
    local = np.mean([row["local_hat_replay_layer_mse"] for row in good], axis=0)
    total = np.mean([row["total_hat_replay_layer_mse"] for row in good], axis=0)
    result = dict(complete=True, rows=rows, failures=len(rows)-len(good),
                  before_layer_mse=avg_before.tolist(), local_replay_layer_mse=local.tolist(),
                  total_replay_layer_mse=total.tolist(), local_over_before=(local/avg_before).tolist(),
                  total_over_before=(total/avg_before).tolist(), scope=manifest["scope"])
    (OUT/"completion.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(last_local_over_before=float(local[-1]/avg_before[-1]),
                          last_total_over_before=float(total[-1]/avg_before[-1]))), flush=True)

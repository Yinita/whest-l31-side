"""Offline train24 tangent deconvolution targets, not an estimator.

Label error vectors are explicitly oracle research inputs; never candidate data.
No H18 rollout is rerun. We regenerate frozen network weights and apply J_l e.
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

PARENT = Path("results/phase2/modal_joint_closure_pilot_r38_r02")
OUT = Path("results/phase2/modal_mean_innovation_targets_r43")
INDICES = list(range(0, 384, 16))
app = modal.App("whest-mean-innovation-targets-r43")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.2.6")
         .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
         .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}))


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=120,
              retries=0, max_containers=24, scaledown_window=2)
def transport(job):
    import resource

    import numpy as np
    import torch

    idx, expected_weight_hash, error_bytes, cdf_bytes = job
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        rng = torch.Generator(device="cpu").manual_seed(770000+idx)
        weights = (torch.randn(16, 1024, 1024, generator=rng, dtype=torch.float32)
                   * (2.0/1024)**.5).numpy()
        weight_hash = hashlib.sha256(weights.tobytes()).hexdigest()
        assert weight_hash == expected_weight_hash
        error = np.frombuffer(error_bytes, dtype=np.float64).reshape(15, 1024)
        cdf = np.frombuffer(cdf_bytes, dtype=np.float64).reshape(15, 1024)
        propagated = np.zeros_like(error)
        for layer in range(1, 15):
            propagated[layer] = cdf[layer]*(weights[layer].astype(np.float64).T@error[layer-1])
        innovation = error-propagated
        buffer = io.BytesIO()
        np.savez_compressed(buffer, idx=np.asarray(idx), innovation=innovation,
                            propagated=propagated, total_error=error)
        return dict(complete=True, idx=idx, weights_sha256=weight_hash, payload=buffer.getvalue(),
                    elapsed_s=time.monotonic()-start, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    numpy_version=np.__version__, torch_version=torch.__version__)
    except Exception as error:
        return dict(complete=False, idx=idx, error_type=type(error).__name__,
                    error=str(error), elapsed_s=time.monotonic()-start)


@app.local_entrypoint()
def main():
    import numpy as np

    OUT.mkdir(exist_ok=False)
    manifest = json.loads((PARENT/"manifest.json").read_text())
    assert manifest["train_indices"] == INDICES
    labels = np.load(PARENT/"research_labels.npz", allow_pickle=False)
    train_truth = labels["moments"][:24, 0, :15].astype(np.float64)
    hashes = {row["idx"]: row["weights_sha256"] for row in
              map(json.loads, (PARENT/"journal.jsonl").read_text().splitlines())
              if row.get("complete") and row.get("phase") in ["smoke", "baseline"]}
    jobs, records = [], []
    for position, idx in enumerate(INDICES):
        path = PARENT/f"baseline_idx{idx:04d}.npz"
        data = np.load(path, allow_pickle=False)
        assert np.array_equal(data["states"][..., 0], data["prediction"][:15])
        error = train_truth[position]-data["prediction"][:15].astype(np.float64)
        cdf = data["features"][..., 7].astype(np.float64)
        jobs.append((idx, hashes[idx], error.tobytes(), cdf.tobytes()))
        records.append(dict(idx=idx, baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            error_vector_sha256=hashlib.sha256(error.tobytes()).hexdigest()))
    info = dict(version="mean.innovation.targets.r43", pid=os.getpid(),
                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                train_indices=INDICES, source_model_sha256=manifest["h18_sha256"],
                records=records, cpu_per_worker=1, memory_mib=2048, max_workers=24,
                timeout_s=120, retries=0,
                formula="innovation_l=e_l-Phi_l*(W_l.T@e_(l-1)); e=MCpostmean-H18actualpostmean",
                scope="Offline oracle train24 target deconvolution only; firstorder frozenmeanJacobian excludes covariance/source responses; no candidate/oracle score")
    (OUT/"manifest.json").write_text(json.dumps(info, indent=2)+"\n")
    rows = []
    with (OUT/"journal.jsonl").open("x") as journal:
        for row in transport.map(jobs, order_outputs=False, return_exceptions=True):
            if isinstance(row, Exception):
                row = dict(complete=False, error_type=type(row).__name__, error=str(row))
            if row.get("complete"):
                payload = row.pop("payload")
                path = OUT/f"train_idx{row['idx']:04d}.npz"
                path.write_bytes(payload)
                row.update(path=str(path), payload_sha256=hashlib.sha256(payload).hexdigest())
            rows.append(row)
            journal.write(json.dumps(row)+"\n")
            journal.flush()
            print("transport", row.get("idx"), row.get("complete"), flush=True)
    (OUT/"completion.json").write_text(json.dumps(dict(complete=True, rows=rows,
        failures=sum(not row["complete"] for row in rows), scope=info["scope"]), indent=2)+"\n")

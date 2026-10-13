"""Prepared R57 actual lawful whole rollouts, new unseen corpus8, no auto launch.

One frozen full14 R56 model/candidate only, baseline and candidate sequentially
in the same worker/weights. Labels fetched separately and scored locally.
16GiB diagnostic, cumulative RSS/time not strict resource acceptance.
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

SOURCE = Path("estimators/sprint_public_v37b_h18_r01/estimator.py")
SOURCE_SHA = "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
DATA_REV = "cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8"
DATA_URL = f"https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k/resolve/{DATA_REV}/shard00.npz?download=true"
DATA_SHA = "aab6930a8a82b9dfe792a0e71456873165f73d143faac06fc32e5ca52b2b0fa5"
INDICES = list(range(768, 896, 16))
OUT = Path("results/phase2/onpolicy_closed_fresh8_r57")
app = modal.App("whest-onpolicy-closed-fresh8-r57")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "flopscope==0.12.1", "whestbench==0.16.1")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_file(str(SOURCE), "/code/h18.py")
)


@app.function(image=image, cpu=(1, 1), memory=(16384, 16384), timeout=300,
              retries=0, max_containers=1, scaledown_window=2)
def read_labels():
    import urllib.request

    import numpy as np

    data = urllib.request.urlopen(DATA_URL, timeout=60).read()
    assert hashlib.sha256(data).hexdigest() == DATA_SHA
    arrays = np.load(io.BytesIO(data), allow_pickle=False)
    ids = arrays["idx"]
    assert ids[48:56].tolist() == INDICES
    assert int(arrays["n_done"]) == 128 and int(arrays["n_samples"]) == 100000000
    buffer = io.BytesIO()
    np.savez_compressed(buffer, idx=ids[48:56], final_mean=arrays["final_mean"][48:56],
                        first_mean=arrays["moments"][48:56, 0, 0])
    return buffer.getvalue()


@app.function(image=image, cpu=(1, 1), memory=(16384, 16384), timeout=600,
              retries=0, max_containers=8, scaledown_window=2)
def paired_rollout(job):
    import gc
    import importlib.metadata
    import importlib.util
    import platform
    import re
    import resource
    import tempfile
    from types import SimpleNamespace

    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np
    import torch
    from whestbench import SetupContext

    assert set(job) == {"idx", "candidate_source", "candidate_sha256", "support_bytes",
                        "support_sha256", "support_metadata", "model_sha256"}
    idx = job["idx"]
    assert idx in INDICES
    started = time.monotonic()
    rows = []
    try:
        for key in tuple(os.environ):
            if re.match(r"V\d+_", key):
                os.environ.pop(key)
        assert hashlib.sha256(Path("/code/h18.py").read_bytes()).hexdigest() == SOURCE_SHA
        assert hashlib.sha256(job["candidate_source"]).hexdigest() == job["candidate_sha256"]
        assert hashlib.sha256(job["support_bytes"]).hexdigest() == job["support_sha256"]
        metadata = json.loads(job["support_metadata"])
        assert metadata["support_sha256"] == job["support_sha256"]
        assert metadata["model_sha256"] == job["model_sha256"]
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu").manual_seed(770000 + idx)
        weights = (torch.randn(16, 1024, 1024, generator=generator, dtype=torch.float32)
                   * (2.0/1024)**.5).numpy()
        weights_hash = hashlib.sha256(weights.tobytes()).hexdigest()
        predictions = {}
        with tempfile.TemporaryDirectory(prefix="r57-frozen-support-") as temporary:
            folder = Path(temporary)
            candidate = folder / "estimator.py"
            candidate.write_bytes(job["candidate_source"])
            (folder / "coefficients.npz").write_bytes(job["support_bytes"])
            (folder / "coefficients.json").write_text(job["support_metadata"])
            phases = [("baseline", Path("/code/h18.py"))]
            if idx == INDICES[0]:
                phases.append(("baseline_control", Path("/code/h18.py")))
            phases.append(("candidate", candidate))
            for phase, source in phases:
                gc.collect()
                spec = importlib.util.spec_from_file_location(f"r57_{phase}", source)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                estimator = module.Estimator()
                setup_start = time.monotonic()
                with flops.BudgetContext(flop_budget=2**41, quiet=True) as setup_context:
                    estimator.setup(SetupContext(width=1024, depth=16, flop_budget=2**41,
                                                 api_version="0.1", seed=0))
                setup_seconds = time.monotonic()-setup_start
                with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
                    mlp = SimpleNamespace(width=1024, weights=[fnp.asarray(w) for w in weights])
                    prediction = np.asarray(estimator.predict(mlp, 2**41)).copy()
                assert prediction.shape == (16, 1024) and np.isfinite(prediction).all()
                predictions[phase] = prediction
                rows.append(dict(phase=phase, complete=True, cost=context.summary_dict(),
                    setup_cost=setup_context.summary_dict(), setup_wall_s=setup_seconds,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    maxrss_cumulative_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    negative_final_outputs=int((prediction[-1]<0).sum())))
                del mlp, estimator, module
        first_sigma = np.sqrt((weights[0].astype(np.float64)**2).sum(axis=0))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, idx=idx, **predictions,
                            analytic_first_mean=first_sigma/np.sqrt(2*np.pi),
                            analytic_first_variance=first_sigma**2*(.5-1/(2*np.pi)))
        control = None
        if idx == INDICES[0]:
            difference = predictions["baseline_control"].astype(np.float64)-predictions["baseline"].astype(np.float64)
            baseline_row, control_row = rows[0], rows[1]
            assert baseline_row["source_sha256"] == control_row["source_sha256"]
            assert baseline_row["cost"]["flops_used"] == control_row["cost"]["flops_used"]
            control = dict(fixed_idx=idx, baseline_duplicate_sameworker=True,
                source_and_weight_identity_equal=True, metered_flops_equal=True,
                maxabs_by_layer=np.max(abs(difference), axis=1).tolist(),
                rms_by_layer=np.sqrt(np.mean(difference**2, axis=1)).tolist(),
                final_prediction_difference_mse=float(np.mean(difference[-1]**2)),
                scope="single fixed sameworker numeric/cachingcontrol; not gain or scoreacceptance")
        return dict(idx=idx, complete=True, rows=rows, weights_sha256=weights_hash,
            weight_seed=770000+idx, baseline_sha256=SOURCE_SHA,
            candidate_sha256=job["candidate_sha256"], model_sha256=job["model_sha256"],
            same_worker_same_weights=True, label_reference_absent=True,
            trace_or_oracle_injection_absent=True, rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)),
            versions={k: importlib.metadata.version(k) for k in ("numpy", "flopscope", "whestbench", "torch")},
            backend_processor=platform.processor(), machine=platform.machine(),
            fixed_numeric_control=control,
            elapsed_s=time.monotonic()-started, payload=buffer.getvalue())
    except Exception as error:
        return dict(idx=idx, complete=False, rows=rows, error_type=type(error).__name__,
                    error=str(error), elapsed_s=time.monotonic()-started)


@app.local_entrypoint()
def main(candidate_dir: str, model_path: str, model_sha256: str,
         candidate_sha256: str, support_sha256: str):
    import numpy as np

    model_file, directory = Path(model_path), Path(candidate_dir)
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == model_sha256
    model = json.loads(model_file.read_text())
    assert model["train_indices"] == list(range(0, 384, 16))
    assert model["fitted_layers"] == list(range(1, 15))
    assert set(model["train_indices"]).isdisjoint(INDICES)
    assert model["training_policy"] == "sequential_on_policy_frozen_previous_heads"
    candidate_file, support_file = directory / "estimator.py", directory / "coefficients.npz"
    assert hashlib.sha256(candidate_file.read_bytes()).hexdigest() == candidate_sha256
    assert hashlib.sha256(support_file.read_bytes()).hexdigest() == support_sha256
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    assert not OUT.exists()
    OUT.mkdir()
    frozen = dict(candidate_source=candidate_file.read_bytes(), candidate_sha256=candidate_sha256,
        support_bytes=support_file.read_bytes(), support_sha256=support_sha256,
        support_metadata=support_file.with_suffix(".json").read_text(), model_sha256=model_sha256)
    manifest = dict(version="r57-actual-lawful-onpolicy-closed-fresh8", pid=os.getpid(),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_sha256=SOURCE_SHA, candidate_sha256=candidate_sha256,
        model_sha256=model_sha256, support_sha256=support_sha256,
        train_indices=model["train_indices"], independent_indices=INDICES,
        dataset_revision=DATA_REV, shard_sha256=DATA_SHA, thread_count=1,
        flop_budget=2**41, development_memory_mib=16384, max_workers=8, retries=0,
        single_change="actualfnpcandidatefrozen14onpolicyclosure versus originalH18 sameworkerW",
        labels_model_boundary="only frozen candidate source/support to predictionworker; MC refs isolatedlocalscore",
        scope="one complete actual16layer pairedrawdiagnostic; sameprocess cumulativeRSS/cacheorderedtiming; no strictRPC/resource/adjusted/stage acceptance",
        numeric_control="fixedidx768 baselineA→baselineB→candidate sameworker; other7 baselineA→candidate; scorebeforealwaysA",
        fixed_raw_gate="zero failures, aggregatecandidate/baseline<.90 and pairednetworkbootstrap95upper<1; tenfold separately<.1")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Reference reading begins only after model/candidate identities are frozen.
    labels_bytes = read_labels.remote()
    (OUT / "private_reference.npz").write_bytes(labels_bytes)
    labels = np.load(io.BytesIO(labels_bytes), allow_pickle=False)
    rows = []
    with (OUT / "journal.jsonl").open("x") as journal:
        for row in paired_rollout.map([dict(idx=idx, **frozen) for idx in INDICES],
                                      order_outputs=False, return_exceptions=True):
            if isinstance(row, Exception):
                row = dict(complete=False, error_type=type(row).__name__, error=str(row))
            if row.get("complete"):
                payload = row.pop("payload")
                path = OUT / f"paired_idx{row['idx']:04d}.npz"
                path.write_bytes(payload)
                trace = np.load(io.BytesIO(payload), allow_pickle=False)
                position = INDICES.index(row["idx"])
                truth = labels["final_mean"][position].astype(np.float64)
                before = float(np.mean((trace["baseline"][-1].astype(np.float64)-truth)**2))
                after = float(np.mean((trace["candidate"][-1].astype(np.float64)-truth)**2))
                row.update(path=str(path), payload_sha256=hashlib.sha256(payload).hexdigest(),
                    before_raw=before, after_raw=after, after_over_before=after/before,
                    first_analytic_mc_noise=float(np.mean((trace["analytic_first_mean"]-
                        labels["first_mean"][position])**2)/(trace["analytic_first_variance"].mean()/1e8)),
                    first_layer_cross_prediction_maxabs=float(np.max(abs(trace["candidate"][0]-trace["baseline"][0]))))
                if row["idx"] == INDICES[0]:
                    duplicate_raw = float(np.mean((trace["baseline_control"][-1].astype(np.float64)-truth)**2))
                    row["fixed_numeric_control"].update(baseline_A_raw=before, baseline_B_raw=duplicate_raw,
                        B_over_A_raw=duplicate_raw/before,
                        delta_RMS_over_baseline_error_RMS=float(np.sqrt(
                            row["fixed_numeric_control"]["final_prediction_difference_mse"]/before)))
            rows.append(row)
            journal.write(json.dumps(row)+"\n")
            journal.flush()
            print(row.get("idx"), row.get("complete"), row.get("after_over_before"), flush=True)
    success = [r for r in rows if r.get("complete")]
    failures = len(rows)-len(success)
    result = dict(complete=True, failures=failures, rows=rows, scope=manifest["scope"])
    if len(success) == 8:
        before = np.asarray([r["before_raw"] for r in success])
        after = np.asarray([r["after_raw"] for r in success])
        sample = np.random.default_rng(570013).integers(0, 8, size=(20000, 8))
        boot = after[sample].mean(axis=1)/before[sample].mean(axis=1)
        bounds = np.quantile(boot, [.025, .975]).tolist()
        ratio = float(after.mean()/before.mean())
        result.update(mean_before_raw=float(before.mean()), mean_after_raw=float(after.mean()),
                      after_over_before=ratio, network_wins=int((after<before).sum()),
                      paired_network_bootstrap95=bounds, raw_gate_passed=ratio<.9 and bounds[1]<1,
                      raw_tenfold_observed=ratio<.1, stage_accepted=False)
    (OUT / "completion.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))

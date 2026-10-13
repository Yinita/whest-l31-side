"""Isolated R60 infrastructure diagnostic of failed R54 layer8 capture.

Not an admissible estimator. Trace/NumPy work is explicitly offline research.
Targets enter only local fitting/scoring, never remote rollout workers.
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

SOURCE = Path("estimators/sprint_public_v37b_h18_r01/estimator.py")
HELPER_HASHES = {
    "on_policy_capture_helper_r54.py": "3e8f31f81d9f2eae0cd8a173c5a0145b9df01d44cac4df4ec9d9b58d11a5efee",
    "tenfold_local_features_r48.py": "584626237242e84acb696aaf51b71724ad05db3007d70b475514819bc0b61bcf",
}
SOURCE_SHA = "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
DATA_REV = "cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8"
DATA_URL = f"https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k/resolve/{DATA_REV}/shard00.npz?download=true"
DATA_SHA = "aab6930a8a82b9dfe792a0e71456873165f73d143faac06fc32e5ca52b2b0fa5"
OUT = Path("results/phase2/modal_onpolicy_isolated_r60")
INDICES = list(range(0, 384, 16))
FEATURES = [
    "ones",
    "mu",
    "var",
    "sigma",
    "alpha",
    "abs_alpha",
    "phi",
    "Phi",
    "pk1",
    "D3",
    "D21norm",
    "K4_sigma",
    "sigma_phi",
    "skew_H2phi",
    "kurt_H3phi",
    "skew2_H5phi",
    "skewkurt_H6phi",
    "kurt2_H7phi",
    "skew",
    "kurt",
    "gate",
    "depth_fraction",
]
raw_volume = modal.Volume.from_name("whest-onpolicy-r54-raw", create_if_missing=True)
app = modal.App("whest-onpolicy-isolated-r60")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "flopscope==0.12.1", "whestbench==0.16.1")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_file(str(SOURCE), "/code/h18.py")
    .add_local_file(
        "experiments/phase2/on_policy_capture_helper_r54.py",
        "/code/on_policy_capture_helper_r54.py",
    )
    .add_local_file(
        "experiments/phase2/tenfold_local_features_r48.py", "/code/tenfold_local_features_r48.py"
    )
)


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(16384, 16384),
    timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/raw": raw_volume},
)
def rollout(job):
    import importlib.metadata
    import importlib.util
    import re
    import resource
    import sys
    from types import SimpleNamespace

    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np
    import torch
    from whestbench import SetupContext

    # No truth argument: only source-defined weights and frozen prefix heads.
    assert set(job) == {
        "idx",
        "expected_weight_hash",
        "target_layer",
        "prefix_model",
        "prefix_sha256",
        "branch",
        "role",
    }
    idx, model = job["idx"], job["prefix_model"]
    current = job["target_layer"]
    assert 1 <= current <= 14
    assert set(model["layer_models"]) == {str(k) for k in range(1, current)}
    assert set(model["train_indices"]).issubset(INDICES)
    if job["role"] == "held":
        assert idx not in model["train_indices"]
    else:
        assert idx in model["train_indices"]
    branch_held = (
        list(range(int(job["branch"][-1]) * 96, (int(job["branch"][-1]) + 1) * 96, 16))
        if job["branch"].startswith("fold")
        else []
    )
    branch_train = [value for value in INDICES if value not in branch_held]
    assert model["train_indices"] == branch_train
    assert idx in (branch_held if job["role"] == "held" else branch_train)
    expected_prefix_hash = hashlib.sha256(
        json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert expected_prefix_hash == job["prefix_sha256"]
    started = time.monotonic()
    try:
        for key in tuple(os.environ):
            if re.match(r"V\d+_", key):
                os.environ.pop(key)
        phases = []

        def boundary(name):
            value = dict(boundary=name, elapsed_s=time.monotonic() - started)
            phases.append(value)
            print(json.dumps(dict(idx=idx, target=current, **value)), flush=True)

        boundary("worker_started")
        source = Path("/code/h18.py")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
        spec = importlib.util.spec_from_file_location("h18_corpus_r38", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        flops.configure(symmetry_warnings=False)  # Offline warning suppression only.
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu").manual_seed(770000 + idx)
        weights = (
            torch.randn(16, 1024, 1024, generator=generator, dtype=torch.float32)
            * (2.0 / 1024) ** 0.5
        ).numpy()
        boundary("weights_generated")
        weights_sha = hashlib.sha256(weights.tobytes()).hexdigest()
        assert weights_sha == job["expected_weight_hash"]
        estimator = module.Estimator()
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            estimator.setup(
                SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0)
            )
        boundary("setup_complete")
        lines = source.read_text().splitlines()
        pretrigger = next(
            i for i, line in enumerate(lines, 1) if "if riders and not NO_CORR:" in line
        )
        posttrigger = next(
            i
            for i, line in enumerate(lines, 1)
            if "mu = K1v if delta is None else K1v + delta" in line
        )
        sys.path.insert(0, "/code")
        from on_policy_capture_helper_r54 import CaptureComplete, OnPolicyCapture

        for name, expected in HELPER_HASHES.items():
            assert hashlib.sha256(Path("/code", name).read_bytes()).hexdigest() == expected
        prefix = {
            int(layer): {key: fnp.asarray(value, dtype=fnp.float64) for key, value in head.items()}
            for layer, head in model["layer_models"].items()
        }
        observer = OnPolicyCapture(1024, current, prefix)
        pre_seen, post_seen = set(), set()

        def tracer(frame, event, arg):
            if event != "line" or frame.f_code.co_filename != str(source):
                return tracer
            if frame.f_lineno not in (pretrigger, posttrigger):
                return tracer
            state = frame.f_locals
            layer = state.get("li", -1)
            assert 0 <= layer <= current
            if frame.f_lineno == pretrigger:
                assert layer not in pre_seen
                pre_seen.add(layer)
                observer.record_pre(state)
            else:
                assert layer not in post_seen
                post_seen.add(layer)
                boundary(f"before_post_layer{layer}")
                observer.record_post(state)
            return tracer

        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            mlp = SimpleNamespace(width=1024, weights=[fnp.asarray(w) for w in weights])
            sys.settrace(tracer)
            try:
                estimator.predict(mlp, 2**41)
            except CaptureComplete:
                pass
            finally:
                sys.settrace(None)
        boundary("compute_capture_complete")
        assert observer.current is not None
        assert observer.injected == list(range(1, current))
        capture = {
            "features26": observer.current["features"],
            "state4": observer.current["states"],
            "scales4": observer.current["scales"],
            "inherited_delta": observer.current["inherited_delta"],
        }
        assert sorted(pre_seen) == sorted(post_seen) == list(range(current + 1))
        assert capture["features26"].shape == (1024, 26)
        assert capture["state4"].shape == capture["scales4"].shape == (1024, 4)
        assert all(np.isfinite(value).all() for value in capture.values())
        assert np.min(capture["scales4"]) > 0
        permute = module.PRUNE
        rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        address_limit = list(resource.getrlimit(resource.RLIMIT_AS))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, idx=np.asarray(idx), layer=np.asarray(current), **capture)
        boundary("serialization_complete")
        payload = buffer.getvalue()
        archive = (
            Path("/raw/r60_isolated")
            / f"layer{current}"
            / job["branch"]
            / job["role"]
            / f"idx{idx:04d}.npz"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        assert not archive.exists(), "No raw evidence overwrite"
        boundary("before_volume_write")
        archive.write_bytes(payload)
        boundary("before_volume_commit")
        raw_volume.commit()
        boundary("after_volume_commit")
        return {
            "complete": True,
            "diagnostic_boundaries": phases,
            "raw_archive_path": str(archive),
            "raw_archive_volume": "whest-onpolicy-r54-raw",
            "raw_archive_sha256": hashlib.sha256(payload).hexdigest(),
            "idx": idx,
            "target_layer": current,
            "branch": job["branch"],
            "role": job["role"],
            "prefix_sha256": job["prefix_sha256"],
            "truth_argument_received": False,
            "full_setup_depth": 16,
            "stopped_before_current_injection": True,
            "full_prediction_claimed": False,
            "weight_seed": 770000 + idx,
            "weights_sha256": weights_sha,
            "h18_sha256": SOURCE_SHA,
            "elapsed_s": time.monotonic() - started,
            "cost": context.summary_dict(),
            "prefix_audit": observer.audit,
            "prefix_head_layers": observer.injected,
            "permutation_flag": permute,
            "maxrss_kib": rss_kib,
            "rlimit_as": address_limit,
            "captured_pre_layers": sorted(pre_seen),
            "captured_post_layers": sorted(post_seen),
            "versions": {
                k: importlib.metadata.version(k)
                for k in ("numpy", "flopscope", "whestbench", "torch")
            },
            "payload": payload,
        }
    except Exception as error:
        return {
            "complete": False,
            "idx": idx,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_s": time.monotonic() - started,
        }


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@app.local_entrypoint()
def main():
    original = Path("results/phase2/modal_onpolicy_local_joint_r54")
    prefix_path = original / "full24_prefix_through7.json"
    prefix = json.loads(prefix_path.read_text())
    manifest = json.loads((original / "manifest.json").read_text())
    weight = next(
        row["weights_sha256"] for row in manifest["baseline_provenance"] if row["idx"] == 320
    )
    job = dict(
        idx=320,
        expected_weight_hash=weight,
        target_layer=8,
        prefix_model=prefix,
        prefix_sha256=canonical_hash(prefix),
        branch="full24",
        role="train",
    )
    OUT.mkdir(exist_ok=False)
    output_manifest = dict(
        version="r60-isolated-r54-timeout-infrastructure-diagnostic",
        pid=os.getpid(),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_source_sha256=manifest["source_sha256"],
        original_failure_sha256=hashlib.sha256(
            (original / "failure_layer8.json").read_bytes()
        ).hexdigest(),
        prefix_file_sha256=hashlib.sha256(prefix_path.read_bytes()).hexdigest(),
        prefix_canonical_sha256=job["prefix_sha256"],
        helper_hashes=HELPER_HASHES,
        job={key: value for key, value in job.items() if key != "prefix_model"},
        cpu_per_worker=1,
        max_containers=1,
        memory_mib=16384,
        timeout_s=300,
        retries=0,
        single_change="isolatefailednetwork and logcompute/serialize/Volumecommit boundaries; same frozenmath",
        scope="newinfra diagnostic identity; does not rewrite original R54 failure or achievedscore",
    )
    (OUT / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    try:
        row = rollout.remote(job)
    except Exception as error:
        row = dict(complete=False, error_type=type(error).__name__, error=str(error))
    if row.get("complete"):
        payload = row.pop("payload")
        path = OUT / "isolated_full24_idx0320_layer8.npz"
        path.write_bytes(payload)
        row.update(path=str(path), payload_sha256=hashlib.sha256(payload).hexdigest())
    (OUT / "completion.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({key: value for key, value in row.items() if key != "cost"}), flush=True)


if __name__ == "__main__":
    raise SystemExit("Launch diagnostic with Modal; no local torch runtime")

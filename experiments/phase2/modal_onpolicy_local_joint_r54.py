"""Sequential on-policy local closure fixed14-layer run.

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
OUT = Path("results/phase2/modal_onpolicy_local_joint_r54")
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
app = modal.App("whest-onpolicy-local-joint-r54")
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
    max_containers=96,
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
        weights_sha = hashlib.sha256(weights.tobytes()).hexdigest()
        assert weights_sha == job["expected_weight_hash"]
        estimator = module.Estimator()
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            estimator.setup(
                SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0)
            )
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
        payload = buffer.getvalue()
        archive = (
            Path("/raw/r54_r01")
            / f"layer{current}"
            / job["branch"]
            / job["role"]
            / f"idx{idx:04d}.npz"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        assert not archive.exists(), "No raw evidence overwrite"
        archive.write_bytes(payload)
        raw_volume.commit()
        return {
            "complete": True,
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


def fit_head(data, targets, train_ids):
    import numpy as np

    x = np.concatenate([data[idx]["features26"] for idx in train_ids])
    y = np.concatenate(
        [(targets[idx] - data[idx]["state4"]) / data[idx]["scales4"] for idx in train_ids]
    )
    mean, scale = x.mean(axis=0), x.std(axis=0)
    mean[0], scale[0] = 0, 1
    scale = np.maximum(scale, 1e-12)
    normalized = (x - mean) / scale
    penalty = np.eye(26) * 0.01 * len(x)
    penalty[0, 0] = 0
    theta = np.linalg.solve(normalized.T @ normalized + penalty, normalized.T @ y)
    return dict(feature_mean=mean.tolist(), feature_scale=scale.tolist(), theta=theta.tolist())


def eval_current(datum, target, head, h18_state):
    import numpy as np

    x = (datum["features26"] - np.asarray(head["feature_mean"])) / np.asarray(head["feature_scale"])
    change = (x @ np.asarray(head["theta"])) * datum["scales4"]
    after = datum["state4"] + change
    clipped_mean = int((after[:, 0] < 0).sum())
    clipped_var = int((after[:, 1] < 1e-10).sum())
    after[:, 0] = np.maximum(after[:, 0], 0)
    after[:, 1] = np.maximum(after[:, 1], 1e-10)
    ideal = after.copy()
    if "inherited_delta" in datum:
        delta = datum["inherited_delta"]
        after[:, 0] = (after[:, 0] - delta).astype(np.float32).astype(np.float64) + delta
        after[:, 1:] = after[:, 1:].astype(np.float32).astype(np.float64)
        rounding_scope = "actualinheriteddelta fp32K1copy followedf64meanplusdelta; fp32K2K3K4"
    else:
        # Layer1 delta is carried from analytic layer0; R38 never retained it.
        # Report float32 rounding proxy and this explicit limitation; realprefix
        # workers capture actual inheriteddelta for layers2..14.
        after = after.astype(np.float32).astype(np.float64)
        rounding_scope = "layer1 baseline lacksinheriteddelta; fp32totalmeanproxy; notexactK1copy"
    quantization = after - ideal
    errors = {
        "zero_current": target - datum["state4"],
        "head_current": target - after,
        "h18": target - h18_state,
    }
    return dict(
        normalized_mse={
            k: np.mean((v / datum["scales4"]) ** 2, axis=0).tolist() for k, v in errors.items()
        },
        physical_mse={k: np.mean(v**2, axis=0).tolist() for k, v in errors.items()},
        clipped_mean=clipped_mean,
        clipped_var=clipped_var,
        applied_poststate_min_per_channel=after.min(axis=0).tolist(),
        applied_poststate_max_per_channel=after.max(axis=0).tolist(),
        fp32_quantization_maxabs_per_channel=np.max(abs(quantization), axis=0).tolist(),
        fp32_quantization_mse_per_channel=np.mean(quantization**2, axis=0).tolist(),
        local_evaluation=rounding_scope + "; no finalscore",
    )


@app.local_entrypoint()
def main():
    import numpy as np

    contract_path = Path("results/phase2/on_policy_capture_contract_r54_r02.json")
    contract = json.loads(contract_path.read_text())
    assert contract["complete"] and len(contract["rows"]) == 8
    assert all(
        row["currentcapture_equal_precorrectionstate"]
        and row["currentfuturehead_rejected"]
        and row["tracer_removed"]
        for row in contract["rows"]
    )
    for name, expected in HELPER_HASHES.items():
        assert hashlib.sha256(Path("experiments/phase2", name).read_bytes()).hexdigest() == expected

    parent = Path("results/phase2/modal_joint_closure_pilot_r38_r02")
    assert not OUT.exists()
    original = json.loads((parent / "manifest.json").read_text())
    assert original["h18_sha256"] == SOURCE_SHA
    assert original["dataset_revision"] == DATA_REV
    assert original["dataset_shard_sha256"] == DATA_SHA
    assert original["train_indices"] == INDICES
    assert original["feature_order"] == FEATURES
    assert original["train_layers"] == list(range(15))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    labels_path = parent / "research_labels.npz"
    labels = np.load(labels_path, allow_pickle=False)
    assert labels["idx"][:24].tolist() == INDICES
    assert labels["moments"].ndim == 4 and labels["moments"].shape[0] >= 24
    assert labels["moments"].shape[1:] == (4, 16, 1024)
    raw = np.moveaxis(labels["moments"][:24, :, :15].astype(np.float64), 1, -1)
    m1, m2, m3, m4 = [raw[..., c] for c in range(4)]
    var = m2 - m1 * m1
    central = np.stack(
        [
            m1,
            var,
            m3 - 3 * m1 * m2 + 2 * m1**3,
            m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1**4 - 3 * var**2,
        ],
        axis=-1,
    )
    assert np.isfinite(central).all()
    baseline = {
        row["idx"]: row
        for row in map(json.loads, (parent / "journal.jsonl").read_text().splitlines())
        if row.get("complete") and row.get("phase") in ("baseline", "smoke")
    }
    baseline_data = {}
    provenance = []
    for idx in INDICES:
        row = baseline[idx]
        path = parent / f"baseline_idx{idx:04d}.npz"
        assert row["h18_sha256"] == SOURCE_SHA
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["payload_sha256"]
        trace = np.load(path, allow_pickle=False)
        assert int(trace["idx"]) == idx
        pre, state, scales = [
            trace[key].astype(np.float64) for key in ("features", "states", "scales")
        ]
        assert pre.shape == (15, 1024, 22) and state.shape == scales.shape == (15, 1024, 4)
        baseline_data[idx] = dict(
            features26=np.concatenate([pre, state / scales], axis=-1), state4=state, scales4=scales
        )
        provenance.append(
            dict(idx=idx, trace_sha256=row["payload_sha256"], weights_sha256=row["weights_sha256"])
        )
    branches = {}
    for fold in range(4):
        held = INDICES[fold * 6 : (fold + 1) * 6]
        train = [idx for idx in INDICES if idx not in held]
        branches[f"fold{fold}"] = dict(
            train=train, held=held, prefix=dict(train_indices=train, layer_models={})
        )
    branches["full24"] = dict(
        train=INDICES.copy(), held=[], prefix=dict(train_indices=INDICES.copy(), layer_models={})
    )
    OUT.mkdir()
    manifest = dict(
        version="onpolicy-localjoint-r54-fixed1to14",
        pid=os.getpid(),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        h18_sha256=SOURCE_SHA,
        helper_hashes=HELPER_HASHES,
        component_contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        dataset_revision=DATA_REV,
        dataset_shard_sha256=DATA_SHA,
        labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        train_indices=INDICES,
        branches={k: dict(train=v["train"], held=v["held"]) for k, v in branches.items()},
        ridge_lambda=0.01,
        features=FEATURES
        + [
            "own_post_mean_over_sigma",
            "own_post_variance_over_sigma2",
            "own_post_kappa3_over_sigma3",
            "own_post_kappa4_over_sigma4",
        ],
        layer1_reuses_frozen_baseline=True,
        cloud_layers=list(range(2, 15)),
        prefix_depth_semantics=16,
        cpu_per_worker=1,
        max_containers=96,
        memory_mib=16384,
        timeout_s=300,
        retries=0,
        threads=1,
        pilot_gate="layers1..3 integrity/finite/prefix-boundary/floors only; earlyMC-noise forbids accuracy stop",
        accuracy_diagnostic="full1..14 CV localmu/var diagnostic; futurefreshclosedscore determines gain; no paramchanges",
        startup_noise_evidence_sha256=hashlib.sha256(
            Path("results/phase2/onpolicy_early_late_noise_r54.json").read_bytes()
        ).hexdigest(),
        raw_archive_volume="whest-onpolicy-r54-raw",
        raw_archive_run="r54_r01",
        local_disk_reserve_bytes=150 * 2**20,
        baseline_provenance=provenance,
        scope="offlineonpolicytrain24outerfoldfixed14; lawfulprefixhelper/earlystop; nofinalraw/adjusted/resource/stage",
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    all_evaluations = []
    with (OUT / "journal.jsonl").open("x") as journal:
        for layer in range(1, 15):
            current = {}
            jobs = []
            for branch, config in branches.items():
                if layer == 1:
                    current[branch] = {
                        idx: {k: value[layer] for k, value in baseline_data[idx].items()}
                        for idx in config["train"] + config["held"]
                    }
                else:
                    current[branch] = {}
                    for role, ids in [("train", config["train"]), ("held", config["held"])]:
                        for idx in ids:
                            jobs.append(
                                dict(
                                    idx=idx,
                                    expected_weight_hash=baseline[idx]["weights_sha256"],
                                    target_layer=layer,
                                    prefix_model=config["prefix"],
                                    prefix_sha256=canonical_hash(config["prefix"]),
                                    branch=branch,
                                    role=role,
                                )
                            )
            if layer > 1:
                # Each role is a separate bounded wave. Both use the frozen prior
                # prefix, never the current head fitted later in this layer.
                wave_jobs = []
                if layer == 2:
                    smoke = [
                        job for job in jobs if job["branch"] == "fold0" and job["role"] == "train"
                    ][:2]
                    smoke_keys = {(job["branch"], job["role"], job["idx"]) for job in smoke}
                    wave_jobs.append(("smoke2", smoke))
                else:
                    smoke_keys = set()
                for role in ("train", "held"):
                    wave_jobs.append(
                        (
                            role,
                            [
                                job
                                for job in jobs
                                if job["role"] == role
                                and (job["branch"], job["role"], job["idx"]) not in smoke_keys
                            ],
                        )
                    )
                for wave, role_jobs in wave_jobs:
                    wave_rows = []
                    for row in rollout.map(role_jobs, order_outputs=False, return_exceptions=True):
                        if isinstance(row, Exception):
                            row = dict(
                                complete=False, error_type=type(row).__name__, error=str(row)
                            )
                        if not row.get("complete"):
                            journal.write(json.dumps(row) + "\n")
                            journal.flush()
                            (OUT / f"failure_layer{layer}.json").write_text(
                                json.dumps(row, indent=2) + "\n"
                            )
                            raise RuntimeError(
                                "R54 failedcapture; retain evidence, no branch expansion"
                            )
                        payload = row.pop("payload")
                        folder = OUT / f"layer{layer}" / row["branch"] / row["role"]
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / f"idx{row['idx']:04d}.npz"
                        assert not path.exists()
                        import shutil

                        local_store = shutil.disk_usage(OUT).free > 150 * 2**20 + len(payload)
                        if local_store:
                            path.write_bytes(payload)
                        assert row["raw_archive_sha256"] == hashlib.sha256(payload).hexdigest()
                        row.update(
                            local_payload_stored=local_store,
                            path=str(path),
                            payload_sha256=hashlib.sha256(payload).hexdigest(),
                        )
                        captured = np.load(io.BytesIO(payload), allow_pickle=False)
                        current[row["branch"]][row["idx"]] = {
                            key: captured[key]
                            for key in ("features26", "state4", "scales4", "inherited_delta")
                        }
                        journal.write(json.dumps(row) + "\n")
                        journal.flush()
                        wave_rows.append(row)
                        print("CAPTURE", layer, row["branch"], row["role"], row["idx"], flush=True)
                    if wave == "smoke2":
                        gate = dict(
                            complete=len(wave_rows) == 2,
                            passed=len(wave_rows) == 2
                            and all(
                                row["complete"]
                                and not row["truth_argument_received"]
                                and row["stopped_before_current_injection"]
                                and row["prefix_head_layers"] == [1]
                                for row in wave_rows
                            ),
                            rows=wave_rows,
                            scope="onpolicycapture integrity only; no score selection",
                        )
                        (OUT / "smoke_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
                        assert gate["passed"]
            for branch, config in branches.items():
                target = {
                    idx: central[INDICES.index(idx), layer]
                    for idx in config["train"] + config["held"]
                }
                head = fit_head(current[branch], target, config["train"])
                headpath = OUT / f"{branch}_head_layer{layer}.json"
                assert not headpath.exists()
                headpath.write_text(
                    json.dumps(
                        dict(
                            head=head,
                            train_indices=config["train"],
                            layer=layer,
                            preceding_prefix_sha256=canonical_hash(config["prefix"]),
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                for idx in config["held"]:
                    evaluation = eval_current(
                        current[branch][idx], target[idx], head, baseline_data[idx]["state4"][layer]
                    )
                    evaluation.update(
                        branch=branch,
                        idx=idx,
                        layer=layer,
                        head_sha256=hashlib.sha256(headpath.read_bytes()).hexdigest(),
                        preceding_prefix_sha256=canonical_hash(config["prefix"]),
                    )
                    all_evaluations.append(evaluation)
                config["prefix"]["layer_models"][str(layer)] = head
                prefixpath = OUT / f"{branch}_prefix_through{layer}.json"
                prefixpath.write_text(json.dumps(config["prefix"], indent=2) + "\n")
            (OUT / f"evaluations_through_layer{layer}.json").write_text(
                json.dumps(all_evaluations, indent=2) + "\n"
            )
            if layer == 2:
                stored_bytes = sum(path.stat().st_size for path in (OUT / "layer2").rglob("*.npz"))
                import shutil

                disk = dict(
                    layer2_measured_local_bytes=stored_bytes,
                    projected_cloud_layers_bytes=stored_bytes * 13,
                    free_bytes=shutil.disk_usage(OUT).free,
                    reserve_bytes=150 * 2**20,
                    archive_volume="whest-onpolicy-r54-raw",
                    archive_all_raw=True,
                )
                (OUT / "storage_forecast.json").write_text(json.dumps(disk, indent=2) + "\n")
            print("FROZEN_LAYER", layer, flush=True)
            if layer == 3:
                print(
                    "PILOT3_INTEGRITY_DONE continuing fixedheads through14; no accuracythreshold selection",
                    flush=True,
                )
    final = branches["full24"]["prefix"]
    envelope = dict(
        on_policy=True,
        train_indices=INDICES,
        layer_models=final["layer_models"],
        feature_order=manifest["features"],
        n_features=26,
        ridge_lambda=0.01,
        floors=dict(mean=0, variance=1e-10),
        first_layer_zero=True,
        last_layer_zero=True,
        first_layer_skip=0,
        last_layer_skip=15,
        sigma_output_powers=[1, 2, 3, 4],
        corrected_channels=["mean", "variance", "kappa3", "kappa4"],
        prefix_hash_chain=[
            dict(
                layer=layer,
                prefix_sha256=hashlib.sha256(
                    (OUT / f"full24_prefix_through{layer}.json").read_bytes()
                ).hexdigest(),
                head_sha256=hashlib.sha256(
                    (OUT / f"full24_head_layer{layer}.json").read_bytes()
                ).hexdigest(),
            )
            for layer in range(1, 15)
        ],
        worker_source_sha256=manifest["source_sha256"],
        h18_sha256=SOURCE_SHA,
        dataset_revision=DATA_REV,
        dataset_shard_sha256=DATA_SHA,
        labels_sha256=manifest["labels_sha256"],
        helper_hashes=HELPER_HASHES,
        versions_expected={
            "numpy": "2.2.6",
            "flopscope": "0.12.1",
            "whestbench": "0.16.1",
            "torch": "2.5.1+cpu",
        },
        post_semantics="signedcentraldiag; fixedcentralcross; originalnonlinearfeed; ordinaryMdiag/Cdiag/G rebuilt",
        target="originalMCpost4−ownpost4 onbranchlearnedprior; normalizedsigma1..4",
        scope="privatecorpus-derivedcoefficients; nofreshclosedscore/RPC/resource/stage yet",
    )
    envelope_path = OUT / "frozen_full24_onpolicy_local_closure_26.json"
    envelope_path.write_text(json.dumps(envelope, indent=2) + "\n")
    metric = {
        key: np.asarray(
            [
                [
                    next(
                        row
                        for row in all_evaluations
                        if row["idx"] == idx and row["layer"] == layer
                    )["normalized_mse"][key]
                    for layer in range(1, 15)
                ]
                for idx in INDICES
            ]
        ).mean(axis=1)
        for key in ("zero_current", "head_current", "h18")
    }
    ratio = metric["head_current"].mean(axis=0) / metric["zero_current"].mean(axis=0)
    wins = (metric["head_current"] < metric["zero_current"]).sum(axis=0)
    rng = np.random.default_rng(261054)
    samples = rng.integers(0, 24, size=(20000, 24))
    boot = metric["head_current"][samples].mean(axis=1) / metric["zero_current"][samples].mean(
        axis=1
    )
    intervals = np.quantile(boot, [0.025, 0.975], axis=0).T
    gate = bool(((ratio[:2] < 0.90) & (wins[:2] >= 16) & (intervals[:2, 1] < 1)).any())
    physical = {
        key: np.asarray(
            [
                [
                    next(
                        row
                        for row in all_evaluations
                        if row["idx"] == idx and row["layer"] == layer
                    )["physical_mse"][key]
                    for layer in range(1, 15)
                ]
                for idx in INDICES
            ]
        ).mean(axis=1)
        for key in ("zero_current", "head_current", "h18")
    }
    bylayer = {
        str(layer): {
            key: np.asarray(
                [row["physical_mse"][key] for row in all_evaluations if row["layer"] == layer]
            )
            .mean(axis=0)
            .tolist()
            for key in ("zero_current", "head_current", "h18")
        }
        for layer in range(1, 15)
    }
    result = dict(
        complete=True,
        pooled_physical_mse={key: value.mean(axis=0).tolist() for key, value in physical.items()},
        physical_head_over_h18_ratio=(
            physical["head_current"].mean(axis=0) / physical["h18"].mean(axis=0)
        ).tolist(),
        physical_by_layer=bylayer,
        failures=0,
        frozen_model_path=str(envelope_path),
        frozen_model_sha256=hashlib.sha256(envelope_path.read_bytes()).hexdigest(),
        rows=all_evaluations,
        pooled_normalized_mse={key: value.mean(axis=0).tolist() for key, value in metric.items()},
        pooled_head_over_zero_ratio=ratio.tolist(),
        pooled_network_wins=wins.tolist(),
        bootstrap_seed=261054,
        bootstrap_replicates=20000,
        network_bootstrap_ratio95=intervals.tolist(),
        local_accuracy_diagnostic_signal=gate,
        pilot_integrity_completed=True,
        accuracy_gate_not_used_to_select_layers=True,
        scope=manifest["scope"],
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    (OUT / "completion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    raise SystemExit("Guarded Modal app; no local torch runtime")

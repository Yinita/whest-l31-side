"""Bounded offline corpus pilot: H18 rollout -> fixed four-state ridge -> rollout.

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
SOURCE_SHA = "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
DATA_REV = "cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8"
DATA_URL = f"https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k/resolve/{DATA_REV}/shard00.npz?download=true"
DATA_SHA = "aab6930a8a82b9dfe792a0e71456873165f73d143faac06fc32e5ca52b2b0fa5"
OUT = Path("results/phase2/modal_joint_closure_quadratic_rollout_r41")
INDICES = list(range(512, 640, 16))
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
app = modal.App("whest-joint-closure-quadratic-r41")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "flopscope==0.12.1", "whestbench==0.16.1")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_file(str(SOURCE), "/code/h18.py")
)


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(16384, 16384),
    timeout=300,
    retries=0,
    max_containers=8,
    scaledown_window=2,
)
def read_labels():
    import urllib.request

    import numpy as np

    data = urllib.request.urlopen(DATA_URL, timeout=60).read()
    assert hashlib.sha256(data).hexdigest() == DATA_SHA
    arrays = np.load(io.BytesIO(data), allow_pickle=False)
    ids = arrays["idx"]
    assert ids[32:40].tolist() == INDICES
    assert int(arrays["n_done"]) == 128 and int(arrays["n_samples"]) == 100000000
    assert int(arrays["w_seed0"]) == 770000 and int(arrays["s_seed0"]) == 880000
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        idx=ids[32:40],
        moments=arrays["moments"][32:40],
        final_mean=arrays["final_mean"][32:40],
        noise_rep_final=arrays["noise_rep_final"],
    )
    return {"data_sha256": DATA_SHA, "payload": buffer.getvalue()}


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(16384, 16384),
    timeout=300,
    retries=0,
    max_containers=8,
    scaledown_window=2,
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

    idx, model = job
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
        features, states, scales = [], [], []
        pre_seen, post_seen = set(), set()
        clipped_mean = clipped_var = 0
        theta = np.asarray(model["theta"]) if model else None
        feature_mean = np.asarray(model["feature_mean"]) if model else None
        feature_scale = np.asarray(model["feature_scale"]) if model else None

        def tracer(frame, event, arg):
            nonlocal clipped_mean, clipped_var
            if event != "line" or frame.f_code.co_filename != str(source):
                return tracer
            if frame.f_lineno not in (pretrigger, posttrigger):
                return tracer
            state = frame.f_locals
            layer = state.get("li", -1)
            if layer < 0 or layer >= 15:
                return tracer
            if frame.f_lineno == pretrigger:
                assert layer not in pre_seen, ("duplicate pretrigger", layer)
                pre_seen.add(layer)

                def array(name):
                    return np.asarray(state[name], dtype=np.float64).copy()

                sigma, alpha, phi, cdf = map(array, ("sigma", "alpha", "phi", "Phi"))
                d3 = array("D3") if state["mode"] == 1 else np.zeros(1024)
                d21norm = (
                    np.abs(np.asarray(state["D21"])).sum(axis=1, dtype=np.float64)
                    if state["mode"] == 1
                    else np.zeros(1024)
                )
                g4 = array("g4row") if state["mode"] == 1 else np.zeros(1024)
                skew, kurt = d3 / sigma**3, g4 / sigma**4
                hermites = [np.ones(1024), alpha]
                for degree in range(1, 7):
                    hermites.append(alpha * hermites[-1] - degree * hermites[-2])
                values = [
                    np.ones(1024),
                    array("mu"),
                    array("var"),
                    sigma,
                    alpha,
                    abs(alpha),
                    phi,
                    cdf,
                    array("pk1v"),
                    d3,
                    d21norm,
                    np.full(1024, float(state["K4_sigma"]))
                    if state["mode"] == 1
                    else np.zeros(1024),
                    sigma * phi,
                    skew * hermites[2] * phi,
                    kurt * hermites[3] * phi,
                    skew**2 * hermites[5] * phi,
                    skew * kurt * hermites[6] * phi,
                    kurt**2 * hermites[7] * phi,
                    skew,
                    kurt,
                    cdf,
                    np.full(1024, layer / 15),
                ]
                features.append(np.stack(values, axis=1))
                scales.append(np.stack([sigma**p for p in range(1, 5)], axis=1))
            elif frame.f_lineno == posttrigger:
                assert layer not in post_seen, ("duplicate posttrigger", layer)
                post_seen.add(layer)
                delta = np.asarray(state["delta"]) if state["delta"] is not None else 0.0
                mean = np.asarray(state["K1v"], dtype=np.float64) + delta
                post = np.stack(
                    [mean]
                    + [np.asarray(state[name], dtype=np.float64) for name in ("K2v", "K3v", "K4v")],
                    axis=1,
                )
                states.append(post.copy())
                inverse = (
                    np.asarray(state["inv_perm"])
                    if state["inv_perm"] is not None
                    else np.arange(1024)
                )
                if model:
                    x = (features[-1] - feature_mean) / feature_scale
                    pairs = model["quadratic_pairs"]
                    expanded = np.concatenate(
                        [x, np.stack([x[:, i] * x[:, j] for i, j in pairs], axis=1)], axis=1
                    )
                    mapped = (expanded - np.asarray(model["mapped_mean"])) / np.asarray(
                        model["mapped_scale"]
                    )
                    change = (mapped @ theta) * scales[-1]
                    # Exact Gaussian-ReLU layer0 has no closure correction.
                    if layer == 0:
                        change[:] = 0.0
                    updated = post + change
                    clipped_mean += int((updated[:, 0] < 0).sum())
                    clipped_var += int((updated[:, 1] < 1e-10).sum())
                    updated[:, 0] = np.maximum(updated[:, 0], 0)
                    updated[:, 1] = np.maximum(updated[:, 1], 1e-10)
                    # Offline NumPy result is copied into live cumulant carriers.
                    # It is never a resource/Phase2 admissibility claim.
                    fnp.copyto(
                        state["K1v"], fnp.asarray((updated[:, 0] - delta).astype(np.float32))
                    )
                    for channel, name in enumerate(("K2v", "K3v", "K4v"), 1):
                        fnp.copyto(state[name], fnp.asarray(updated[:, channel].astype(np.float32)))
                # Local labels are canonical original neuron IDs. Features used
                # by live roll-in above remain internal; saved training rows do not.
                features[-1] = features[-1][inverse]
                scales[-1] = scales[-1][inverse]
                states[-1] = states[-1][inverse]
            return tracer

        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            mlp = SimpleNamespace(width=1024, weights=[fnp.asarray(w) for w in weights])
            sys.settrace(tracer)
            try:
                prediction = np.asarray(estimator.predict(mlp, 2**41)).copy()
            finally:
                sys.settrace(None)
        assert len(features) == len(states) == len(scales) == 15
        assert prediction.shape == (16, 1024) and np.isfinite(prediction).all()
        first_sigma = np.sqrt((weights[0].astype(np.float64) ** 2).sum(axis=0))
        analytic_first_mean = first_sigma / np.sqrt(2 * np.pi)
        analytic_first_variance = first_sigma**2 * (0.5 - 1 / (2 * np.pi))
        permute = module.PRUNE
        rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        address_limit = list(resource.getrlimit(resource.RLIMIT_AS))
        assert all(np.isfinite(value).all() for value in features + states + scales)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            idx=np.asarray(idx),
            features=np.stack(features),
            states=np.stack(states),
            scales=np.stack(scales),
            prediction=prediction,
            analytic_first_mean=analytic_first_mean,
            analytic_first_variance=analytic_first_variance,
        )
        return {
            "complete": True,
            "idx": idx,
            "weight_seed": 770000 + idx,
            "weights_sha256": weights_sha,
            "h18_sha256": SOURCE_SHA,
            "elapsed_s": time.monotonic() - started,
            "cost": context.summary_dict(),
            "clipped_mean": clipped_mean,
            "clipped_var": clipped_var,
            "permutation_flag": permute,
            "maxrss_kib": rss_kib,
            "rlimit_as": address_limit,
            "captured_pre_layers": sorted(pre_seen),
            "captured_post_layers": sorted(post_seen),
            "versions": {
                k: importlib.metadata.version(k)
                for k in ("numpy", "flopscope", "whestbench", "torch")
            },
            "payload": buffer.getvalue(),
        }
    except Exception as error:
        return {
            "complete": False,
            "idx": idx,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_s": time.monotonic() - started,
        }


def score(prediction, truth):
    import numpy as np

    return float(np.mean((prediction.astype(np.float64) - truth.astype(np.float64)) ** 2))


@app.local_entrypoint()
def main():
    import numpy as np

    model_path = Path("results/phase2/joint_closure_quadratic_cv_r41/frozen_train24_quadratic.json")
    model_sha = "0610032b01bb3538a5ca6bea09079c92601222e7155775d1ea4ff0b292231e3b"
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_sha
    model = json.loads(model_path.read_text())
    assert model["train_indices"] == list(range(0, 384, 16))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    OUT.mkdir(exist_ok=False)
    manifest = dict(
        version="quadratic-joint-closure-r41-fresh8",
        pid=os.getpid(),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        h18_sha256=SOURCE_SHA,
        model_sha256=model_sha,
        dataset_revision=DATA_REV,
        dataset_shard_sha256=DATA_SHA,
        train_indices=model["train_indices"],
        validation_indices=INDICES,
        model_fixed_lambda=0.01,
        model_degree=2,
        model_selection="train24 networkCV; fresh8 labels unseen tofit",
        feature_order=FEATURES,
        constraint="layer0 analyticGaussian-ReLU correction exactlyzero; all4channelsjointelsewhere",
        cpu_per_worker=1,
        max_workers=8,
        memory_mib=16384,
        timeout_s=300,
        retries=0,
        thread_count=1,
        state_semantics="formal signed central diagonal closure; ordinaryMdiag/Cdiag/G rebuilt, mixedcentral frozen, original nonlinearfeed unchanged",
        scope="Offline diagnostic; unmetered NumPy correction; 16GiBdevelopment; no candidate, adjustedscore, resource or stage acceptance",
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    labels_result = read_labels.remote()
    payload = labels_result["payload"]
    (OUT / "research_labels.npz").write_bytes(payload)
    labels = np.load(io.BytesIO(payload), allow_pickle=False)
    baseline, candidate = {}, []
    with (OUT / "journal.jsonl").open("x") as journal:
        for phase, jobs in [
            ("baseline", [(idx, None) for idx in INDICES]),
            ("quadratic_joint", [(idx, model) for idx in INDICES]),
        ]:
            if phase == "quadratic_joint":
                gate = []
                for i, idx in enumerate(INDICES):
                    if idx not in baseline:
                        break
                    trace = np.load(baseline[idx]["path"], allow_pickle=False)
                    mse = score(trace["analytic_first_mean"], labels["moments"][i, 0, 0])
                    nominal = float(trace["analytic_first_variance"].mean() / 100000000)
                    gate.append(dict(idx=idx, first_mean_mc_noise_ratio=mse / nominal))
                passed = (
                    len(gate) == 8
                    and 0.4 < np.mean([r["first_mean_mc_noise_ratio"] for r in gate]) < 2.5
                )
                (OUT / "weight_rng_gate.json").write_text(
                    json.dumps(dict(passed=bool(passed), rows=gate), indent=2) + "\n"
                )
                if not passed:
                    (OUT / "completion.json").write_text(
                        json.dumps(
                            dict(
                                complete=False,
                                reason="baseline/analyticMCgatefailed",
                                successful=len(baseline),
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
                    return
            for row in rollout.map(jobs, order_outputs=False, return_exceptions=True):
                if isinstance(row, Exception):
                    row = dict(complete=False, error_type=type(row).__name__, error=str(row))
                if row.get("complete"):
                    payload = row.pop("payload")
                    path = OUT / f"{phase}_idx{row['idx']:04d}.npz"
                    path.write_bytes(payload)
                    row.update(path=str(path), payload_sha256=hashlib.sha256(payload).hexdigest())
                    if phase == "baseline":
                        baseline[row["idx"]] = row
                    else:
                        i = INDICES.index(row["idx"])
                        before = np.load(baseline[row["idx"]]["path"], allow_pickle=False)
                        after = np.load(io.BytesIO(payload), allow_pickle=False)
                        before_raw = score(before["prediction"][-1], labels["final_mean"][i])
                        after_raw = score(after["prediction"][-1], labels["final_mean"][i])
                        row.update(
                            before_raw=before_raw,
                            after_raw=after_raw,
                            after_over_before=after_raw / before_raw,
                            paired_weight_hash_equal=row["weights_sha256"]
                            == baseline[row["idx"]]["weights_sha256"],
                            first_layer_prediction_exact_equal=bool(
                                np.array_equal(before["prediction"][0], after["prediction"][0])
                            ),
                        )
                if phase == "quadratic_joint":
                    candidate.append(row)
                journal.write(json.dumps(dict(phase=phase, **row)) + "\n")
                journal.flush()
                print(
                    phase,
                    row.get("idx"),
                    row.get("complete"),
                    row.get("after_over_before"),
                    flush=True,
                )
    success = [r for r in candidate if r.get("complete")]
    result = dict(
        complete=True,
        rows=candidate,
        failures=len(candidate) - len(success),
        model_sha256=model_sha,
        dataset_revision=DATA_REV,
        mean_before_raw=float(np.mean([r["before_raw"] for r in success])) if success else None,
        mean_after_raw=float(np.mean([r["after_raw"] for r in success])) if success else None,
        scope=manifest["scope"],
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    (OUT / "completion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    raise SystemExit("Launch with modal run; no local torch installation/download.")

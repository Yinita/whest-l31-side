"""Target-free H18 final26 capture, smoke2 then remaining22 TRAIN networks.

Offline tracer only; no labels/reference/model supplied to cloud prediction.
Actual final PK1 is converted to ordinary marginal cumulants in float64.
No state injection, no natural rollout modification, no candidate score claim.
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
OUT = Path("results/phase2/final_readout_capture_r52")
BASE = Path("results/phase2/modal_joint_closure_pilot_r38_r02")
IDS = list(range(0, 384, 16))
FEATURES = [
    "ones", "mu", "var", "sigma", "alpha", "abs_alpha", "phi", "Phi", "pk1",
    "D3", "D21norm", "K4_sigma", "sigma_phi", "skew_H2phi", "kurt_H3phi",
    "skew2_H5phi", "skewkurt_H6phi", "kurt2_H7phi", "skew", "kurt", "gate",
    "depth_fraction", "own_post_mean_over_sigma", "own_post_variance_over_sigma2",
    "own_post_kappa3_over_sigma3", "own_post_kappa4_over_sigma4",
]
app = modal.App("whest-final-readout-capture-r52")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "flopscope==0.12.1", "whestbench==0.16.1")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_file(str(SOURCE), "/code/h18.py")
)


@app.function(image=image, cpu=(1, 1), memory=(16384, 16384), timeout=300,
              retries=0, max_containers=22, scaledown_window=2)
def capture(job):
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

    idx, expected_weight_hash = job
    started = time.monotonic()
    try:
        for key in tuple(os.environ):
            if re.match(r"V\d+_", key):
                os.environ.pop(key)
        source = Path("/code/h18.py")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
        spec = importlib.util.spec_from_file_location("h18_final_r52", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        flops.configure(symmetry_warnings=False)
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu").manual_seed(770000 + idx)
        weights = (torch.randn(16, 1024, 1024, generator=generator, dtype=torch.float32)
                   * (2.0 / 1024) ** 0.5).numpy()
        weight_hash = hashlib.sha256(weights.tobytes()).hexdigest()
        assert weight_hash == expected_weight_hash
        estimator = module.Estimator()
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            estimator.setup(SetupContext(width=1024, depth=16, flop_budget=2**41,
                                         api_version="0.1", seed=0))
        lines = source.read_text().splitlines()
        trigger = next(i for i, line in enumerate(lines, 1)
                       if "row_ = pk1v if delta is None else pk1v + delta" in line)
        captured = {}

        def tracer(frame, event, arg):
            if event != "line" or frame.f_code.co_filename != str(source):
                return tracer
            if frame.f_lineno != trigger:
                return tracer
            state = frame.f_locals
            assert state["li"] == 15 and state["last"] and state["trim"]
            assert not captured

            def vector(name):
                return np.asarray(state[name], dtype=np.float64).copy()

            sigma, alpha, phi, cdf = map(vector, ("sigma", "alpha", "phi", "Phi"))
            d3 = vector("D3") if state["mode"] == 1 else np.zeros(1024)
            d21 = state["D21"]
            assert d21 is None and state["C_off"] is None
            d21norm = np.zeros(1024)  # Actual final trimmed H18 has no pair readout.
            g4 = vector("g4row") if state["mode"] == 1 else np.zeros(1024)
            skew, kurt = d3 / sigma**3, g4 / sigma**4
            hermites = [np.ones(1024), alpha]
            for degree in range(1, 7):
                hermites.append(alpha * hermites[-1] - degree * hermites[-2])
            delta = state["delta"]
            actual_mean = np.asarray(state["pk1v"] if delta is None
                                     else state["pk1v"] + delta, dtype=np.float64).copy()
            raw = np.asarray(state["PK1"], dtype=np.float64).copy()
            p1, p2, p3, p4 = raw
            variance = p2 - p1**2
            k3 = p3 - 3 * p1 * p2 + 2 * p1**3
            k4 = p4 - 4 * p1 * p3 - 3 * p2**2 + 12 * p1**2 * p2 - 6 * p1**4
            ownpost = np.stack([actual_mean, variance, k3, k4], axis=1)
            values = [np.ones(1024), vector("mu"), vector("var"), sigma, alpha,
                      abs(alpha), phi, cdf, vector("pk1v"), d3, d21norm,
                      np.full(1024, float(state["K4_sigma"])) if state["mode"] == 1
                      else np.zeros(1024), sigma * phi, skew * hermites[2] * phi,
                      kurt * hermites[3] * phi, skew**2 * hermites[5] * phi,
                      skew * kurt * hermites[6] * phi, kurt**2 * hermites[7] * phi,
                      skew, kurt, cdf, np.ones(1024)]
            features = np.concatenate([np.stack(values, axis=1),
                                       ownpost / np.stack([sigma**p for p in range(1, 5)], axis=1)],
                                      axis=1)
            inverse = np.asarray(state["inv_perm"]) if state["inv_perm"] is not None else np.arange(1024)
            captured.update(features=features[inverse], sigma=sigma[inverse],
                            raw_power=raw[:, inverse], ownpost=ownpost[inverse],
                            expected_final=actual_mean[inverse])
            return tracer

        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            mlp = SimpleNamespace(width=1024, weights=[fnp.asarray(w) for w in weights])
            sys.settrace(tracer)
            try:
                prediction = np.asarray(estimator.predict(mlp, 2**41)).copy()
            finally:
                sys.settrace(None)
        assert captured["features"].shape == (1024, 26)
        assert np.array_equal(prediction[-1], captured["expected_final"])
        assert all(np.isfinite(x).all() for x in captured.values())
        first_sigma = np.sqrt((weights[0].astype(np.float64) ** 2).sum(axis=0))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, idx=idx, prediction=prediction,
                            analytic_first_mean=first_sigma / np.sqrt(2 * np.pi),
                            analytic_first_variance=first_sigma**2 * (0.5 - 1 / (2 * np.pi)),
                            **captured)
        return dict(complete=True, idx=idx, weight_seed=770000 + idx,
                    weights_sha256=weight_hash, h18_sha256=SOURCE_SHA,
                    elapsed_s=time.monotonic() - started, cost=context.summary_dict(),
                    maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)),
                    versions={k: importlib.metadata.version(k)
                              for k in ("numpy", "flopscope", "whestbench", "torch")},
                    captured_layer=15, pair_readout_unavailable=True,
                    labels_reference_model_absent=True, payload=buffer.getvalue())
    except Exception as error:
        return dict(complete=False, idx=idx, error_type=type(error).__name__,
                    error=str(error), elapsed_s=time.monotonic() - started)


@app.local_entrypoint()
def main():
    import numpy as np

    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    original = json.loads((BASE / "manifest.json").read_text())
    baseline = {}
    for line in (BASE / "journal.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["phase"] in ("baseline", "smoke") and row.get("complete") and row["idx"] in IDS:
            baseline[row["idx"]] = row
    assert sorted(baseline) == IDS
    labels_path = BASE / "research_labels.npz"
    labels = np.load(labels_path, allow_pickle=False)
    assert labels["idx"][:24].tolist() == IDS
    OUT.mkdir(exist_ok=False)
    manifest = dict(version="final-readout26-targetfree-train24-capture-r52", pid=os.getpid(),
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    h18_sha256=SOURCE_SHA, dataset_revision=original["dataset_revision"],
                    train_indices=IDS, feature_order=FEATURES, labels_path=str(labels_path),
                    labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                    weight_hashes={str(i): baseline[i]["weights_sha256"] for i in IDS},
                    single_change="capture final15 marginal26 on original H18 only; no injections",
                    ordinary_post_semantics="float64 rawPK1-to-central formula, actualparentfloat32meanplusdelta",
                    final_pair_semantics="trimmed D21/C_off/wk4m unavailable, D21normfixedzero; no32features",
                    gate="after2smokefirstanalyticnoise .4..2.5 and maxpredictiondiff<=1e-6 then22",
                    preregistered_cv_gate="linear26 ridge.01,4blocked6networkfolds,ratio<.90,wins>=16,bootstrap95upper<1",
                    cpu_per_worker=1, max_workers=22, memory_mib=16384, timeout_s=300,
                    retries=0, thread_count=1,
                    scope="offline capture only, nativeNumPy tracer, no adjusted/resource/candidate/tenfold acceptance")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with (OUT / "journal.jsonl").open("x") as journal:
        for phase, indices in (("smoke", IDS[:2]), ("capture", IDS[2:])):
            if phase == "capture":
                passed = len(rows) == 2 and all(r.get("complete") and
                    .4 < r["analytic_first_noise_ratio"] < 2.5 and
                    r["prediction_maxabs_vs_original_capture"] <= 1e-6 for r in rows)
                (OUT / "smoke_gate.json").write_text(json.dumps(dict(passed=passed, rows=rows), indent=2) + "\n")
                if not passed:
                    break
            jobs = [(i, baseline[i]["weights_sha256"]) for i in indices]
            for row in capture.map(jobs, order_outputs=False, return_exceptions=True):
                if isinstance(row, Exception):
                    row = dict(complete=False, error_type=type(row).__name__, error=str(row))
                if row.get("complete"):
                    payload = row.pop("payload")
                    path = OUT / f"capture_idx{row['idx']:04d}.npz"
                    path.write_bytes(payload)
                    row.update(path=str(path), payload_sha256=hashlib.sha256(payload).hexdigest())
                    trace = np.load(io.BytesIO(payload), allow_pickle=False)
                    i = IDS.index(row["idx"])
                    before = np.load(baseline[row["idx"]]["path"], allow_pickle=False)
                    row.update(analytic_first_noise_ratio=float(np.mean((trace["analytic_first_mean"] -
                        labels["moments"][i, 0, 0])**2) / (trace["analytic_first_variance"].mean()/1e8)),
                        prediction_maxabs_vs_original_capture=float(np.max(abs(trace["prediction"] - before["prediction"]))),
                        raw_mse_vs_final_mean=float(np.mean((trace["prediction"][-1].astype(np.float64)-labels["final_mean"][i])**2)))
                row["phase"] = phase
                rows.append(row)
                journal.write(json.dumps(row) + "\n")
                journal.flush()
                print(phase, row.get("idx"), row.get("complete"), row.get("error_type"), flush=True)
    successful = [r for r in rows if r.get("complete")]
    report = dict(complete=len(successful) == 24, successful=len(successful),
                  failures=sum(not r.get("complete") for r in rows), rows=rows,
                  scope=manifest["scope"],
                  finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (OUT / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)

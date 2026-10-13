"""One preregistered final-only26 ridge gate on independent corpus TRAIN24.

Four blocked network folds, lambda .01, no parameter/features/layer sweep.
Target=(final_mean-H18final)/sigma; prediction parent+sigma*ridge then float32.
Offline labels/models remain private; this is OOF diagnostic, not stage score.
"""

# ruff: noqa: E402
import os

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
import hashlib
import json
import time
from pathlib import Path

import numpy as np

CAPTURE = Path("results/phase2/final_readout_capture_r52_r02")
OUT = Path("results/phase2/final_readout_cv_r52")
IDS = list(range(0, 384, 16))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(data, indices):
    x = np.concatenate([data[i]["x"] for i in indices])
    y = np.concatenate([data[i]["target"] for i in indices])
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)
    mean[0], scale[0] = 0, 1
    z = (x - mean) / scale
    penalty = np.eye(26) * 0.01 * len(x)
    penalty[0, 0] = 0
    theta = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return dict(feature_mean=mean, feature_scale=scale, theta=theta)


def predict(datum, model):
    z = (datum["x"] - model["feature_mean"]) / model["feature_scale"]
    return (datum["parent"] + datum["sigma"] * (z @ model["theta"])).astype(np.float32)


def main():
    started = time.monotonic()
    assert not OUT.exists()
    manifest = json.loads((CAPTURE / "manifest.json").read_text())
    completion = json.loads((CAPTURE / "completion.json").read_text())
    assert completion["complete"] and completion["successful"] == 24 and completion["failures"] == 0
    assert completion["capture_numerical_gate_passed"]
    assert manifest["train_indices"] == IDS and len(manifest["feature_order"]) == 26
    labels_path = Path(manifest["labels_path"])
    assert sha(labels_path) == manifest["labels_sha256"]
    labels = np.load(labels_path, allow_pickle=False)
    assert labels["idx"][:24].tolist() == IDS
    traces = {r["idx"]: r for r in completion["rows"]}
    assert sorted(traces) == IDS
    data, provenance = [], []
    for position, idx in enumerate(IDS):
        row = traces[idx]
        path = Path(row["path"])
        assert sha(path) == row["payload_sha256"]
        assert row["weights_sha256"] == manifest["weight_hashes"][str(idx)]
        assert row["h18_sha256"] == manifest["h18_sha256"]
        assert row["labels_reference_model_absent"] and row["pair_readout_unavailable"]
        assert row["prediction_maxabs_vs_original_capture"] <= 1e-5
        trace = np.load(path, allow_pickle=False)
        x = trace["features"].astype(np.float64)
        parent = trace["prediction"][-1].astype(np.float64)
        sigma = trace["sigma"].astype(np.float64)
        truth = labels["final_mean"][position].astype(np.float64)
        assert x.shape == (1024, 26) and parent.shape == sigma.shape == truth.shape == (1024,)
        assert np.array_equal(parent, trace["expected_final"])
        assert np.all(sigma > 0) and np.isfinite(x).all()
        data.append(dict(x=x, parent=parent, sigma=sigma, truth=truth,
                         target=(truth-parent)/sigma))
        provenance.append(dict(idx=idx, trace_path=str(path), trace_sha256=sha(path),
                               weights_sha256=row["weights_sha256"]))
    OUT.mkdir()
    cv_manifest = dict(version="final-only26-fixed-ridge-networkCV-r52", pid=os.getpid(),
        source_sha256=sha(Path(__file__)), capture_manifest_sha256=sha(CAPTURE / "manifest.json"),
        capture_completion_sha256=sha(CAPTURE / "completion.json"),
        parent_sha256=manifest["h18_sha256"], dataset_revision=manifest["dataset_revision"],
        labels_sha256=sha(labels_path), train_indices=IDS, feature_order=manifest["feature_order"],
        ridge_lambda=0.01, folds=[IDS[f*6:(f+1)*6] for f in range(4)],
        target="(originalfloat64final_mean-H18float32final)/originalpresigma",
        inference="(parentfloat32asfloat64+sigma*normalized_delta).astype(float32), no floor/clip",
        fixed_gate="aggregate rawratio<.90,>=16/24networkwins,pairednetworkbootstrap95upper<1",
        bootstrap_seed=520013, bootstrap_resamples=10000, provenance=provenance,
        scope="TRAIN-only OOF finalreadout diagnostic; no heldout/strictRPC/adjusted/tenfold acceptance")
    (OUT / "manifest.json").write_text(json.dumps(cv_manifest, indent=2) + "\n")
    rows, foldmodels = [], {}
    for fold in range(4):
        held = list(range(fold*6, (fold+1)*6))
        train = [i for i in range(24) if i not in held]
        model = fit(data, train)
        foldmodels[str(fold)] = {k: v.tolist() for k, v in model.items()}
        for i in held:
            candidate = predict(data[i], model)
            assert np.isfinite(candidate).all()
            before = float(np.mean((data[i]["parent"]-data[i]["truth"])**2))
            after = float(np.mean((candidate.astype(np.float64)-data[i]["truth"])**2))
            rows.append(dict(idx=IDS[i], fold=fold, before_raw=before, after_raw=after,
                             after_over_before=after/before, negative_outputs=int((candidate < 0).sum()),
                             delta_rms=float(np.sqrt(np.mean((candidate-data[i]["parent"])**2)))))
        print(json.dumps(dict(fold=fold, elapsed_s=time.monotonic()-started)), flush=True)
    before = np.asarray([r["before_raw"] for r in rows])
    after = np.asarray([r["after_raw"] for r in rows])
    ratio = float(after.mean()/before.mean())
    wins = int((after < before).sum())
    rng = np.random.default_rng(520013)
    selection = rng.integers(0, 24, size=(10000, 24))
    bootstrap = after[selection].mean(axis=1)/before[selection].mean(axis=1)
    bounds = np.quantile(bootstrap, [.025, .975]).tolist()
    passed = ratio < .90 and wins >= 16 and bounds[1] < 1
    model = fit(data, list(range(24)))
    model_record = dict(version="r52-final-only26-frozen-train24", n_features=26,
        feature_order=manifest["feature_order"], train_indices=IDS, ridge_lambda=.01,
        target=cv_manifest["target"], inference=cv_manifest["inference"],
        corrected_layers=[15], midlayer_injection=False,
        **{k: v.tolist() for k, v in model.items()})
    model_path = OUT / "frozen_train24_final26.json"
    model_path.write_text(json.dumps(model_record, indent=2) + "\n")
    (OUT / "private_fold_models.json").write_text(json.dumps(foldmodels, indent=2) + "\n")
    report = dict(complete=True, failures=0, rows=rows, mean_before_raw=float(before.mean()),
        mean_after_raw=float(after.mean()), after_over_before=ratio, wins=wins,
        paired_network_bootstrap95=bounds, gate_passed=bool(passed),
        negative_outputs=sum(r["negative_outputs"] for r in rows), model_sha256=sha(model_path),
        elapsed_s=time.monotonic()-started, scope=cv_manifest["scope"])
    (OUT / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()

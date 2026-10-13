"""Train24 network CV: local mean innovation vs total accumulated mean error.

One fixed per-layer 22-feature linear model, lambda0.01 and sigma scaling.
Layer0 has exact Gaussian-ReLU zero correction and is excluded from fit.
The first-order target decomposition is not a nonlinear closed-rollout score.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
import time
from pathlib import Path

import numpy as np

BASE = Path("results/phase2/modal_joint_closure_pilot_r38_r02")
TARGET = Path("results/phase2/modal_mean_innovation_targets_r43")
OUT = Path("results/phase2/causal_mean_innovation_cv_r44")
IDS = list(range(0, 384, 16))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(data, train, layer, label):
    x = np.concatenate([data[i]["features"][layer] for i in train])
    y = np.concatenate([data[i][label][layer] / data[i]["sigma"][layer] for i in train])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    mean[0], scale[0] = 0, 1
    scale = np.maximum(scale, 1e-12)
    x = (x - mean) / scale
    penalty = np.eye(22) * 0.01 * len(x)
    penalty[0, 0] = 0
    theta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return dict(feature_mean=mean, feature_scale=scale, theta=theta)


def infer(data, model, layer):
    return (
        ((data["features"][layer] - model["feature_mean"]) / model["feature_scale"])
        @ model["theta"]
        * data["sigma"][layer]
    )


def serialize(model):
    return {k: v.tolist() for k, v in model.items()}


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    manifest = json.loads((BASE / "manifest.json").read_text())
    target_manifest = json.loads((TARGET / "manifest.json").read_text())
    target_report = json.loads((TARGET / "completion.json").read_text())
    assert target_report["complete"] and target_report["failures"] == 0
    assert manifest["train_indices"] == IDS
    assert target_manifest["train_indices"] == IDS
    target_rows = {r["idx"]: r for r in target_report["rows"]}
    baseline_rows = {}
    for text in (BASE / "journal.jsonl").read_text().splitlines():
        row = json.loads(text)
        if row["phase"] in ("baseline", "smoke"):
            baseline_rows[row["idx"]] = row
    data, provenance = [], []
    for idx in IDS:
        row = target_rows[idx]
        assert row["weights_sha256"] == baseline_rows[idx]["weights_sha256"]
        target_path = Path(row["path"])
        assert digest(target_path) == row["payload_sha256"]
        target = np.load(target_path, allow_pickle=False)
        trace_path = Path(baseline_rows[idx]["path"])
        assert digest(trace_path) == baseline_rows[idx]["payload_sha256"]
        trace = np.load(trace_path, allow_pickle=False)
        total, innovation, propagated = [
            target[k].astype(np.float64) for k in ("total_error", "innovation", "propagated")
        ]
        assert total.shape == innovation.shape == propagated.shape == (15, 1024)
        assert np.max(abs(total - innovation - propagated)) < 1e-15
        data.append(
            dict(
                features=trace["features"],
                sigma=trace["scales"][..., 0],
                total_error=total,
                innovation=innovation,
                propagated=propagated,
            )
        )
        provenance.append(
            dict(
                idx=idx,
                weights_sha256=row["weights_sha256"],
                target_payload_sha256=row["payload_sha256"],
                trace_payload_sha256=digest(trace_path),
            )
        )
    predeclared = dict(
        version="causal-mean-innovation-cv-r44",
        pid=os.getpid(),
        source_sha256=digest(Path(__file__)),
        parent_sha256=manifest["h18_sha256"],
        dataset_revision=manifest["dataset_revision"],
        train_indices=IDS,
        validation_labels_used=False,
        source_provenance=provenance,
        feature_order=manifest["feature_order"],
        ridge_lambda=0.01,
        normalization="rawfeatures foldtrain-only; mean residual/sigma",
        folds=[[IDS[i] for i in range(f * 6, (f + 1) * 6)] for f in range(4)],
        mechanism="innovation_l = total_error_l - Phi_l W_l.T total_error_(l-1)",
        first_layer_zero=True,
        fitted_layers=list(range(1, 15)),
        final_layer_local_correction=0,
        fixed_gate="heldtrain24 pooled normalized innovation MSE <.90 baseline AND >=16/24 networkwins; no lambda scan",
        limits=[
            "meanJacobian linearization excludes covariance/source feedback",
            "networkCVtrain24 is development",
            "no fullclosedrollout or finaloutputscore from this fit",
        ],
    )
    (OUT / "manifest.json").write_text(json.dumps(predeclared, indent=2) + "\n")
    hats = [{"local_hat": np.zeros((15, 1024)), "total_hat": np.zeros((15, 1024))} for i in IDS]
    for fold in range(4):
        held = list(range(fold * 6, (fold + 1) * 6))
        train = [i for i in range(24) if i not in held]
        for layer in range(1, 15):
            local_model = fit(data, train, layer, "innovation")
            total_model = fit(data, train, layer, "total_error")
            for i in held:
                hats[i]["local_hat"][layer] = infer(data[i], local_model, layer)
                hats[i]["total_hat"][layer] = infer(data[i], total_model, layer)
        print(json.dumps(dict(fold=fold, elapsed_s=time.monotonic() - started)), flush=True)
    rows = []
    for i, idx in enumerate(IDS):
        local, total = data[i]["innovation"], data[i]["total_error"]
        sigma = data[i]["sigma"]
        local_after, total_after = local - hats[i]["local_hat"], total - hats[i]["total_hat"]
        row = dict(
            idx=idx,
            fold=i // 6,
            local_normalized_before=float(((local[1:] / sigma[1:]) ** 2).mean()),
            local_normalized_after=float(((local_after[1:] / sigma[1:]) ** 2).mean()),
            local_raw_before=float((local[1:] ** 2).mean()),
            local_raw_after=float((local_after[1:] ** 2).mean()),
            total_normalized_before=float(((total[1:] / sigma[1:]) ** 2).mean()),
            total_normalized_after=float(((total_after[1:] / sigma[1:]) ** 2).mean()),
            layer_mse=[
                dict(
                    layer=l,
                    local_before=float((local[l] ** 2).mean()),
                    local_after=float((local_after[l] ** 2).mean()),
                    total_before=float((total[l] ** 2).mean()),
                    total_after=float((total_after[l] ** 2).mean()),
                    propagated_mse=float((data[i]["propagated"][l] ** 2).mean()),
                    current_error_propagation_cross=float(
                        (total[l] * data[i]["propagated"][l]).mean()
                    ),
                )
                for l in range(15)
            ],
        )
        rows.append(row)
        path = OUT / f"cv_predictions_idx{idx:04d}.npz"
        np.savez_compressed(
            path,
            idx=idx,
            **hats[i],
            phi=data[i]["features"][..., 7],
            total_error=total,
            innovation=local,
        )
        row["cv_payload_path"], row["cv_payload_sha256"] = str(path), digest(path)
    ratio = np.mean([r["local_normalized_after"] for r in rows]) / np.mean(
        [r["local_normalized_before"] for r in rows]
    )
    wins = sum(r["local_normalized_after"] < r["local_normalized_before"] for r in rows)
    result = dict(
        complete=True,
        rows=rows,
        local_innovation_normalized_ratio=float(ratio),
        local_innovation_raw_ratio=float(
            np.mean([r["local_raw_after"] for r in rows])
            / np.mean([r["local_raw_before"] for r in rows])
        ),
        total_error_normalized_readout_ratio=float(
            np.mean([r["total_normalized_after"] for r in rows])
            / np.mean([r["total_normalized_before"] for r in rows])
        ),
        local_networkwins=wins,
        signal_gate_passed=bool(ratio < 0.90 and wins >= 16),
        elapsed_s=time.monotonic() - started,
    )
    # Freeze exactly the declared mechanism, even a negative, for audit/replay.
    models = {
        str(layer): serialize(fit(data, list(range(24)), layer, "innovation"))
        for layer in range(1, 15)
    }
    frozen = dict(
        feature_order=manifest["feature_order"],
        layer_models=models,
        sigma_output_power=1,
        train_indices=IDS,
        first_layer_zero=True,
        last_layer_local_correction=0,
        ridge_lambda=0.01,
        target="local_mean_innovation",
        corrected_channels=["mean"],
        no_variance_floor=True,
    )
    path = OUT / "frozen_train24_mean_innovation.json"
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    result["frozen_model_sha256"] = digest(path)
    (OUT / "completion.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()

"""Network-CV fixed graph-context6 addition to local teacher-forced closure.

Fixed per-layer linear ridge0.01. Two prespecified feature mechanisms26/32,
where the latter adds six target-free preactivation graph contractions. No
held-out competition or validation labels enter fitting/model selection.
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

BASE = Path("results/phase2/modal_joint_closure_pilot_r38_r02")
FIRST = Path("results/phase2/modal_joint_oracle_reset_r46")
REMAINING = Path("results/phase2/modal_joint_oracle_reset_r47")
OUT = Path("results/phase2/local_joint_graph6_cv_r53")
GRAPH = Path("results/phase2/modal_local_joint_pre6_capture_r51")
GRAPH_ORDER = ["rho2_rowmean", "rho3_rowmean", "rho_triangle_diagmean", "rho_D21_left_rowmean", "rho_D21_right_rowmean", "rho2_K22_rowmean"]
IDS = list(range(0, 384, 16))
EXTRA = [
    "own_post_mean_over_sigma",
    "own_post_variance_over_sigma2",
    "own_post_kappa3_over_sigma3",
    "own_post_kappa4_over_sigma4",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(data, indices, layer, size):
    x = np.concatenate([data[i]["x"][layer, :, :size] for i in indices])
    y = np.concatenate([data[i]["y"][layer] for i in indices])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    mean[0], scale[0] = 0, 1
    scale = np.maximum(scale, 1e-12)
    x = (x - mean) / scale
    penalty = np.eye(size) * 0.01 * len(x)
    penalty[0, 0] = 0
    theta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return dict(feature_mean=mean, feature_scale=scale, theta=theta)


def infer(datum, model, layer, size):
    return ((datum["x"][layer, :, :size] - model["feature_mean"]) / model["feature_scale"]) @ model[
        "theta"
    ]


def serialize(model):
    return {k: v.tolist() for k, v in model.items()}


def main():
    started = time.monotonic()
    assert not OUT.exists()
    original = json.loads((BASE / "manifest.json").read_text())
    first_report = json.loads((FIRST / "completion.json").read_text())
    rest_report = json.loads((REMAINING / "completion.json").read_text())
    assert first_report["complete"] and rest_report["complete"]
    assert first_report["failures"] == rest_report["failures"] == 0
    graph_manifest=json.loads((GRAPH/"manifest.json").read_text())
    graph_report=json.loads((GRAPH/"completion.json").read_text())
    assert graph_report["complete"] and graph_report["failures"] ==0
    assert graph_manifest["graph_feature_order"] == GRAPH_ORDER
    graph_rows={row["idx"]:row for row in graph_report["rows"]}
    assert sorted(graph_rows)==IDS
    rows = {row["idx"]: row for report in (first_report, rest_report) for row in report["rows"]}
    assert sorted(rows) == IDS
    labels_path = BASE / "research_labels.npz"
    labels = np.load(labels_path, allow_pickle=False)
    assert labels["idx"][:24].tolist() == IDS
    raw = np.moveaxis(labels["moments"][:24, :, :15].astype(np.float64), 1, -1)
    m1, m2, m3, m4 = [raw[..., c] for c in range(4)]
    var = m2 - m1**2
    target = np.stack(
        [
            m1,
            var,
            m3 - 3 * m1 * m2 + 2 * m1**3,
            m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4 - 3 * var**2,
        ],
        axis=-1,
    )
    baseline = {}
    for line in (BASE / "journal.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["phase"] in ("baseline", "smoke") and row.get("complete"):
            baseline[row["idx"]] = row
    data, provenance = [], []
    for position, idx in enumerate(IDS):
        row = rows[idx]
        assert row["complete"] and row["weights_sha256"] == baseline[idx]["weights_sha256"]
        assert row["h18_sha256"] == original["h18_sha256"]
        path = Path(row["path"])
        assert digest(path) == row["payload_sha256"]
        trace = np.load(path, allow_pickle=False)
        x, state, scales = [trace[k].astype(np.float64) for k in ("features", "states", "scales")]
        assert x.shape == (15, 1024, 22) and state.shape == scales.shape == (15, 1024, 4)
        assert np.array_equal(trace["after_states"][1:], target[position, 1:])
        if "actual_after_states" in trace.files:
            actual = trace["actual_after_states"]
            assert np.max(abs(actual[1:] - target[position, 1:])) < 5e-5
        grow=graph_rows[idx]
        assert grow["complete"] and grow["weights_sha256"]==row["weights_sha256"]
        assert grow["prior_trace_sha256"]==digest(path)
        graph_path=Path(grow["path"])
        assert digest(graph_path)==grow["payload_sha256"]
        graph_trace=np.load(graph_path,allow_pickle=False)
        assert graph_trace.files==["idx","graph_features"]
        assert int(graph_trace["idx"])==idx
        graph=graph_trace["graph_features"].astype(np.float64)
        assert graph.shape==(15,1024,6) and np.isfinite(graph).all()
        assert np.array_equal(graph[0],np.zeros((1024,6)))
        assert grow["prior_features26_maxabs_delta"]<1e-3
        assert grow["prior_state_maxabs_delta"]<1e-5
        assert grow["prior_prediction_maxabs_delta"]<1e-5
        mapped = np.concatenate([x, state / scales,graph], axis=-1)
        y = (target[position] - state) / scales
        assert np.isfinite(mapped).all() and np.isfinite(y).all()
        data.append(dict(x=mapped, y=y, scales=scales))
        provenance.append(
            dict(
                idx=idx,
                teacher_forced_trace_sha256=digest(path),
                weight_sha256=row["weights_sha256"],
                source_path=str(path),
                graph_path=str(graph_path), graph_sha256=digest(graph_path),
                prior_features26_maxabs_delta=grow["prior_features26_maxabs_delta"],
                prior_state_maxabs_delta=grow["prior_state_maxabs_delta"],
                prior_prediction_maxabs_delta=grow["prior_prediction_maxabs_delta"],
            )
        )
    OUT.mkdir()
    manifest = dict(
        version="teacher-forced-local-joint-graph6-cv-r53",
        pid=os.getpid(),
        source_sha256=digest(Path(__file__)),
        parent_sha256=original["h18_sha256"],
        dataset_revision=original["dataset_revision"],
        labels_sha256=digest(labels_path),
        train_indices=IDS,
        feature_orders={"26": original["feature_order"]+EXTRA, "32": original["feature_order"]+EXTRA+GRAPH_ORDER},
        graph_capture_manifest_sha256=digest(GRAPH/"manifest.json"),
        ridge_lambda=0.01,
        fitted_layers=list(range(1, 15)),
        first_layer_zero=True,
        last_layer_zero=True,
        folds=[[IDS[i] for i in range(f * 6, (f + 1) * 6)] for f in range(4)],
        target="MCpost4−ownpost4beforecurrentreset, normalizedsigma^1..4; priorlayersoraclecorrected",
        single_change="feature26→32 six graphcontext appended; fixedperlayerlinear/localtarget both",
        fixed_signal_gate="atleastonechannel model32/model26 normalizedMSE<.90 with>=16/24wins and networkbootstrap95upper<1; allchannels report especiallymean/variance; no validation tuning",
        trace_provenance=provenance,
        scope="train24teacherforcednetworkCV, not closed learned model score; MCoracle covariate noise/exposurebias remain",
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cvrows = []
    for fold in range(4):
        held = list(range(fold * 6, (fold + 1) * 6))
        train = [i for i in range(24) if i not in held]
        prediction = {size: {i: np.zeros((15, 1024, 4)) for i in held} for size in (26, 32)}
        for layer in range(1, 15):
            for size in (26, 32):
                model = fit(data, train, layer, size)
                for i in held:
                    prediction[size][i][layer] = infer(data[i], model, layer, size)
        for i in held:
            errors = {"baseline": data[i]["y"]}
            errors.update({str(size): data[i]["y"] - prediction[size][i] for size in (26, 32)})
            cvrows.append(
                dict(
                    idx=IDS[i],
                    fold=fold,
                    normalized_mse={
                        k: (value[1:] ** 2).mean(axis=(0, 1)).tolist()
                        for k, value in errors.items()
                    },
                    raw_mse={
                        k: ((value[1:] * data[i]["scales"][1:]) ** 2).mean(axis=(0, 1)).tolist()
                        for k, value in errors.items()
                    },
                    normalized_mse_by_layer={
                        k: (value**2).mean(axis=1).tolist() for k, value in errors.items()
                    },
                )
            )
        print(json.dumps(dict(fold=fold, elapsed_s=time.monotonic() - started)), flush=True)
    metrics = {
        k: np.asarray([row["normalized_mse"][k] for row in cvrows])
        for k in ("baseline", "26", "32")
    }
    result = dict(
        complete=True,
        rows=cvrows,
        channels=["mean", "variance", "kappa3", "kappa4"],
        pooled_normalized_mse={k: v.mean(axis=0).tolist() for k, v in metrics.items()},
        elapsed_s=time.monotonic() - started,
        models={},
    )
    for size in (26, 32):
        ratios = metrics[str(size)].mean(axis=0) / metrics["baseline"].mean(axis=0)
        wins = (metrics[str(size)] < metrics["baseline"]).sum(axis=0)
        model = dict(
            feature_order=manifest["feature_orders"][str(size)],
            n_features=size,
            layer_models={
                str(layer): serialize(fit(data, list(range(24)), layer, size))
                for layer in range(1, 15)
            },
            sigma_output_powers=[1, 2, 3, 4],
            train_indices=IDS,
            first_layer_zero=True,
            last_layer_zero=True,
            ridge_lambda=0.01,
            target=manifest["target"],
            corrected_channels=["mean", "variance", "kappa3", "kappa4"],
            post_semantics="signedcentraldiag; fixedcentralcross; originalnonlinearfeed; ordinaryMdiag/Cdiag/G rebuilt",
            floors=dict(mean=0, variance=1e-10),
        )
        path = OUT / f"frozen_train24_local_closure_{size}.json"
        path.write_text(json.dumps(model, indent=2) + "\n")
        result["models"][str(size)] = dict(
            model_sha256=digest(path),
            model_path=str(path),
            normalized_ratio_vs_baseline=ratios.tolist(),
            network_wins_vs_baseline=wins.tolist(),
            development_signal_vs_zero_correction_channels=((ratios < 0.80) & (wins >= 16)).tolist(),
        )
    rawmetrics={k:np.asarray([row["raw_mse"][k] for row in cvrows]) for k in ("baseline","26","32")}
    result["pooled_physical_raw_mse"]={k:v.mean(axis=0).tolist() for k,v in rawmetrics.items()}
    result["model32_over26_physical_raw"]=(rawmetrics["32"].mean(axis=0)/rawmetrics["26"].mean(axis=0)).tolist()
    ratio32_26=metrics["32"].mean(axis=0)/metrics["26"].mean(axis=0)
    wins32_26=(metrics["32"]<metrics["26"]).sum(axis=0)
    result["model32_over26_normalized"]=ratio32_26.tolist()
    result["model32_network_wins_over26"]=wins32_26.tolist()
    r48=json.loads(Path("results/phase2/local_joint_closure_cv_r48/completion.json").read_text())
    old26=np.asarray(r48["pooled_normalized_mse"]["26"])
    new26=metrics["26"].mean(axis=0)
    result["r48_26_pooled_maxabs_difference"]=float(np.max(abs(new26-old26)))
    assert np.max(abs(new26-old26))<1e-15
    rng=np.random.default_rng(261053)
    samples=rng.integers(0,24,size=(20000,24))
    bootstrap=metrics["32"][samples].mean(axis=1)/metrics["26"][samples].mean(axis=1)
    result["paired_network_bootstrap32_over26_normalized95"]=np.quantile(bootstrap,[.025,.975],axis=0).T.tolist()
    upper=np.quantile(bootstrap,.975,axis=0)
    signal=(ratio32_26<.90)&(wins32_26>=16)&(upper<1)
    result["fixed_graph_signal_channels_passed"]=signal.tolist()
    result["fixed_graph_signal_gate_passed"]=bool(signal.any())
    rawbootstrap=rawmetrics["32"][samples].mean(axis=1)/rawmetrics["26"][samples].mean(axis=1)
    result["paired_network_bootstrap32_over26_physical_raw95"]=np.quantile(rawbootstrap,[.025,.975],axis=0).T.tolist()
    result["bootstrap_scope"]="network24conditional teacherforceddevelopment only; excludes MC sampling uncertainty and exposurebias; no neuron-iid inference"
    result["bootstrap_seed"]=261053
    result["bootstrap_replicates"]=20000
    (OUT / "completion.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()

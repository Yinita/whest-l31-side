"""Fixed quadratic closure mechanism, network CV within corpus train24 only.

Never loads validation8 predictions or their labels. Degree2 and ridge0.01
are fixed before evaluation. CV is a mechanism gate, not a competition score.
"""

# ruff: noqa: E402
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import hashlib
import json
import time
from pathlib import Path

import numpy as np

DIRECTORY = Path("results/phase2/modal_joint_closure_pilot_r38_r02")
OUT = Path("results/phase2/joint_closure_quadratic_cv_r41")
INDICES = list(range(0, 384, 16))
PARENT_SHA = "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
DATA_REV = "cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8"
PAIRS = [(i, j) for i in range(1, 22) for j in range(i, 22)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(x):
    return np.concatenate([x, np.stack([x[:, i] * x[:, j] for i, j in PAIRS], axis=1)], axis=1)


def fit(data, indices, quadratic):
    count = sum(len(data[i]["x"]) for i in indices)
    mean = sum(data[i]["x"].sum(axis=0) for i in indices) / count
    variance = sum((data[i]["x"] ** 2).sum(axis=0) for i in indices) / count - mean**2
    scale = np.sqrt(np.maximum(variance, 1e-24))
    mean[0], scale[0] = 0, 1
    mapped_mean, mapped_scale = None, None
    if quadratic:
        total = np.zeros(22 + len(PAIRS))
        total2 = total.copy()
        for i in indices:
            x = expand((data[i]["x"] - mean) / scale)
            total += x.sum(axis=0)
            total2 += (x * x).sum(axis=0)
        mapped_mean = total / count
        mapped_scale = np.sqrt(np.maximum(total2 / count - mapped_mean**2, 1e-24))
        mapped_mean[0], mapped_scale[0] = 0, 1
    size = 22 + len(PAIRS) if quadratic else 22
    gram, cross = np.zeros((size, size)), np.zeros((size, 4))
    for i in indices:
        x = (data[i]["x"] - mean) / scale
        if quadratic:
            x = (expand(x) - mapped_mean) / mapped_scale
        gram += x.T @ x
        cross += x.T @ data[i]["y"]
    penalty = np.eye(size) * 0.01 * count
    penalty[0, 0] = 0
    theta = np.linalg.solve(gram + penalty, cross)
    return dict(
        theta=theta,
        feature_mean=mean,
        feature_scale=scale,
        mapped_mean=mapped_mean,
        mapped_scale=mapped_scale,
        quadratic=quadratic,
        ridge_lambda=0.01,
        train_indices=[INDICES[i] for i in indices],
    )


def predict(data, model):
    x = (data["x"] - model["feature_mean"]) / model["feature_scale"]
    if model["quadratic"]:
        x = (expand(x) - model["mapped_mean"]) / model["mapped_scale"]
    return x @ model["theta"]


def serializable(model):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    parent = json.loads((DIRECTORY / "manifest.json").read_text())
    assert parent["h18_sha256"] == PARENT_SHA and parent["dataset_revision"] == DATA_REV
    assert parent["train_indices"] == INDICES
    labels_path = DIRECTORY / "research_labels.npz"
    # Slice train24 at loading; validation8 moments never participate.
    raw = np.moveaxis(
        np.load(labels_path, allow_pickle=False)["moments"][:24, :, :15, :].astype(np.float64),
        1,
        -1,
    )
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
    data, paths = [], []
    for position, idx in enumerate(INDICES):
        path = DIRECTORY / f"baseline_idx{idx:04d}.npz"
        trace = np.load(path, allow_pickle=False)
        x, state, powers = trace["features"], trace["states"], trace["scales"]
        assert x.shape == (15, 1024, 22) and state.shape == powers.shape == (15, 1024, 4)
        residual = (target[position] - state) / powers
        data.append(
            dict(x=x.reshape(-1, 22), y=residual.reshape(-1, 4), scales=powers.reshape(-1, 4))
        )
        paths.append(dict(idx=idx, path=str(path), sha256=sha(path)))
    manifest = dict(
        version="quadratic-joint-closure-cv-r41",
        pid=os.getpid(),
        script_sha256=sha(Path(__file__)),
        baseline_sha256=PARENT_SHA,
        dataset_revision=DATA_REV,
        labels_sha256=sha(labels_path),
        train_indices=INDICES,
        validation_indices_not_used=parent["validation_indices"],
        feature_order=parent["feature_order"],
        quadratic_pairs=PAIRS,
        folds=[[INDICES[i] for i in range(j * 6, (j + 1) * 6)] for j in range(4)],
        ridge_lambda=0.01,
        degree=2,
        n_features=22 + len(PAIRS),
        normalization="raw and mapped columns fit only fold training18 networks",
        signal_gate="anychannel quadratic/linear pooledCV normalizedMSE<=.90 and >=16/24 networkwins; fixedbeforeCV",
        source_traces=paths,
        limits=[
            "train24 network-CV is development",
            "MCN1e8 labels",
            "no admissibility/score/stage claim",
        ],
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with (OUT / "journal.jsonl").open("x") as journal:
        for fold in range(4):
            held = list(range(fold * 6, (fold + 1) * 6))
            train = [i for i in range(24) if i not in held]
            linear = fit(data, train, False)
            quadratic = fit(data, train, True)
            for i in held:
                y = data[i]["y"]
                errors = {
                    "baseline": y,
                    "linear": y - predict(data[i], linear),
                    "quadratic": y - predict(data[i], quadratic),
                }
                row = dict(
                    idx=INDICES[i],
                    fold=fold,
                    normalized_mse={
                        k: (value**2).mean(axis=0).tolist() for k, value in errors.items()
                    },
                    raw_mse={
                        k: ((value * data[i]["scales"]) ** 2).mean(axis=0).tolist()
                        for k, value in errors.items()
                    },
                    normalized_mse_by_layer={
                        k: (value.reshape(15, 1024, 4) ** 2).mean(axis=1).tolist()
                        for k, value in errors.items()
                    },
                )
                rows.append(row)
                journal.write(json.dumps(row) + "\n")
                journal.flush()
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "held_indices": [INDICES[i] for i in held],
                        "elapsed_s": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    scores = {
        k: np.array([row["normalized_mse"][k] for row in rows])
        for k in ("baseline", "linear", "quadratic")
    }
    ratios = scores["quadratic"].mean(axis=0) / scores["linear"].mean(axis=0)
    wins = (scores["quadratic"] < scores["linear"]).sum(axis=0)
    signal = (ratios <= 0.90) & (wins >= 16)
    result = dict(
        complete=True,
        rows=rows,
        signal_gate_passed=bool(signal.any()),
        channel_order=["mean", "variance", "kappa3", "kappa4"],
        pooled_normalized_mse={k: v.mean(axis=0).tolist() for k, v in scores.items()},
        quadratic_over_linear=ratios.tolist(),
        network_wins_vs_linear=wins.tolist(),
        signal_channels=signal.tolist(),
        elapsed_s=time.monotonic() - started,
    )
    if signal.any():
        frozen = serializable(fit(data, list(range(24)), True))
        frozen.update(feature_order=parent["feature_order"], quadratic_pairs=PAIRS)
        model_path = OUT / "frozen_train24_quadratic.json"
        model_path.write_text(json.dumps(frozen, indent=2) + "\n")
        result["frozen_model_sha256"] = sha(model_path)
        result["frozen_model_scope"] = (
            "train24 only; requires independent held-network closed rollout and legal estimator"
        )
    (OUT / "completion.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()

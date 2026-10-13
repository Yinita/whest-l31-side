"""Fixed train-network CV for per-layer ridge; no validation label access.

Only the original 24 train network rows are read. Analytic layer0 is fixed zero.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

PARENT = Path("results/phase2/modal_joint_closure_pilot_r38_r02")
OUT = Path("results/phase2/joint_closure_per_layer_cv_r42")
TRAIN = list(range(0, 384, 16))


def fit(xs, ys):
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    mean[0], scale[0] = 0, 1
    scale = np.maximum(scale, 1e-12)
    z = (x-mean)/scale
    penalty = np.eye(x.shape[1])*.01*len(x)
    penalty[0, 0] = 0
    theta = np.linalg.solve(z.T@z+penalty, z.T@y)
    return dict(theta=theta.tolist(), feature_mean=mean.tolist(), feature_scale=scale.tolist())


def predict(model, x):
    return (x-np.asarray(model["feature_mean"]))/np.asarray(model["feature_scale"]) @ np.asarray(model["theta"])


def main():
    OUT.mkdir(exist_ok=False)
    manifest = json.loads((PARENT/"manifest.json").read_text())
    assert manifest["train_indices"] == TRAIN
    labels = np.load(PARENT/"research_labels.npz", allow_pickle=False)
    # This copy is the only target slice used. Validation rows are never touched.
    raw = labels["moments"][:24, :, :15].astype(np.float64)
    m1, m2, m3, m4 = [raw[:, i] for i in range(4)]
    variance = m2-m1*m1
    targets = np.stack([m1, variance, m3-3*m1*m2+2*m1**3,
                        m4-4*m1*m3+6*m1*m1*m2-3*m1**4-3*variance**2], axis=-1)
    arrays = [np.load(PARENT/f"baseline_idx{idx:04d}.npz", allow_pickle=False) for idx in TRAIN]
    residuals = [(targets[i]-data["states"])/data["scales"] for i, data in enumerate(arrays)]
    rows = []
    for fold in range(4):
        train = [i for i in range(24) if i % 4 != fold]
        held = [i for i in range(24) if i % 4 == fold]
        shared = fit([arrays[i]["features"].reshape(-1, 22) for i in train],
                     [residuals[i].reshape(-1, 4) for i in train])
        layer_models = [None] + [fit([arrays[i]["features"][layer] for i in train],
                                     [residuals[i][layer] for i in train]) for layer in range(1, 15)]
        for i in held:
            data = arrays[i]
            shared_error = predict(shared, data["features"])*data["scales"]
            layer_error = np.stack([np.zeros((1024, 4))] +
                                  [predict(layer_models[layer], data["features"][layer])
                                   for layer in range(1, 15)])*data["scales"]
            before = np.mean((data["states"]-targets[i])**2, axis=(0, 1))
            shared_mse = np.mean((data["states"]+shared_error-targets[i])**2, axis=(0, 1))
            layer_mse = np.mean((data["states"]+layer_error-targets[i])**2, axis=(0, 1))
            rows.append(dict(idx=TRAIN[i], fold=fold, before=before.tolist(),
                             shared_linear=shared_mse.tolist(), per_layer=layer_mse.tolist()))
    shared_avg = np.mean([row["shared_linear"] for row in rows], axis=0)
    layer_avg = np.mean([row["per_layer"] for row in rows], axis=0)
    models = [dict(theta=np.zeros((22, 4)).tolist(), feature_mean=np.zeros(22).tolist(),
                   feature_scale=np.ones(22).tolist())]
    models += [fit([data["features"][layer] for data in arrays],
                   [residual[layer] for residual in residuals]) for layer in range(1, 15)]
    model_path = OUT/"frozen_train24_per_layer.json"
    model_path.write_text(json.dumps(dict(layer_models=models, feature_order=manifest["feature_order"],
                                         train_indices=TRAIN, lambda_fixed=.01,
                                         layer0="analytic known distribution; zero residual, no fitting"), indent=2)+"\n")
    report = dict(complete=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  feature_order=manifest["feature_order"], train_indices=TRAIN,
                  fresh_validation_indices=list(range(512, 640, 16)),
                  folds="network i modulo4, 18train6held", rows=sorted(rows, key=lambda row: row["idx"]),
                  aggregate_shared=shared_avg.tolist(), aggregate_per_layer=layer_avg.tolist(),
                  per_layer_over_shared=(layer_avg/shared_avg).tolist(),
                  wins_over_shared=[sum(row["per_layer"][c] < row["shared_linear"][c] for row in rows) for c in range(4)],
                  scope="Train24 networkCV only; fixed per-layer mechanism/no held8 labels/no candidate or stage claim")
    (OUT/"completion.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("model_sha256", "per_layer_over_shared", "wins_over_shared")}))


if __name__ == "__main__":
    main()

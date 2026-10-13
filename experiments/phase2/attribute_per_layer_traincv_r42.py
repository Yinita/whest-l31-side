"""Separate per-layer model effect from exact-layer0 constraint in trainCV."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.phase2.train_per_layer_joint_closure_r42 import fit, predict


def main():
    parent = Path("results/phase2/modal_joint_closure_pilot_r38_r02")
    output = Path("results/phase2/joint_closure_per_layer_cv_r42/analytic_layer0_attribution.json")
    assert not output.exists()
    raw = np.load(parent/"research_labels.npz")["moments"][:24, :, :15].astype(float)
    m1, m2, m3, m4 = [raw[:, i] for i in range(4)]
    var = m2-m1*m1
    targets = np.stack([m1, var, m3-3*m1*m2+2*m1**3,
                        m4-4*m1*m3+6*m1*m1*m2-3*m1**4-3*var*var], -1)
    arrays = [np.load(parent/f"baseline_idx{i*16:04d}.npz") for i in range(24)]
    residual = [(targets[i]-data["states"])/data["scales"] for i, data in enumerate(arrays)]
    rows = []
    depth_shared = np.zeros((14, 4))
    depth_layer = np.zeros((14, 4))
    for fold in range(4):
        train = [i for i in range(24) if i % 4 != fold]
        held = [i for i in range(24) if i % 4 == fold]
        model = fit([arrays[i]["features"].reshape(-1, 22) for i in train],
                    [residual[i].reshape(-1, 4) for i in train])
        layers = [fit([arrays[i]["features"][li] for i in train],
                      [residual[i][li] for i in train]) for li in range(1, 15)]
        for i in held:
            data = arrays[i]
            shared_delta = predict(model, data["features"])*data["scales"]
            shared_delta[0] = 0
            layer_delta = np.zeros_like(shared_delta)
            for li in range(1, 15):
                layer_delta[li] = predict(layers[li-1], data["features"][li])*data["scales"][li]
            shared_mse = np.mean((data["states"]+shared_delta-targets[i])**2, axis=1)
            layer_mse = np.mean((data["states"]+layer_delta-targets[i])**2, axis=1)
            depth_shared += shared_mse[1:]/24
            depth_layer += layer_mse[1:]/24
            rows.append(dict(idx=i*16, fold=fold, shared_zero_layer0=shared_mse.mean(0).tolist(),
                             per_layer_zero_layer0=layer_mse.mean(0).tolist(),
                             shared_only_layers1_14=shared_mse[1:].mean(0).tolist(),
                             per_layer_only_layers1_14=layer_mse[1:].mean(0).tolist()))
    shared_avg = np.mean([row["shared_zero_layer0"] for row in rows], 0)
    layer_avg = np.mean([row["per_layer_zero_layer0"] for row in rows], 0)
    shared_only = np.mean([row["shared_only_layers1_14"] for row in rows], 0)
    layer_only = np.mean([row["per_layer_only_layers1_14"] for row in rows], 0)
    result = dict(complete=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=sorted(rows, key=lambda row: row["idx"]),
                  per_layer_over_shared_both_layer0_zero=(layer_avg/shared_avg).tolist(),
                  per_layer_over_shared_layers1_14=(layer_only/shared_only).tolist(),
                  depth_layers1_14=[dict(layer=li, shared=depth_shared[li-1].tolist(),
                                        per_layer=depth_layer[li-1].tolist(),
                                        ratio=(depth_layer[li-1]/depth_shared[li-1]).tolist()) for li in range(1, 15)],
                  scope="Train24samefold refit for attribution only, no model/outputselection, no val8labels read. Frozen model unchanged.")
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ["per_layer_over_shared_both_layer0_zero", "per_layer_over_shared_layers1_14"]}))


if __name__ == "__main__":
    main()

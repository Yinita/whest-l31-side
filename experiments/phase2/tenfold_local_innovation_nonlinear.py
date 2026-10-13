"""One fixed nonlinear learner on frozen r01 local-innovation LOO protocol.

Research environment only: scikit-learn is never an estimator dependency.
No hyperparameter scan; whole-network LOO, targets only label training folds.
"""

# ruff: noqa: E402
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tenfold_local_innovation_pilot import design


def learner():
    return HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05,
                                         max_iter=100, max_depth=3, min_samples_leaf=128,
                                         early_stopping=False, random_state=261013601)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    root = Path(__file__).resolve().parents[2]
    before_path = root / "results/phase2/tenfold_local_innovation_pilot_r01.json"
    original = json.loads(before_path.read_text())
    start = time.monotonic()
    inputs = json.loads((root / "benchmark/data/rpc_inputs/sprint_public_v37b_h18_r04_rpc8/inputs.json").read_text())
    data, weights = [], []
    for i in range(8):
        trace = root / f"benchmark/data/v29_features_v30_096/network_{i}.npz"
        wp = Path(inputs[i]["weights"])
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == original["identities"][i]["trace_sha256"]
        assert hashlib.sha256(wp.read_bytes()).hexdigest() == original["identities"][i]["weights_sha256"]
        data.append(dict(np.load(trace)))
        with np.load(wp) as archive:
            weights.append(archive["weights"].astype(np.float64))
    errors = [d["reference"].astype(np.float64) - d["predictions"].astype(np.float64) for d in data]
    features = [[design(d["features"][layer]) for layer in range(16)] for d in data]
    phi = [d["features"][:, :, 7].astype(np.float64) for d in data]
    labels = []
    for i in range(8):
        local = np.zeros_like(errors[i])
        for layer in range(1, 16):
            local[layer] = errors[i][layer] - phi[i][layer] * (weights[i][layer].T @ errors[i][layer - 1])
        labels.append(local)
    rows = []
    for held in range(8):
        train = [i for i in range(8) if i != held]
        delta = np.zeros_like(errors[held])
        layer_rows = []
        for layer in range(1, 15):
            x = np.concatenate([features[i][layer] for i in train])
            y = np.concatenate([labels[i][layer] for i in train])
            # Absolute squared loss is invariant to train-only scaling. Scaling
            # helps tree numeric splits; no held-label statistic is available.
            scale = float(np.std(y))
            if scale < 1e-30:
                predicted = np.zeros(1024)
            else:
                model = learner().fit(x, y / scale)
                predicted = model.predict(features[held][layer]) * scale
            delta[layer] = phi[held][layer] * (weights[held][layer].T @ delta[layer - 1]) + predicted
            layer_rows.append({"layer": layer, "before_local_mse": float(np.mean(labels[held][layer]**2)),
                               "after_local_mse": float(np.mean((labels[held][layer] - predicted)**2)),
                               "before_total_mse": float(np.mean(errors[held][layer]**2)),
                               "after_replay_total_mse": float(np.mean((errors[held][layer] - delta[layer])**2)),
                               "train_label_scale": scale})
        delta[15] = phi[held][15] * (weights[held][15].T @ delta[14])
        before = float(np.mean(errors[held][15]**2))
        after = float(np.mean((errors[held][15] - delta[15])**2))
        rows.append({"held_network_id": held, "train_network_ids": train,
                     "before_raw": before, "after_nonlinear_local_replay_raw": after,
                     "after_linear_local_replay_raw": original["rows"][held]["after_local_innovation_replay_raw"],
                     "raw_ratio": after / before, "layers": layer_rows})
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "layers"}), flush=True)
    before = float(np.mean([r["before_raw"] for r in rows]))
    after = float(np.mean([r["after_nonlinear_local_replay_raw"] for r in rows]))
    proxy = json.loads((root / "results/phase2/reference_noise_proxy_mini8.json").read_text())
    report = {"version": "tenfold-local-innovation-r02-nonlinear", "complete": True,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "parent_report_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
              "parent_script_sha256": hashlib.sha256((Path(__file__).parent / "tenfold_local_innovation_pilot.py").read_bytes()).hexdigest(),
              "single_change": "replace ridge linear model with fixed shallow histogram gradient boosting; same labels18features/layersLOO",
              "libraries": {"python": sys.version, "numpy": np.__version__, "scikit-learn": sklearn.__version__},
              "fixed_learner": learner().get_params(), "early_stopping": False,
              "before_raw": before, "after_linear_local_replay_raw": original["after_local_innovation_replay_raw"],
              "after_nonlinear_local_replay_raw": after, "raw_ratio": after / before,
              "optimistic_floor_adjusted": after * 0.1,
              "production_gate_required_raw_ratio": 0.25,
              "production_gate_pass": after / before < 0.25,
              "reference_noise_iid_proxy": proxy["mean_iid_raw_proxy"],
              "rows": rows, "elapsed_s": time.monotonic() - start,
              "adjusted_score": None,
              "limitations": original["limitations"] + ["No covariance feedback: cached expected-gate meanJacobian only",
                                                        "Trees trained offline, not allowed runtime dependency; production representation unimplemented",
                                                        "Reference noise proxy is not observed CI or label-denoising guarantee",
                                                        "After linear parent already observed; nonlinear configuration fixed before this run"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("before_raw", "after_linear_local_replay_raw", "after_nonlinear_local_replay_raw", "raw_ratio", "production_gate_pass", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()

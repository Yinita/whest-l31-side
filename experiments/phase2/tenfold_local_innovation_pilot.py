"""Network-LOO local innovation regression using existing frozen V29 traces.

Offline diagnostic only: references form training labels, never features.
The holdout network's reference is only read to score, not fit or select.
Compare local versus total-error labels with identical fixed features/ridge.
Replay is first order and does not prove a changed estimator's resource pass.
"""

# ruff: noqa: E402
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def design(features):
    alpha, phi, sigma, d3 = [features[:, i] for i in (4, 6, 3, 9)]
    skew = d3 / np.maximum(sigma, 1e-8)**3
    extras = np.stack([alpha**k * phi for k in (2, 3, 4)] + [skew, skew * phi], axis=1)
    return np.concatenate([features, extras], axis=1).astype(np.float64)


def fit(xs, ys, ridge=1e-3):
    x, y = np.concatenate(xs), np.concatenate(ys)
    mean, std = x.mean(axis=0), x.std(axis=0)
    constant = std < 1e-12
    mean[constant], std[constant] = 0, 1
    xx = (x - mean) / std
    beta = np.linalg.solve(xx.T @ xx + ridge * np.eye(xx.shape[1]), xx.T @ y)
    return mean, std, beta


def apply(model, features):
    mean, std, beta = model
    return ((features - mean) / std) @ beta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    start = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    cache_root = root / "benchmark/data/v29_features_v30_096"
    input_manifest = root / "benchmark/data/rpc_inputs/sprint_public_v37b_h18_r04_rpc8/inputs.json"
    manifest = json.loads(input_manifest.read_text())
    trace_manifest = json.loads((root / "results/phase2/ablation_v30_096_features_manifest.json").read_text())
    data, weights, identities = [], [], []
    for i in range(8):
        path = cache_root / f"network_{i}.npz"
        expected = trace_manifest["records"][i]
        assert expected["network_id"] == manifest[i]["index"] == i
        assert expected["name"] == manifest[i]["name"] and expected["seed"] == manifest[i]["seed"]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == expected["sha256"]
        data.append(dict(np.load(path)))
        wp = Path(manifest[i]["weights"])
        wsha = hashlib.sha256(wp.read_bytes()).hexdigest()
        assert wsha == manifest[i]["sha256"]
        with np.load(wp) as archive:
            assert archive.files == ["weights"]
            weights.append(archive["weights"].astype(np.float64))
        identities.append({"network_id": i, "name": expected["name"], "seed": expected["seed"],
                           "trace_sha256": sha, "weights_sha256": wsha})
    # MLP export preserves dataset weights. Both estimator and original diagnostic
    # use W=weights[layer].T, pre=W@previous_mean (no V37 permutation in V29).
    errors = [(d["reference"].astype(np.float64) - d["predictions"].astype(np.float64)) for d in data]
    features = [[design(d["features"][layer]) for layer in range(16)] for d in data]
    phi = [d["features"][:, :, 7].astype(np.float64) for d in data]
    innovations = []
    for i in range(8):
        local = np.zeros_like(errors[i])
        for layer in range(1, 16):
            transported = phi[i][layer] * (weights[i][layer].T @ errors[i][layer - 1])
            local[layer] = errors[i][layer] - transported
        innovations.append(local)
    # Verify exact decomposition on same trace against its archived diagnostic.
    old = json.loads((root / "results/phase2/ablation_v30_119_error_transport.json").read_text())["rows"]
    decomposition_maxdelta = max(abs(float(np.mean(innovations[row["network_id"]][row["layer"]]**2)) -
                                       row["local_residual_mse"]) for row in old)
    assert decomposition_maxdelta < 2e-16
    rows, folds = [], []
    for held in range(8):
        train = [i for i in range(8) if i != held]
        local_delta = np.zeros_like(errors[held])
        total_readout = np.zeros_like(errors[held])
        per_layer, coefficients = [], []
        # Layer15 is deliberately not trained: isolate inherited correction from
        # layers1..14, then propagate it to the final readout with unchanged gate.
        for layer in range(1, 15):
            xs = [features[i][layer] for i in train]
            local_model = fit(xs, [innovations[i][layer] for i in train])
            total_model = fit(xs, [errors[i][layer] for i in train])
            local_hat = apply(local_model, features[held][layer])
            total_readout[layer] = apply(total_model, features[held][layer])
            local_delta[layer] = phi[held][layer] * (weights[held][layer].T @ local_delta[layer - 1]) + local_hat
            per_layer.append({"layer": layer, "before_total_mse": float(np.mean(errors[held][layer]**2)),
                              "after_innovation_replay_mse": float(np.mean((errors[held][layer] - local_delta[layer])**2)),
                              "after_total_error_readout_mse": float(np.mean((errors[held][layer] - total_readout[layer])**2)),
                              "before_local_innovation_mse": float(np.mean(innovations[held][layer]**2)),
                              "after_local_innovation_mse": float(np.mean((innovations[held][layer] - local_hat)**2))})
            coefficients.append({"layer": layer, "local_mean": local_model[0].tolist(),
                                 "local_std": local_model[1].tolist(), "local_beta": local_model[2].tolist(),
                                 "total_mean": total_model[0].tolist(), "total_std": total_model[1].tolist(),
                                 "total_beta": total_model[2].tolist()})
        local_delta[15] = phi[held][15] * (weights[held][15].T @ local_delta[14])
        total_readout[15] = phi[held][15] * (weights[held][15].T @ total_readout[14])
        before = float(np.mean(errors[held][15]**2))
        after = float(np.mean((errors[held][15] - local_delta[15])**2))
        old_after = float(np.mean((errors[held][15] - total_readout[15])**2))
        rows.append({"held_network_id": held, "train_network_ids": train,
                     "before_raw": before, "after_local_innovation_replay_raw": after,
                     "after_total_error_parent_readout_raw": old_after,
                     "local_replay_ratio": after / before, "per_layer": per_layer})
        folds.append({"held_network_id": held, "train_network_ids": train, "coefficients": coefficients})
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "per_layer"}), flush=True)
    before = float(np.mean([r["before_raw"] for r in rows]))
    after = float(np.mean([r["after_local_innovation_replay_raw"] for r in rows]))
    report = {"version": "tenfold-local-innovation-r01", "complete": True,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "base_estimator_sha256": trace_manifest["source_sha256"],
              "single_primary_mechanism": "fit local error innovation instead of full accumulated error; frozen18features/ridge",
              "feature_names": trace_manifest["feature_order"] + ["alpha2phi", "alpha3phi", "alpha4phi", "D3_sigma3", "D3_sigma3_phi"],
              "predeclared": {"ridge": 1e-3, "train_layers": list(range(1, 15)),
                              "held_network_rule": "whole-network leave-one-out; no held-label fit or coefficient selection",
                              "layer0_correction": 0, "layer15_local_correction": 0},
              "identities": identities, "decomposition_maxdelta_mse": decomposition_maxdelta,
              "before_raw": before, "after_local_innovation_replay_raw": after,
              "raw_ratio": after / before,
              "after_total_error_parent_readout_raw": float(np.mean([r["after_total_error_parent_readout_raw"] for r in rows])),
              "rows": rows, "folds": folds, "elapsed_s": time.monotonic() - start,
              "adjusted_score": None,
              "scope": "Offline Mini8 networkLOO first-order correction replay; no new production candidate or independentholdout",
              "limitations": ["Frozen V29 traces, not best V37bH18; cached phi/state are unchanged by correction",
                              "Teacher local labels depend on publicreference only in training; no target at inference",
                              "References also score held network; reused development set with prior selections",
                              "No covariance/cumulant feedback or nonlinear trajectory recomputation",
                              "Transport and fitting ordinary NumPy, no flopscope score/resource evidence",
                              "Fixed features insufficient does not rule out new actualcore features"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("version", "before_raw", "after_local_innovation_replay_raw", "raw_ratio", "after_total_error_parent_readout_raw", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()

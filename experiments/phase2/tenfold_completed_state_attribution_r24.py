"""Offline before/after attribution, never a truth-fitted production correction."""
# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mse(value):
    return float(np.mean(value**2))


def cosine(left, right):
    denominator = np.linalg.norm(left)*np.linalg.norm(right)
    return float(np.dot(left, right)/denominator) if denominator else None


def propagated(local, cdf, weights):
    current = np.zeros(1024)
    output = []
    for layer in range(16):
        current = local[layer] + (0 if layer == 0 else cdf[layer]*(weights[layer].T @ current))
        output.append(current.copy())
    return np.stack(output)


def main():
    output = Path("results/phase2/tenfold_completed_state_attribution_r24.json")
    assert not output.exists()
    root = Path("results/phase2/modal_covariance_causal_r02")
    aggregate_path = root/"aggregate_final_r08.npz"
    aggregate_meta = json.loads((root/"attribution_final_r08.json").read_text())
    assert sha(aggregate_path) == aggregate_meta["aggregate_sha256"]
    cloud = np.load(aggregate_path)
    weights_path = Path("benchmark/data/rpc_inputs/mlp_0.npz")
    weights = np.load(weights_path)["weights"].astype(np.float64)
    reference_path = Path("benchmark/data/mini_first8/prepared/mini/data.arrow")
    target = Dataset.from_file(str(reference_path)).select_columns(["all_layer_means"])[0]
    truth = np.asarray(target["all_layer_means"], dtype=np.float64)
    paths = {"parent": Path("results/phase2/covariance_causal_parent_trace_r02.npz"),
             "r20": Path("results/phase2/tenfold_residual_remainder_r20_canonical_state_r01.npz")}
    states, rows = {}, []
    for name, path in paths.items():
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == metadata["output_sha256"]
        assert sha(weights_path) == metadata["weights_sha256"]
        cache = np.load(path)
        feature = cache["pre_features"].astype(np.float64)
        prediction = cache["predictions"].astype(np.float64)
        states[name] = feature
        error = truth-prediction
        local = np.stack([error[layer] if layer == 0 else
                          error[layer]-feature[layer, :, 4]*(weights[layer].T @ error[layer-1])
                          for layer in range(16)])
        effects = []
        for stream in (0, 1):
            count = int(cloud[f"stream{stream}_count"])
            variance = cloud[f"stream{stream}_variance"][0]
            cumulant3 = cloud[f"stream{stream}_central_third"][0]
            cumulant4 = cloud[f"stream{stream}_central_fourth"][0]-3*(variance*(count-1)/count)**2
            alpha, sigma, phi = feature[:, :, 0]/feature[:, :, 2], feature[:, :, 2], feature[:, :, 3]
            effects.append(np.stack([(variance-feature[:, :, 1])*phi/(2*sigma),
                                     (cumulant3-feature[:, :, 6])*(-alpha*phi)/(6*sigma**2),
                                     (cumulant4-feature[:, :, 7])*(alpha**2-1)*phi/(24*sigma**3)]))
        transport = [propagated(np.sum(effect, axis=0), feature[:, :, 4], weights) for effect in effects]
        strict_path = Path(f"results/phase2/tenfold_{'parent_h18_samecycle_r01' if name == 'parent' else 'residual_remainder_r20'}_strict1.json")
        strict = json.loads(strict_path.read_text())
        assert strict["manifest"]["completed"] and strict["per_mlp"][0]["status"] == "ok"
        assert strict["manifest"]["estimator_sha256"] == metadata["source_sha256"]
        strict_prediction = np.asarray(strict["per_mlp"][0]["predictions"], dtype=np.float64)
        layer_rows = []
        for layer in range(16):
            layer_rows.append({"layer": layer, "strict_mean_error_mse": mse(truth[layer]-strict_prediction[layer]),
                               "captured_mean_error_mse": mse(error[layer]), "local_innovation_mse": mse(local[layer]),
                               "trace_vs_strict_max_abs": float(np.max(abs(prediction[layer]-strict_prediction[layer]))),
                               "K2_K3_K4_effect_mse_stream": [[mse(value[layer]) for value in effect] for effect in effects],
                               "joint_local_positive_residual_mse": mse(local[layer]-sum(np.sum(e[:, layer], axis=0) for e in effects)/2),
                               "joint_local_cross_residual": float(np.mean((local[layer]-np.sum(effects[0][:, layer], axis=0))*(local[layer]-np.sum(effects[1][:, layer], axis=0)))),
                               "joint_chain_positive_residual_mse": mse(error[layer]-(transport[0][layer]+transport[1][layer])/2),
                               "joint_chain_cross_residual": float(np.mean((error[layer]-transport[0][layer])*(error[layer]-transport[1][layer])))})
        rows.append({"version": name, "trace_path": str(path), "trace_sha256": sha(path),
                     "source_sha256": metadata["source_sha256"], "strict_path": str(strict_path),
                     "strict_sha256": sha(strict_path), "layers": layer_rows})
    comparisons = []
    for layer in range(16):
        parent, candidate = states["parent"][layer], states["r20"][layer]
        moments = []
        for key, index in (("K2", 1), ("K3", 6), ("K4", 7)):
            update = candidate[:, index]-parent[:, index]
            missing = []
            for stream in (0, 1):
                count = int(cloud[f"stream{stream}_count"])
                variance = cloud[f"stream{stream}_variance"][0, layer]
                value = variance if key == "K2" else cloud[f"stream{stream}_central_third"][0, layer]
                if key == "K4":
                    value = cloud[f"stream{stream}_central_fourth"][0, layer]-3*(variance*(count-1)/count)**2
                missing.append(value-parent[:, index])
            moments.append({"moment": key, "update_rms": float(np.sqrt(mse(update))),
                            "update_missing_cosine_stream": [cosine(update, value) for value in missing],
                            "parent_missing_cross_mse": float(np.mean(missing[0]*missing[1])),
                            "candidate_missing_cross_mse": float(np.mean((missing[0]-update)*(missing[1]-update)))})
        comparisons.append({"layer": layer, "moments": moments})
    result = {"models": rows, "state_changes": comparisons,
              "cloud_successes": aggregate_meta["complete_jobs"], "cloud_failures": len(aggregate_meta["failures"]),
              "aggregate_sha256": sha(aggregate_path), "reference_sha256": sha(reference_path),
              "weights_sha256": sha(weights_path), "script_sha256": sha(Path(__file__)), "command": sys.argv,
              "scope": "Offline diagnosis only. Unchanged original reference, each captured state's own prediction/gain. Oracle moments never enter candidate. Fixed theoretical readout; no fitted scale. Stream MC/reference noise and trace drift retained. One public development network; no independent generalization, resource acceptance or tenfold claim."}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "last": [r["layers"][-1] for r in rows],
                      "last_state_change": comparisons[-1]}, indent=2))


if __name__ == "__main__":
    main()

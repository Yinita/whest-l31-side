"""Attribute completed strict negatives per layer; no estimator or fitted values."""

import hashlib
import json
from pathlib import Path

import numpy as np
from whestbench import load_dataset


def main():
    paths = {
        "parent": Path("results/phase2/tenfold_parent_h18_samecycle_r01_strict1.json"),
        "r25": Path("results/phase2/tenfold_sparse_d4_r25_strict1.json"),
        "r29": Path("results/phase2/tenfold_pair_grade4_consistent_r29_strict1.json"),
    }
    reports, hashes = {}, {}
    for name, path in paths.items():
        data = path.read_bytes()
        reports[name] = json.loads(data)
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        assert reports[name]["manifest"]["completed"]
        assert reports[name]["per_mlp"][0]["status"] == "ok"
    cohort = reports["parent"]["manifest"]
    for value in reports.values():
        for key in ("inputs_sha256", "dataset_commit", "network_ids", "limits"):
            assert value["manifest"][key] == cohort[key]
    dataset = load_dataset("benchmark/data/mini_first8")
    target = dataset[0]
    truth = np.asarray(target["all_layer_means"], dtype=np.float32)
    assert truth.shape == (16, 1024)
    predictions = {
        name: np.asarray(value["per_mlp"][0]["predictions"], dtype=np.float32)
        for name, value in reports.items()
    }
    rows = []
    for layer in range(16):
        row = {"layer": layer, "versions": {}}
        for name, prediction in predictions.items():
            error = prediction[layer] - truth[layer]
            row["versions"][name] = {
                "raw_mse_float32": float(np.mean(error * error)),
                "signed_error_mean": float(np.mean(error.astype(np.float64))),
                "centered_error_variance": float(np.var(error.astype(np.float64))),
            }
        rows.append(row)
    result = {
        "complete": True,
        "scope": "Completed development network0 attribution only, original references; no oracle candidate, fit or independent holdout",
        "inputs_sha256": cohort["inputs_sha256"],
        "dataset_commit": cohort["dataset_commit"],
        "report_sha256": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
    }
    out = Path("results/phase2/tenfold_completed_layer_scores_r36.json")
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows[1:3] + rows[-1:], indent=2))


if __name__ == "__main__":
    main()

"""Frozen M05 final-output diagnostic on one official 1024x16 network.

This returns only the final moment, so it is not a submission contract or an
official adjusted-score evaluation. Original reference is loaded after compute.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import resource
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from m05_float32_method import adaptive
from whestbench import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    path = Path(args.weights)
    weights = np.load(path)["weights"].astype(np.float32)
    assert weights.shape == (16, 1024, 1024)
    rows = []
    report = {"version": "paper.M05.r11", "complete": False, "rows": rows,
              "single_change": "max components1 ->9, same maxdepth12/threshold.02/paper_sum",
              "weights_npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "method_sha256": hashlib.sha256(Path("experiments/phase2/m05_float32_method.py").read_bytes()).hexdigest(),
              "versions": {p: importlib.metadata.version(p) for p in ("flopscope", "whestbench", "numpy")},
              "limits": {"flop_budget": 2**41, "wall_s": 120, "threads": 1},
              "scope": "Billed final-output math/local diagnostic only; no all-layer return contract or official adjusted score",
              "limitations": ["Component/depth cap is not convergence", "Public development network, no generalization claim",
                              "Reference noise unchanged", "No RPC resource acceptance"]}
    for cap in (1, 9):
        row = {"component_cap": cap, "depth_cap": 12, "threshold": .02,
               "dtype": "float32", "taylor_policy": "paper_sum", "failure": None}
        ctx = None
        try:
            with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=120, quiet=True) as ctx:
                prediction, _, mixture, history, evaluated = adaptive(
                    [fnp.asarray(w) for w in weights], 1024, .02, cap, 12, "paper_sum")
            row["prediction"] = np.asarray(prediction).tolist()
            row["accepted_components"] = len(mixture)
            row["evaluated_components"] = evaluated
            row["threshold_satisfied"] = all(not h["cap_or_depth_limited"] for h in history)
            row["history"] = history
            row["sum_leaf_weighted_criterion"] = sum(h["weighted_criterion"] for h in history if not h["split"])
        except Exception as exc:
            row["failure"] = f"{type(exc).__name__}: {exc}"
        if ctx is not None:
            row["legal_cost"] = ctx.summary_dict(by_namespace=True)
        row["peak_rss_kib_cumulative"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rows.append(row)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(cap, row.get("evaluated_components"), row.get("legal_cost", {}).get("flops_used"), row["failure"], flush=True)
    # No target or reference is supplied to adaptive() or used for split choices.
    data = load_dataset(args.dataset, split="mini")
    target = data.select_columns(["mlp_name", "mlp_seed", "final_means"])[0]
    truth = np.asarray(target["final_means"], dtype=np.float32)
    official_weights = np.asarray(data.select_columns(["weights"])[0]["weights"], dtype=np.float32)
    assert np.array_equal(weights, official_weights)
    report["network_identity"] = {"index": 0, "mlp_name": target["mlp_name"], "mlp_seed": target["mlp_seed"]}
    report["dataset_provenance"] = json.loads((Path(args.dataset)/"provenance.json").read_text())
    report["reference_sha256"] = hashlib.sha256(truth.tobytes()).hexdigest()
    for row in rows:
        if "prediction" in row:
            row["final_raw_mse_diagnostic"] = float(np.mean((np.asarray(row["prediction"], dtype=np.float64)-truth.astype(np.float64))**2))
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps([{k: r.get(k) for k in ("component_cap", "accepted_components", "threshold_satisfied", "final_raw_mse_diagnostic", "failure")} for r in rows], indent=2))


if __name__ == "__main__":
    main()

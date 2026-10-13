"""Separate depth-cap and component-cap effects with exact 1D references."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from m05_small_split_ut import adaptive
from m05_split_error_confirmation import relu_normal


def main():
    output = Path("results/phase2/m05_exact_cap_depth_ablation_r08.json")
    assert not output.exists()
    weights = np.array([[[1.2]], [[.8]], [[1.1]]])
    slope = float(np.prod(weights))
    exact = slope / math.sqrt(2*math.pi)
    rows = []
    for cap, depth in ((27, 5), (27, 12), (81, 12)):
        for policy in ("upstream_mean", "paper_sum"):
            with flops.BudgetContext(flop_budget=2**35, quiet=True) as ctx:
                predicted, _, mixture, history, evaluated = adaptive(
                    [fnp.asarray(w) for w in weights], 1, .02, cap, depth, policy)
            got = float(predicted.tolist()[0])
            split = sum(mass*slope*relu_normal(float(mean.tolist()[0]), float(cov.tolist()[0][0]))
                        for mass, mean, cov, _, _ in mixture)
            ut_bias, split_bias = got-split, split-exact
            terms = ut_bias**2 + split_bias**2 + 2*ut_bias*split_bias
            assert abs(terms-(got-exact)**2) < 1e-14
            rows.append({"component_cap": cap, "depth_cap": depth, "taylor_policy": policy,
                         "accepted_components": len(mixture), "evaluated_components": evaluated,
                         "threshold_satisfied": all(not h["cap_or_depth_limited"] for h in history),
                         "sum_leaf_weighted_criterion": sum(h["weighted_criterion"] for h in history if not h["split"]),
                         "max_leaf_weighted_criterion": max(h["weighted_criterion"] for h in history if not h["split"]),
                         "original_exact_mean": exact, "split_exact_mean": split, "ut_mean": got,
                         "ut_only_bias": ut_bias, "split_only_bias": split_bias,
                         "ut_only_mse": ut_bias**2, "split_only_mse": split_bias**2,
                         "cross_term": 2*ut_bias*split_bias, "total_mse": (got-exact)**2,
                         "legal_cost": ctx.summary_dict(), "history": history})
    previous = json.loads(Path("results/phase2/m05_split_error_confirmation_r05.json").read_text())
    for row in rows[:2]:
        before = next(x for x in previous["exact_1d"] if x["component_cap"] == 27 and x["policy"] == row["taylor_policy"])
        assert abs(row["total_mse"]-before["total_mse"]) < 1e-14
    report = {"version": "paper.M05.r08", "complete": True, "rows": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "method_sha256": hashlib.sha256(Path("experiments/phase2/m05_small_split_ut.py").read_bytes()).hexdigest(),
              "frozen_reference_report": "results/phase2/m05_split_error_confirmation_r05.json",
              "frozen_reference_report_sha256": hashlib.sha256(Path("results/phase2/m05_split_error_confirmation_r05.json").read_bytes()).hexdigest(),
              "single_change_pairs": ["cap27 depth5 -> depth12", "depth12 cap27 -> cap81"],
              "scope": "Exact 1D structural diagnosis, not competition MSE or highdimensional convergence",
              "criterion_caveat": "Per-leaf acceptance does not certify that the sum of weighted leaf errors is below threshold"}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps([{k: x[k] for k in ("component_cap", "depth_cap", "taylor_policy", "accepted_components", "evaluated_components", "threshold_satisfied", "sum_leaf_weighted_criterion", "ut_only_mse", "split_only_mse", "total_mse")} for x in rows], indent=2))


if __name__ == "__main__":
    main()

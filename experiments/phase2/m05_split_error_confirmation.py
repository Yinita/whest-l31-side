"""Exact 1D deep-ReLU split/UT error separation and 3D quadrature refinement."""

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
from m05_small_split_ut import adaptive, quadrature


def relu_normal(mu, variance):
    sigma = math.sqrt(variance)
    alpha = mu / sigma
    return sigma * math.exp(-alpha**2 / 2) / math.sqrt(2 * math.pi) + mu * (1 + math.erf(alpha / math.sqrt(2))) / 2


def main():
    output = Path("results/phase2/m05_split_error_confirmation_r05.json")
    assert not output.exists()
    rows = []
    # No bias: the full 1D ReLU network is a*x+ + b*(-x)+ exactly.
    weights = np.array([[[1.2]], [[.8]], [[1.1]]])
    slope = float(np.prod(weights))
    original_exact = slope / math.sqrt(2 * math.pi)
    for cap in (1, 27):
        for policy in ("upstream_mean", "paper_sum"):
            with flops.BudgetContext(flop_budget=2**35, quiet=True) as ctx:
                predicted, _, mixture, history, evaluated = adaptive([fnp.asarray(w) for w in weights], 1, .02, cap, 5, policy)
            actual = float(predicted.tolist()[0])
            split_exact = sum(mass * slope * relu_normal(float(mean.tolist()[0]), float(cov.tolist()[0][0]))
                              for mass, mean, cov, _, _ in mixture)
            error_ut, error_split = actual - split_exact, split_exact - original_exact
            total = actual - original_exact
            assert abs(total - error_ut - error_split) < 1e-14
            assert abs(total**2 - (error_ut**2 + error_split**2 + 2 * error_ut * error_split)) < 1e-14
            rows.append({"width": 1, "depth": 3, "component_cap": cap,
                         "policy": policy, "original_exact_mean": original_exact,
                         "split_exact_mean": split_exact, "predicted_ut_mean": actual,
                         "ut_only_bias": error_ut, "split_only_bias": error_split,
                         "total_bias": total, "ut_only_mse": error_ut**2,
                         "split_only_mse": error_split**2, "cross_term": 2 * error_ut * error_split,
                         "total_mse": total**2, "evaluated_components": evaluated,
                         "accepted_components": len(mixture),
                         "threshold_satisfied": all(not h["cap_or_depth_limited"] for h in history),
                         "history": history, "legal_cost": ctx.summary_dict()})
    source = Path("results/phase2/m05_small_split_ut_r04.json")
    previous = json.loads(source.read_text())
    quadrature_rows = []
    # Existing frozen component sets, no further splitting or parameter choice.
    for prior in previous["rows"]:
        if prior["width"] != 3 or prior["taylor_policy"] != "paper_sum":
            continue
        rng = np.random.default_rng(prior["weight_seed"])
        weights = rng.normal(size=(prior["depth"], 3, 3)) * math.sqrt(2 / 3)
        assert hashlib.sha256(weights.tobytes()).hexdigest() == prior["weights_sha256"]
        estimates = []
        for order in (32, 64, 96):
            original = quadrature(np.zeros(3), np.eye(3), weights, order)
            split = sum(m["mass"] * quadrature(np.array(m["mean"]), np.array(m["cov"]), weights, order)
                        for m in prior["mixture"])
            ut = np.array(prior["predicted_mean"])
            estimates.append({"order": order, "original_mean": original.tolist(),
                              "split_mean": split.tolist(), "ut_vs_split_mse": float(np.mean((ut - split)**2)),
                              "split_vs_original_mse": float(np.mean((split - original)**2)),
                              "total_mse": float(np.mean((ut - original)**2))})
        quadrature_rows.append({"weight_seed": prior["weight_seed"], "estimates": estimates,
                                "original_64to96_delta_mse": float(np.mean((np.array(estimates[-1]["original_mean"]) - estimates[-2]["original_mean"])**2)),
                                "split_64to96_delta_mse": float(np.mean((np.array(estimates[-1]["split_mean"]) - estimates[-2]["split_mean"])**2)),
                                "uncertainty": "Successive quadrature changes are convergence diagnostics, not certified bounds or benchmark truth"})
    report = {"version": "paper.M05.r05", "complete": True,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "method_source_sha256": hashlib.sha256(Path("experiments/phase2/m05_small_split_ut.py").read_bytes()).hexdigest(),
              "previous_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "exact_1d": rows, "quadrature_3d": quadrature_rows,
              "scope": "Analytic 1D error decomposition and frozen 3D quadrature diagnosis only; no official-score/reference replacement",
              "limitations": ["1D exact check does not prove high-dimensional feasibility",
                              "finite27component/5depth caps retained; no population convergence claim",
                              "no1024wide port or benchmark score"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{key: row[key] for key in ("policy", "component_cap", "accepted_components", "ut_only_bias", "split_only_bias", "total_mse", "threshold_satisfied")} for row in rows], indent=2))


if __name__ == "__main__":
    main()

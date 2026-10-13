"""Target-free GH2/4/8/16 error gate for the frozen r08 weight-only direction.

Use the candidate's actual metered direction function, then compute isolated
first-ReLU quadrature error in float64 against its exact Gaussian expectation.
This is a cheap integration diagnostic, not a full estimator score or resource
pass; no cached Phi, predictions, labels, or network references are read.
"""

# ruff: noqa: E402
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def gaussian_relu(mean, variance):
    sigma = np.sqrt(np.maximum(variance, 0.0))
    safe_sigma = np.maximum(sigma, 1e-300)
    alpha = mean / safe_sigma
    cdf = np.asarray([0.5*(1+math.erf(float(a)/np.sqrt(2))) for a in alpha.ravel()]).reshape(alpha.shape)
    value = sigma * np.exp(-alpha * alpha / 2) / np.sqrt(2 * np.pi) + mean * cdf
    return np.where(sigma > 0, value, np.maximum(mean, 0.0))


def integrate(first_weight, direction, count):
    nodes, weights = np.polynomial.hermite.hermgauss(count)
    nodes, weights = nodes * np.sqrt(2), weights / np.sqrt(np.pi)
    b = first_weight.T @ direction
    variance = np.sum(first_weight * first_weight, axis=0)
    conditional_variance = variance - b * b
    assert np.min(conditional_variance) > 0
    components = gaussian_relu(nodes[:, None] * b[None, :], conditional_variance[None, :])
    return weights @ components, nodes, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    assert not out.exists()
    start = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    candidate = root / "estimators/tenfold_conditioning4node_r08/estimator.py"
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    contract = json.loads((root / "results/phase2/tenfold_conditioning4node_contract_r08.json").read_text())
    assert candidate_sha == contract["candidate_sha256"]
    spec = importlib.util.spec_from_file_location("conditioning_frozen_r08", candidate)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    manifest_path = root / "benchmark/data/rpc_inputs/sprint_public_v37b_h18_r04_rpc8/inputs.json"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for entry in manifest:
        path = Path(entry["weights"])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == entry["sha256"]
        with np.load(path) as source:
            assert source.files == ["weights"]
            weights = source["weights"]
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            api_weights = [fnp.asarray(w) for w in weights]
            direction = np.asarray(mod.conditioning_direction(api_weights, weights.shape[1])).astype(np.float64)
        first = weights[0].astype(np.float64)
        variance = np.sum(first * first, axis=0)
        exact = np.sqrt(variance / (2 * np.pi))
        q = first.T @ direction
        projected_fraction = q * q / variance
        estimates, node_metadata, score = {}, {}, {}
        for count in (2, 4, 8, 16):
            value, nodes, weights_gh = integrate(first, direction, count)
            estimates[count] = value
            node_metadata[str(count)] = {"nodes": nodes.tolist(), "weights": weights_gh.tolist()}
            score[str(count)] = {"mse_vs_exact": float(np.mean((value-exact)**2)),
                                 "maxabs_vs_exact": float(np.max(np.abs(value-exact))),
                                 "mean_error_vs_exact": float(np.mean(value-exact)),
                                 "mse_vs_GH16": None if count == 16 else 0.0}
        for count in (2, 4, 8):
            score[str(count)]["mse_vs_GH16"] = float(np.mean((estimates[count]-estimates[16])**2))
        row = {"network_id": entry["index"], "name": entry["name"], "seed": entry["seed"],
               "weights_sha256": sha, "direction_norm": float(np.linalg.norm(direction)),
               "first_output_projected_fraction_mean": float(np.mean(projected_fraction)),
               "first_output_projected_fraction_max": float(np.max(projected_fraction)),
               "quadrature": score, "direction_only_meter": ctx.summary_dict()}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "direction_only_meter"}), flush=True)
    report = {"version": "tenfold-conditioning-firstrelu-gate-r01", "complete": True,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "candidate_sha256": candidate_sha,
              "weights_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "tool_versions": {"flopscope": flops.__version__, "numpy": np.__version__},
              "rows": rows, "node_metadata": node_metadata,
              "average_first_relu_quadrature_mse": {str(n): float(np.mean([r["quadrature"][str(n)]["mse_vs_exact"] for r in rows]))
                                                     for n in (2, 4, 8, 16)},
              "elapsed_s": time.monotonic()-start,
              "scope": "Isolated float64 firstReLU integration error against analytic unconditional Gaussian mean; actual r08 float32 weight-only direction",
              "before": "Exact Gaussian first-layer mean sqrt(sum(w0**2)/(2*pi))",
              "after": "Fixed GH2/4/8/16 averages of analytic conditional first-ReLU means",
              "no_adjusted_score": True,
              "limitations": ["No full conditional K3 propagation or candidate raw MSE measured",
                              "Float64 isolated quadrature does not include candidate float32 Gram/mean rounding",
                              "Small firstlayer quadrature error does not bound deep-network quadrature bias",
                              "Direction-only metering is not fullpredict/residual resource acceptance",
                              "Float32 direction norm rounding is reported; no normalization change made"]}
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("version", "average_first_relu_quadrature_mse", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()

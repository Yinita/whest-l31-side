"""Bounded contracts/cost extrapolation for mixedfloat fullcov GH8 response."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def independent_response(weights, direction, node):
    weight64 = [weight.astype(np.float64) for weight in weights]
    q32 = weights[0].T@direction
    firstcov = weights[0].T@weights[0]-q32[:, None]*q32[None, :]
    q64 = weight64[0].T@direction.astype(np.float64)
    firstvar = np.maximum(np.sum(weight64[0]**2, axis=0)-q64*q64, 1e-12)
    mean = node*direction.astype(np.float64)
    rows, covariance = [], None
    for layer, weight in enumerate(weights):
        mu = node*q64 if layer == 0 else weight64[layer].T@mean
        if layer == 0:
            pre, var = firstcov, firstvar
        else:
            pre = weight.T@covariance@weight
            pre = (pre+pre.T)/2
            var = np.maximum(np.diag(pre).astype(np.float64), 1e-12)
        sigma = np.sqrt(var)
        alpha = mu/sigma
        phi = np.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
        gate = np.array([.5*(1+math.erf(float(a)/math.sqrt(2))) for a in alpha])
        mean = sigma*phi+mu*gate
        marginalvar = np.maximum((mu*mu+var)*gate+mu*sigma*phi-mean*mean, 1e-12)
        covariance = pre*gate.astype(np.float32)[:, None]*gate.astype(np.float32)[None, :]
        np.fill_diagonal(covariance, marginalvar.astype(np.float32))
        rows.append(mean)
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    path = Path("estimators/tenfold_conditioning_cvfull_r11/estimator.py")
    spec = importlib.util.spec_from_file_location("cvfullcontract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(674912)
    rows = []
    for width in (8, 32, 64):
        weights = [(rng.normal(size=(width, width))/math.sqrt(width)).astype(np.float32)
                   for _ in range(16)]
        direction = rng.normal(size=width)
        direction = (direction/np.linalg.norm(direction)).astype(np.float32)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            got = module.cheap_conditional_fullcov_correction(list(map(fnp.asarray, weights)), fnp.asarray(direction))
        expected = sum(weight*independent_response(weights, direction, node)
                       for node, weight in zip(module.GH8_NODES, module.GH8_WEIGHTS))
        expected -= independent_response(weights, direction, 0.)
        error = float(np.max(np.abs(np.asarray(got).astype(np.float64)-expected)))
        relative = float(np.linalg.norm(np.asarray(got)-expected)/max(np.linalg.norm(expected), 1e-30))
        assert error < 2e-6 and relative < 2e-4, (width, error, relative)
        cost = context.summary_dict()
        matmul_cost = cost["operations"].get("matmul", {}).get("flop_cost", 0)
        projected = matmul_cost*(1024/width)**3+(cost["flops_used"]-matmul_cost)*(1024/width)**2
        rows.append({"width": width, "depth": 16, "independent_mixedfloat_max_abs_error": error,
                     "relative_output_error": relative, "finite": bool(np.all(np.isfinite(np.asarray(got)))),
                     "matmul_flops_used": matmul_cost,
                     "projected1024_flops_upper_from_cubic_matmul_quadratic_rest": projected,
                     "projected1024_fraction_budget": projected/2**41,
                     "legal_cost": cost})
    report = {"version": "tenfold.cvfull.contract.r11", "complete": True, "rows": rows,
              "seed": 674912, "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cheap_node_count": 9, "single_change": "r10 cheap diagonalpostcov→fullcov firstchaos response with covariancefloat32, vectorsfloat64; parent/direction/fixedGH8 unchanged",
              "resource_status": "prepared_only", "no_adjusted_score": True,
              "not_resource_acceptance": True,
              "limitations": ["Fullcov covariancefirstchaos is not exactbivariateGaussian covariance", "Projectionis FLOP estimate, not1024 actualwall/resourceacceptance", "Float32cov recurrence introduceschangedroundoff fromdiagvectors", "120secondwallmayfail despitebudgetfit"]}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({"output": str(args.output), "candidate_sha256": report["candidate_sha256"], "rows": [{k: v for k, v in row.items() if k != "legal_cost"} for row in rows]}, flush=True)


if __name__ == "__main__":
    main()

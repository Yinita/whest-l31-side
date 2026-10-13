"""Bounded reproducible contract for oneparent plus GH8 diagonal correction."""

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
from types import SimpleNamespace

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext


def independent_diagonal_response(weights, u, node):
    mean, variance = node*u, np.ones(len(u))
    rows = []
    for layer, weight in enumerate(weights):
        weight = weight.astype(np.float64)
        linear_mean = weight.T@mean
        if layer == 0:
            covariance = weight.T@(np.eye(len(u))-np.outer(u, u))@weight
            linear_var = np.maximum(np.diag(covariance), 1e-12)
        else:
            linear_var = np.maximum((weight*weight).T@variance, 1e-12)
        sigma = np.sqrt(linear_var)
        alpha = linear_mean/sigma
        phi = np.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
        gate = np.array([.5*(1+math.erf(float(x)/math.sqrt(2))) for x in alpha])
        mean = sigma*phi+linear_mean*gate
        variance = np.maximum((linear_mean*linear_mean+linear_var)*gate
                              +linear_mean*sigma*phi-mean*mean, 1e-12)
        rows.append(mean)
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    path = Path("estimators/tenfold_conditioning_cvdiag_r10/estimator.py")
    spec = importlib.util.spec_from_file_location("cvdiagbounded", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(674911)
    rows = []
    for width in (4, 8):
        weights = [rng.normal(size=(width, width)).astype(np.float32)/np.sqrt(width)
                   for _ in range(3)]
        u = rng.normal(size=width)
        u = (u/np.linalg.norm(u)).astype(np.float32)
        expected = sum(weight*independent_diagonal_response(weights, u.astype(np.float64), node)
                       for node, weight in zip(module.GH8_NODES, module.GH8_WEIGHTS))
        expected -= independent_diagonal_response(weights, u.astype(np.float64), 0.)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            fw = list(map(fnp.asarray, weights))
            actual = module.cheap_conditional_diagonal_correction(fw, fnp.asarray(u))
            zero = module.cheap_conditional_diagonal_correction(fw, fnp.zeros(width, dtype=fnp.float32))
        error = float(np.max(np.abs(np.asarray(actual).astype(np.float64)-expected)))
        assert error < 2e-7 and np.max(np.abs(np.asarray(zero))) < 1e-7
        # Independent exact unconditional first-ReLU mean supplies a quadrature
        # truncation diagnostic; do not assert GH8 is exact on ReLU responses.
        sd = np.sqrt(np.sum(weights[0].astype(np.float64)**2, axis=0))
        true_first = sd/math.sqrt(2*math.pi)
        conditional0 = independent_diagonal_response(weights, u.astype(np.float64), 0.)[0]
        first_gh_error = float(np.max(np.abs(conditional0+np.asarray(actual)[0]-true_first)))
        estimator = module.Estimator()
        network = SimpleNamespace(width=width, weights=fw)
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            estimator.setup(SetupContext(width=width, depth=3, flop_budget=2**41,
                                         api_version="0.1", seed=0))
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            parent = estimator._predict_one_condition(network, 2**41)
            saved = np.asarray(parent).copy()
            zero_parent = estimator._predict_one_condition(network, 2**41,
                                                           fnp.zeros(width, dtype=fnp.float32), 0.)
        # Repeated source pooling can reorder floating point paths slightly;
        # zeros preserve the same input distribution, tolerance records this.
        zero_parent_error = float(np.max(np.abs(np.asarray(parent)-np.asarray(zero_parent))))
        alias_error = float(np.max(np.abs(np.asarray(parent)-saved)))
        assert zero_parent_error < 1e-6 and alias_error == 0
        assert np.all(np.isfinite(np.asarray(actual)))
        rows.append({"width": width, "correction_independent_skeleton_max_abs_error": error,
                     "zero_direction_correction_max_abs": float(np.max(np.abs(np.asarray(zero)))),
                     "zero_direction_parent_max_abs_error": zero_parent_error,
                     "parent_return_pool_alias_error": alias_error,
                     "first_layer_GH8_vs_exact_unconditional_mean_max_abs_error": first_gh_error,
                     "legal_correction_cost": context.summary_dict()})
    report = {"version": "tenfold.conditioning.cvdiag.contract.r10", "complete": True,
              "rows": rows, "seed": 674911,
              "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Bounded conditionalmean/cov marginal skeleton, zero-u reduction, oneparent output ownership; GH8 bias diagnostic",
              "no_adjusted_score": True, "not_resource_acceptance": True,
              "limitations": ["GH8 mean response not exact for deepReLU", "diagcov skeleton response may be wrong despite exact node formulas", "parentconditionalbias and realresource not evaluated"]}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({"output": str(args.output), "candidate_sha256": report["candidate_sha256"], "rows": [{k: v for k, v in r.items() if k != "legal_correction_cost"} for r in rows]}, flush=True)


if __name__ == "__main__":
    main()

"""Reproducible bounded contracts for frozen rank-one conditioning candidates.

No benchmark references, actual network scores, heavy predict or candidate edits.
Independent NumPy/scalar integrals are diagnostics outside legal BudgetContext.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext

VERSIONS = {
    "r08": ("estimators/tenfold_conditioning4node_r08/estimator.py", "GH4", 7),
    "r09": ("estimators/tenfold_conditioning2node_r09/estimator.py", "GH2", 3),
}


def scalar_cdf(values):
    return np.array([.5*(1+math.erf(float(x)/math.sqrt(2))) for x in values])


def independent_direction(weights):
    """Explicit forward expected-gate Jacobian product, unlike backward helper."""
    width = weights[0].shape[0]
    mean, variance, jacobian = np.zeros(width), np.ones(width), np.eye(width)
    for raw in weights:
        weight = raw.astype(np.float64)
        linear_mean = weight.T@mean
        linear_var = np.maximum((weight*weight).T@variance, 1e-12)
        sigma = np.sqrt(linear_var)
        alpha = linear_mean/sigma
        phi = np.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
        gate = scalar_cdf(alpha)
        mean = sigma*phi+linear_mean*gate
        variance = np.maximum((linear_mean*linear_mean+linear_var)*gate
                              +linear_mean*sigma*phi-mean*mean, 1e-12)
        jacobian = (weight.T@jacobian)*gate[:, None]
    gradient = jacobian.sum(axis=0)
    return gradient/np.linalg.norm(gradient)


def relu_integral_contract(nodes):
    """Gaussian residual integral per condition; no reused flopscope Wicks."""
    maximum = 0.
    for node in nodes:
        for loading, sigma in ((.25, 1.), (-.3, .7)):
            mu = float(node)*loading
            alpha = mu/sigma
            analytic = sigma*math.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
            analytic += mu*.5*(1+math.erf(alpha/math.sqrt(2)))
            lower = -mu/sigma
            residual = np.linspace(lower, lower+14, 200001)
            values = (mu+sigma*residual)*np.exp(-residual*residual/2)/math.sqrt(2*math.pi)
            integral = float(np.trapezoid(values, residual))
            maximum = max(maximum, abs(analytic-integral))
    assert maximum < 2e-9
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=VERSIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve original evidence"
    candidate, prefix, exact_degree = VERSIONS[args.version]
    path = Path(candidate)
    spec = importlib.util.spec_from_file_location("bounded_conditioning_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nodes, weights = getattr(module, prefix+"_NODES"), getattr(module, prefix+"_WEIGHTS")
    seed = 674808
    rng = np.random.default_rng(seed)
    rows = []
    for width in (4, 8, 32):
        network = [rng.normal(size=(width, width)).astype(np.float32)/np.sqrt(width)
                   for _ in range(3)]
        expected_direction = independent_direction(network)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            legal_weights = list(map(fnp.asarray, network))
            direction = module.conditioning_direction(legal_weights, width)
            covariance = module.conditioned_first_covariance(legal_weights[0], direction)
            projected = legal_weights[0].T@direction
            means = [float(node)*projected for node in nodes]
            mixed_mean = sum(float(weight)*mu for weight, mu in zip(weights, means))
            mixed_second = covariance+sum(float(weight)*mu[:, None]*mu[None, :]
                                          for weight, mu in zip(weights, means))
        actual_direction = np.asarray(direction).astype(np.float64)
        input_cov = np.eye(width)-np.outer(actual_direction, actual_direction)
        direct = network[0].astype(np.float64).T@input_cov@network[0].astype(np.float64)
        gram = network[0].astype(np.float64).T@network[0].astype(np.float64)
        direction_error = float(np.max(np.abs(actual_direction-expected_direction)))
        covariance_error = float(np.max(np.abs(np.asarray(covariance)-direct)))
        moment_error = float(np.max(np.abs(np.asarray(mixed_second)-gram)))
        assert direction_error < 1e-7
        assert covariance_error < 3e-7 and moment_error < 3e-7
        norm_squared = float(actual_direction@actual_direction)
        # This is the exact rank-one eigenvalue in float64 diagnostic arithmetic.
        # Document potential slightly negative roundoff instead of silently
        # shifting covariance or claiming an exact PSD float32 matrix.
        minimum_eigenvalue = float(min(1., 1.-norm_squared))
        assert minimum_eigenvalue >= -2e-7
        covariance64 = np.asarray(covariance).astype(np.float64)
        rows.append({"width": width,
                     "direction_vs_independent_chain_jacobian_max_error": direction_error,
                     "direction_squared_norm": norm_squared,
                     "input_covariance_min_eigenvalue": minimum_eigenvalue,
                     "float32_input_PSD_tolerance": 2e-7,
                     "first_conditional_covariance_max_error": covariance_error,
                     "first_conditional_covariance_min_eigenvalue": float(np.linalg.eigvalsh((covariance64+covariance64.T)/2).min()),
                     "recombined_mean_max_abs": float(np.max(np.abs(np.asarray(mixed_mean)))),
                     "recombined_covariance_max_error": moment_error,
                     "legal_cost": context.summary_dict()})
    moment_values = [float(sum(float(w)*float(t)**degree for t, w in zip(nodes, weights)))
                     for degree in range(exact_degree+1)]
    gaussian = [1., 0., 1., 0., 3., 0., 15., 0.]
    assert max(abs(a-b) for a, b in zip(moment_values, gaussian)) < 1e-12
    integral_error = relu_integral_contract(nodes)
    estimator = module.Estimator()
    shape = SimpleNamespace(width=32, weights=list(map(fnp.asarray, network)))
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        estimator.setup(SetupContext(width=32, depth=3, flop_budget=2**41,
                                     api_version="0.1", seed=0))
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as generic_context:
        generic = estimator.predict(shape, 2**41)
    assert generic.shape == (3, 32) and np.all(np.isfinite(np.asarray(generic)))
    report = {"version": f"tenfold.conditioning.contract.{args.version}.r10", "complete": True,
              "rows": rows, "seed": seed, "fixed_nodes": nodes, "fixed_weights": weights,
              "normal_moments_degree0toN": moment_values,
              "conditional_relu_vs_independent_scalar_integral_max_error": integral_error,
              "generic_parent_shape": [3, 32], "generic_parent_finite": True,
              "generic_parent_legal_cost": generic_context.summary_dict(),
              "candidate": candidate, "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "flopscope", "whestbench")},
              "scope": "Target-free bounded direction/firstmean/firstcovariance/normalquadrature/integral/generic contracts; no heavy1024predict",
              "limitations": ["Frozen expected gates define a skeleton Jacobian, not derivative of gate dependence", "Input covariance is PSD up to measured float32 direction norm roundoff", "Moment exactness does not imply exact deep ReLU mean", "Empirical parent riders/source/metric remain approximations under conditioning"],
              "no_adjusted_score": True, "not_resource_acceptance": True}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({"output": str(args.output), "candidate_sha256": report["candidate_sha256"],
           "max_direction_error": max(row["direction_vs_independent_chain_jacobian_max_error"] for row in rows),
           "min_input_covariance_eigenvalue": min(row["input_covariance_min_eigenvalue"] for row in rows),
           "max_recombined_covariance_error": max(row["recombined_covariance_max_error"] for row in rows)}, flush=True)


if __name__ == "__main__":
    main()

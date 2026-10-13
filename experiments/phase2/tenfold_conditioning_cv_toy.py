"""Target-free toy gate for one expensive zero-condition plus cheap GH correction.

Full Gaussian-quadrature toy forwards are offline diagnostics, never production.
The cheap skeleton uses only billed operations; full-cov mode retains first chaos
and exact Gaussian marginal variances rather than bivariate ReLU moments.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def cheap_skeleton(weights, direction, node, fullcov=False):
    width = len(direction)
    mean = direction*node
    covariance = fnp.eye(width, dtype=fnp.float64)-direction[:, None]*direction[None, :]
    variance = fnp.diag(covariance)
    rows = []
    for layer, weight in enumerate(weights):
        mu = weight.T@mean
        if fullcov or layer == 0:
            pre = weight.T@covariance@weight
            var = fnp.maximum(fnp.diag(pre), 1e-12)
        else:
            var = fnp.maximum((weight*weight).T@variance, 1e-12)
        sigma = fnp.sqrt(var)
        alpha = mu/sigma
        phi = flops.stats.norm.pdf(alpha).astype(fnp.float64)
        gate = flops.stats.norm.cdf(alpha).astype(fnp.float64)
        mean = sigma*phi+mu*gate
        second = (mu*mu+var)*gate+mu*sigma*phi
        variance = fnp.maximum(second-mean*mean, 1e-12)
        if fullcov:
            covariance = pre*gate[:, None]*gate[None, :]
            # Extra Gaussian higher-chaos marginal variance is nonnegative.
            correction = fnp.maximum(variance-gate*gate*var, 0.)
            covariance = covariance+fnp.diagflat(correction)
        rows.append(mean)
    return fnp.stack(rows)


def grid(order, dim):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes, weights = nodes*np.sqrt(2), weights/np.sqrt(np.pi)
    indices = np.array(list(itertools.product(range(order), repeat=dim)))
    return nodes[indices], np.prod(weights[indices], axis=1)


def true_means(weights, samples, probabilities):
    rows = []
    for weight in weights:
        samples = np.maximum(samples@weight, 0)
        rows.append(probabilities@samples)
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    path = Path("estimators/tenfold_conditioning2node_r09/estimator.py")
    spec = importlib.util.spec_from_file_location("cvtoy_direction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(674910)
    rows = []
    for network_id in range(3):
        width, depth = 3, 4
        weights = [rng.normal(size=(width, width))/np.sqrt(width) for _ in range(depth)]
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            legal_weights = list(map(fnp.asarray, weights))
            u = module.conditioning_direction(legal_weights, width).astype(fnp.float64)
        direction = np.asarray(u).copy()
        direction /= np.linalg.norm(direction)
        # Orthogonal residual basis from known direction; offline diagnostic only.
        q, _ = np.linalg.qr(np.column_stack((direction, np.eye(width))))
        residual_basis = q[:, 1:]
        x, probs = grid(40, width)
        truth = true_means(weights, x, probs)
        x2, probs2 = grid(56, width)
        truth2 = true_means(weights, x2, probs2)
        residual, residual_prob = grid(56, width-1)
        conditional_zero = true_means(weights, residual@residual_basis.T, residual_prob)
        gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(8)
        gh_nodes, gh_weights = gh_nodes*np.sqrt(2), gh_weights/np.sqrt(np.pi)
        methods = []
        for mode in ("diag", "fullcov_firstchaos"):
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
                lu = fnp.asarray(direction)
                zero = cheap_skeleton(legal_weights, lu, 0., fullcov=mode != "diag")
                integrated = fnp.zeros((depth, width), dtype=fnp.float64)
                for node, weight in zip(gh_nodes, gh_weights):
                    integrated = integrated+float(weight)*cheap_skeleton(legal_weights, lu, float(node), fullcov=mode != "diag")
                correction = integrated-zero
            estimate = conditional_zero+np.asarray(correction)
            reference_delta = truth2-conditional_zero
            methods.append({"method": mode,
                            "perlayer_correction_MSE": np.mean((np.asarray(correction)-reference_delta)**2, axis=1).tolist(),
                            "perlayer_no_correction_MSE": np.mean(reference_delta**2, axis=1).tolist(),
                            "perlayer_corrected_estimate_MSE": np.mean((estimate-truth2)**2, axis=1).tolist(),
                            "legal_cost": context.summary_dict()})
        rows.append({"toy_network_id": network_id, "width": width, "depth": depth,
                     "unconditional_GH40_vs_GH56_perlayer_MSE": np.mean((truth-truth2)**2, axis=1).tolist(),
                     "methods": methods})
    report = {"version": "tenfold.conditioning.cv.toy.r01", "complete": True, "seed": 674910,
              "rows": rows, "cheap_nodes": 8,
              "formula": "expensive_conditional_parent(t=0)+GH8_t cheap_skeleton(t)-cheap_skeleton(0)",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "direction_candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "scope": "Known-weight n3depth4 toy expectation integration; conditional_zero replaced by high-order offline toy integral to isolate correction approximation",
              "limitations": ["Not exact distribution identity for approximate cheap skeleton", "Fullcov_firstchaos mode is not exact bivariateGaussianReLUcovariance", "Deterministic toy GH40/GH56 convergence noise recorded", "Actual parentconditionalbias/riders/resource not evaluated"],
              "no_adjusted_score": True, "not_resource_acceptance": True}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({"output": str(args.output), "rows": [{"id": r["toy_network_id"], "methods": [{k: v for k, v in m.items() if k != "legal_cost"} for m in r["methods"]]} for r in rows]}, flush=True)


if __name__ == "__main__":
    main()

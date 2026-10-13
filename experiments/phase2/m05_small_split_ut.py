"""Lawful small-object adaptive input-GMM / UT / Gaussian-W2 reproduction.

This finite, capped reproduction does not certify the paper's population bound:
the Taylor expectation and output moments use UT; input splitting changes the
distribution. Original-code and paper-sum Taylor criteria are reported separately.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

SPLIT_WEIGHTS = (.6364, .1818, .1818)
SPLIT_OFFSETS = (0.0, 1.0579, -1.0579)
SPLIT_SIGMA = .7687


def psd_sqrt(cov):
    eigenvalues, eigenvectors = fnp.linalg.eigh((cov + cov.T) * .5)
    return (eigenvectors * fnp.sqrt(fnp.maximum(eigenvalues, 0))[None, :]) @ eigenvectors.T


def gaussian_w2(mean_a, cov_a, mean_b, cov_b):
    root = psd_sqrt(cov_b)
    middle = psd_sqrt(root @ cov_a @ root)
    trace = fnp.sum(fnp.diag(cov_a + cov_b - 2 * middle))
    return fnp.sqrt(fnp.maximum(fnp.sum((mean_a - mean_b)**2) + trace, 0))


def propagate(points, mean, weights):
    values = points
    center = mean
    jacobian = fnp.eye(mean.shape[0], dtype=fnp.float64)
    for weight in weights:
        pre = center @ weight
        gate = (pre > 0).astype(fnp.float64)
        jacobian = (jacobian @ weight) * gate[None, :]
        center = fnp.maximum(pre, 0)
        values = fnp.maximum(values @ weight, 0)
    linear = (points - mean) @ jacobian + center
    return values, linear, center, jacobian


def weighted_moments(values, point_weights):
    mean = fnp.sum(point_weights[:, None] * values, axis=0)
    centered = values - mean
    cov = centered.T @ (point_weights[:, None] * centered)
    return mean, cov


def component(mean, cov, weights, taylor_policy):
    n = mean.shape[0]
    kappa = 2 if n == 1 else 1 if n == 2 else n - 3
    eigenvalues, eigenvectors = fnp.linalg.eigh((cov + cov.T) * .5)
    # Column eigenvector sigma points; retain all input dimensions.
    root = eigenvectors * fnp.sqrt(fnp.maximum(eigenvalues, 0))[None, :]
    offset = root.T * math.sqrt(n + kappa)
    points = fnp.concatenate((mean[None, :], mean[None, :] + offset,
                              mean[None, :] - offset), axis=0)
    point_weights = fnp.asarray([kappa / (n + kappa)] + [1 / (2 * (n + kappa))] * (2 * n), dtype=fnp.float64)
    values, linear, _, _ = propagate(points, mean, weights)
    out_mean, out_cov = weighted_moments(values, point_weights)
    linear_mean, linear_cov = weighted_moments(linear, point_weights)
    discrepancy = fnp.sum(point_weights * fnp.sum((values - linear)**2, axis=1))
    if taylor_policy == "upstream_mean":
        discrepancy = discrepancy / (2 * n + 1)
    taylor = fnp.sqrt(fnp.maximum(discrepancy, 0))
    criterion = gaussian_w2(out_mean, out_cov, linear_mean, linear_cov) + taylor
    # Decisions/materialization are included in the real budget, not unmetered kernels.
    return out_mean, out_cov, float(criterion.tolist()), eigenvalues, eigenvectors


def adaptive(weights, n, threshold, max_components, max_depth, taylor_policy):
    pending = [(1.0, fnp.zeros(n, dtype=fnp.float64), fnp.eye(n, dtype=fnp.float64), 0)]
    accepted, history = [], []
    evaluated = 0
    while pending:
        mass, mean, cov, depth = pending.pop(0)
        out_mean, out_cov, criterion, eig, vec = component(mean, cov, weights, taylor_policy)
        evaluated += 1
        split = mass * criterion > threshold and depth < max_depth and len(pending) + len(accepted) + 3 <= max_components
        history.append({"mass": mass, "depth": depth, "criterion": criterion,
                        "weighted_criterion": mass * criterion, "split": split,
                        "cap_or_depth_limited": mass * criterion > threshold and not split})
        if split:
            # eigh ascending order, so the last column is a maximal-variance axis.
            direction = vec[:, -1]
            largest = eig[-1]
            new_cov = cov - (1 - SPLIT_SIGMA**2) * largest * (direction[:, None] * direction[None, :])
            for probability, offset in zip(SPLIT_WEIGHTS, SPLIT_OFFSETS):
                child_mean = mean + offset * fnp.sqrt(fnp.maximum(largest, 0)) * direction
                pending.append((mass * probability, child_mean, new_cov, depth + 1))
        else:
            accepted.append((mass, mean, cov, out_mean, out_cov))
    masses = fnp.asarray([a[0] for a in accepted], dtype=fnp.float64)
    means = fnp.stack([a[3] for a in accepted])
    final_mean = fnp.sum(masses[:, None] * means, axis=0)
    covs = fnp.stack([a[4] for a in accepted])
    centered = means - final_mean
    final_cov = fnp.sum(masses[:, None, None] * (covs + centered[:, :, None] * centered[:, None, :]), axis=0)
    return final_mean, final_cov, accepted, history, evaluated


def exact_relu_mean(mean, cov, weight):
    pre = mean @ weight
    variance = np.maximum(np.einsum("ij,ia,ja->a", cov, weight, weight), 1e-30)
    sigma = np.sqrt(variance)
    alpha = pre / sigma
    cdf = (1 + np.vectorize(math.erf)(alpha / math.sqrt(2))) / 2
    return sigma * np.exp(-alpha**2 / 2) / math.sqrt(2 * math.pi) + pre * cdf


def quadrature(mean, cov, weights, order):
    nodes, probabilities = np.polynomial.hermite.hermgauss(order)
    grid = np.asarray(np.meshgrid(*([nodes] * len(mean)), indexing="ij")).reshape(len(mean), -1).T
    masses = np.asarray(np.meshgrid(*([probabilities] * len(mean)), indexing="ij"))
    masses = np.prod(masses, axis=0).ravel() / math.pi**(len(mean) / 2)
    eig, vec = np.linalg.eigh(cov)
    root = vec * np.sqrt(np.maximum(eig, 0))[None, :]
    values = mean + math.sqrt(2) * grid @ root.T
    for weight in weights:
        values = np.maximum(values @ weight, 0)
    return masses @ values


def contracts():
    rng = np.random.default_rng(61013030)
    rows = []
    for n in (3, 5):
        mean = rng.normal(size=n)
        a = rng.normal(size=(n, n))
        cov = a @ a.T + np.eye(n) * .2
        w = rng.normal(size=(n, n))
        kappa = n - 3
        with flops.BudgetContext(flop_budget=2**35, quiet=True) as ctx:
            m, c, fw = fnp.asarray(mean), fnp.asarray(cov), fnp.asarray(w)
            eig, vec = fnp.linalg.eigh(c)
            root = vec * fnp.sqrt(eig)[None, :]
            off = root.T * math.sqrt(n + kappa)
            points = fnp.concatenate((m[None, :], m[None, :] + off, m[None, :] - off))
            mass = fnp.asarray([kappa / (n + kappa)] + [1 / (2 * (n + kappa))] * (2 * n))
            out_mean, out_cov = weighted_moments(points @ fw, mass)
            distance = gaussian_w2(fnp.zeros(n), fnp.eye(n), fnp.ones(n), 4 * fnp.eye(n))
        mean_error = float(abs(np.asarray(out_mean) - mean @ w).max())
        cov_error = float(abs(np.asarray(out_cov) - w.T @ cov @ w).max())
        w2_error = abs(float(distance.tolist()) - math.sqrt(2 * n))
        assert max(mean_error, cov_error, w2_error) < 1e-10
        # Independent finite differences of the full ReLU MLP away from kinks.
        ws = rng.normal(size=(3, n, n)) / math.sqrt(n)
        with flops.BudgetContext(flop_budget=2**35, quiet=True):
            _, _, center, jac = propagate(fnp.asarray(mean[None, :]), fnp.asarray(mean), [fnp.asarray(w) for w in ws])
        def forward(x):
            for w in ws:
                x = np.maximum(x @ w, 0)
            return x
        finite = np.stack([(forward(mean + 1e-6 * np.eye(n)[j]) - forward(mean - 1e-6 * np.eye(n)[j])) / 2e-6 for j in range(n)])
        jac_error = float(abs(np.asarray(jac) - finite).max())
        assert jac_error < 1e-8 and np.allclose(np.asarray(center), forward(mean))
        rows.append({"width": n, "linear_mean_max_error": mean_error,
                     "linear_cov_max_error": cov_error, "gaussian_w2_closed_form_error": w2_error,
                     "jacobian_finite_difference_error": jac_error, "legal_contract_flops": ctx.summary_dict()["flops_used"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    rows = []
    contract_rows = contracts()
    for n, depth, seed in ((3, 3, 61013031), (3, 3, 61013032), (5, 2, 61013033)):
        rng = np.random.default_rng(seed)
        weights = rng.normal(size=(depth, n, n)) * math.sqrt(2 / n)
        exact_first = exact_relu_mean(np.zeros(n), np.eye(n), weights[0])
        for policy in ("upstream_mean", "paper_sum"):
            started = time.monotonic()
            with flops.BudgetContext(flop_budget=2**35, quiet=True) as ctx:
                mean, cov, mixture, history, evaluated = adaptive([fnp.asarray(w) for w in weights], n, .02, 27, 5, policy)
            summary = ctx.summary_dict()
            materialized = [(mass, np.asarray(m), np.asarray(c)) for mass, m, c, _, _ in mixture]
            mixture_first = sum(mass * exact_relu_mean(m, c, weights[0]) for mass, m, c in materialized)
            input_mean = sum(mass * m for mass, m, _ in materialized)
            input_second = sum(mass * (c + np.outer(m, m)) for mass, m, c in materialized)
            row = {"width": n, "depth": depth, "weight_seed": seed, "taylor_policy": policy,
                   "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
                   "threshold": .02, "max_components": 27, "max_depth": 5,
                   "accepted_components": len(mixture), "evaluated_components": evaluated,
                   "weighted_criterion_final": sum(h["weighted_criterion"] for h in history if not h["split"]),
                   "threshold_satisfied": all(not h["cap_or_depth_limited"] for h in history),
                   "history": history, "legal_cost": summary, "elapsed_s": time.monotonic() - started,
                   "predicted_mean": np.asarray(mean).tolist(), "predicted_cov": np.asarray(cov).tolist(),
                   "input_mixture_mass": sum(a[0] for a in materialized),
                   "input_mean_max_abs": float(abs(input_mean).max()),
                   "input_covariance_max_error_vs_identity": float(abs(input_second - np.outer(input_mean, input_mean) - np.eye(n)).max()),
                   "first_layer_split_only_mse_exact_gaussian_moments": float(np.mean((mixture_first - exact_first)**2)),
                   "mixture": [{"mass": mass, "mean": m.tolist(), "cov": c.tolist()} for mass, m, c in materialized]}
            assert abs(row["input_mixture_mass"] - 1) < 1e-12
            if n == 3:
                diagnostics = []
                for order in (12, 20, 32):
                    original = quadrature(np.zeros(n), np.eye(n), weights, order)
                    split = sum(mass * quadrature(m, c, weights, order) for mass, m, c in materialized)
                    ut_error = np.asarray(mean) - split
                    split_error = split - original
                    diagnostics.append({"order": order, "original_mean": original.tolist(), "split_mixture_mean": split.tolist(),
                                        "ut_vs_split_mse": float(np.mean(ut_error**2)),
                                        "split_vs_original_mse": float(np.mean(split_error**2)),
                                        "cross_term": float(2 * np.mean(ut_error * split_error)),
                                        "total_ut_vs_original_mse": float(np.mean((np.asarray(mean) - original)**2))})
                row["quadrature_diagnostic"] = diagnostics
                row["original_quadrature_20to32_delta_mse"] = float(np.mean((np.asarray(diagnostics[-1]["original_mean"]) - diagnostics[-2]["original_mean"])**2))
                row["split_quadrature_20to32_delta_mse"] = float(np.mean((np.asarray(diagnostics[-1]["split_mixture_mean"]) - diagnostics[-2]["split_mixture_mean"])**2))
            rows.append(row)
            print(n, seed, policy, "components", len(mixture), "FLOPs", summary["flops_used"], flush=True)
    report = {"version": "paper.M05.r03", "complete": True, "contracts": contract_rows, "rows": rows,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "upstream_commit": "915eaacea0ae3906536b8ffc6662a84b43b8487c",
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "scope": "Small full adaptive input-GMM/UT/GaussianW2 pipeline using billed allowed operators; no competition candidate or benchmark score",
              "limitations": ["finite component/depth caps explicitly reported; no convergence claim",
                              "upstream TensorFlow runtime not independently reproduced",
                              "UT Taylor/GaussianW2 criterion is a heuristic population-bound estimate",
                              "deterministic quadrature has nonsmooth ReLU convergence error; diagnostics not official truth",
                              "3/5widths do not establish1024wide accuracy/resourcefeasibility",
                              "no input subspace truncation and no merge; matrix square roots use unclipped PSDeigenvalues unlike upstream1e-5floor"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

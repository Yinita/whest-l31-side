"""Frozen M05 float32 dtype ablation. Same fullcomponent and adaptive formulas.

Derived from m05_small_split_ut.py; the sole computational change is float64
constants/arrays -> float32. Diagnostic helpers/CLI are not copied.
"""

import math

import flopscope.numpy as fnp

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
    jacobian = fnp.eye(mean.shape[0], dtype=fnp.float32)
    for weight in weights:
        pre = center @ weight
        gate = (pre > 0).astype(fnp.float32)
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
    point_weights = fnp.asarray([kappa / (n + kappa)] + [1 / (2 * (n + kappa))] * (2 * n), dtype=fnp.float32)
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
    pending = [(1.0, fnp.zeros(n, dtype=fnp.float32), fnp.eye(n, dtype=fnp.float32), 0)]
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
    masses = fnp.asarray([a[0] for a in accepted], dtype=fnp.float32)
    means = fnp.stack([a[3] for a in accepted])
    final_mean = fnp.sum(masses[:, None] * means, axis=0)
    covs = fnp.stack([a[4] for a in accepted])
    centered = means - final_mean
    final_cov = fnp.sum(masses[:, None, None] * (covs + centered[:, :, None] * centered[:, None, :]), axis=0)
    return final_mean, final_cov, accepted, history, evaluated

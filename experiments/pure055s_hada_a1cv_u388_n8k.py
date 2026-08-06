"""pure-0.2.55s — Hadamard + a1 multi-CV with subsampled β @ util≈0.40.

Same as pure-0.2.51s but estimate β on n_beta≈2048 rows, apply with full
empirical (c−E[c]). Cuts Gram FLOPs ~12× while keeping most CV gain
(offline probe: n2k often ≥ full-n β).
"""

from __future__ import annotations

import argparse
import math

import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_UTIL = 0.388
_UTIL_CALIB = 1.036
_MATMUL_PER_LAYER = 2.0
_MAX_SAMPLES = 500_000
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_RIDGE_REL = 1.0e-4
_N_BETA = 8192
_CV_RESERVE = 0.0  # subsample Gram is cheap


def _chi_mean_radius(dim: int) -> float:
    half = 0.5 * float(dim)
    return math.sqrt(2.0) * math.exp(math.lgamma(half + 0.5) - math.lgamma(half))


def _sylvester_hadamard(dim: int) -> fnp.ndarray:
    if dim & (dim - 1) != 0 or dim < 1:
        raise ValueError(f"Hadamard size must be power of 2, got {dim}")
    h = [[1.0]]
    while len(h) < dim:
        top = [row + row for row in h]
        bot = [row + [-x for x in row] for row in h]
        h = top + bot
    return fnp.asarray(h, dtype=fnp.float32)


_H256 = _sylvester_hadamard(256)
_INV_SQRT_256 = fnp.float32(1.0 / math.sqrt(256.0))


def _hadamard_directions(n_frames: int, dim: int, seed: int, e_r: float) -> fnp.ndarray:
    if dim != 256:
        H = _sylvester_hadamard(dim) * fnp.float32(1.0 / math.sqrt(dim))
    else:
        H = _H256 * _INV_SQRT_256
    rng = fnp.random.default_rng(seed)
    scale = fnp.float32(e_r)
    blocks = []
    for _ in range(n_frames):
        perm = rng.permutation(dim)
        signs = rng.choice(fnp.asarray([-1.0, 1.0], dtype=fnp.float32), size=dim)
        M = H[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks.append(pos)
        blocks.append(-pos)
    return fnp.concatenate(blocks, axis=0)


def _multi_cv_mean_sub(h, controls, expectations, ridge_rel, n_beta, seed):
    n_all = int(h.shape[0])
    mu = fnp.mean(h, axis=0)
    cc_all = controls - expectations.reshape(1, -1)
    # subsample rows for β
    n_b = int(min(n_beta, n_all))
    rng = fnp.random.default_rng(seed)
    idx = rng.choice(n_all, size=n_b, replace=False)
    hs = h[idx]
    cc = cc_all[idx]
    ns = float(n_b)
    gram = (cc.T @ cc) / ns
    scale = float(fnp.mean(fnp.diag(gram))) + 1e-18
    fnp.fill_diagonal(gram, fnp.diag(gram) + ridge_rel * scale)
    cov = ((hs - fnp.mean(hs, axis=0)).T @ cc) / ns
    beta = fnp.linalg.solve(gram.T, cov.T).T
    # apply with FULL control residual mean
    return mu - beta @ fnp.mean(cc_all, axis=0)


class Estimator(BaseEstimator):
    VERSION = "pure-0.2.55s"

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        depth = mlp.depth
        weights = list(mlp.weights)
        n2 = float(width * width)

        flops_per_sample = depth * _MATMUL_PER_LAYER * n2 + depth * width * 20.0
        target = max(float(budget) * _UTIL * _UTIL_CALIB - _CV_RESERVE, flops_per_sample)
        n_forward = int(max(2 * width, min(target / flops_per_sample, _MAX_SAMPLES)))
        n_frames = max(1, n_forward // (2 * width))

        seed = int(mlp.seed) & 0x7FFFFFFF
        e_r = _chi_mean_radius(width)
        x = _hadamard_directions(n_frames, width, seed=seed, e_r=e_r)
        w0 = weights[0]
        a1 = fnp.maximum(x @ w0, 0.0)
        h = a1
        for w in weights[1:]:
            h = fnp.maximum(h @ w, 0.0)
        col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
        e_a1 = col_norm * _INV_SQRT_2PI
        mu_final = _multi_cv_mean_sub(h, a1, e_a1, _RIDGE_REL, _N_BETA, seed ^ 0xC0FFEE)

        zeros = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zeros, mu_final.reshape(1, width)], axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)
    print(f"--- hada+a1cv-sub β={_N_BETA} (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

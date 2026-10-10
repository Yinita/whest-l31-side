"""pure-0.2.52s-2k — Hadamard + L1-ReLU multi-CV (beta on 2k subsample) @ util≈0.40.

Same spherical design as pure-0.2.1s. Pathwise multi-CV of final activations
onto first-layer ReLU units with E[relu]=σ/√(2π). No Σrelu² (that term alone
destroys ~2×). Antipodal-odd controls (L0/pre) are null under ± pairs.
"""

from __future__ import annotations

import argparse
import math

import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_UTIL = 0.40
_UTIL_CALIB = 1.036
_MATMUL_PER_LAYER = 2.0
_MAX_SAMPLES = 500_000
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_RIDGE_REL = 1.0e-4


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


_BETA_SUB = 2000


def _multi_cv_mean(h, controls, expectations, ridge_rel, beta_n=None, seed=0):
    n_all = int(h.shape[0])
    mu = fnp.mean(h, axis=0)
    cc_full = controls - expectations.reshape(1, -1)
    if beta_n is None or beta_n >= n_all:
        hs, ccs = h, cc_full
        ns = float(n_all)
    else:
        rng = fnp.random.default_rng(seed ^ 0xBE7A)
        idx = rng.choice(n_all, size=int(beta_n), replace=False)
        hs = h[idx]
        ccs = cc_full[idx]
        ns = float(beta_n)
    gram = (ccs.T @ ccs) / ns
    scale = float(fnp.mean(fnp.diag(gram))) + 1e-18
    fnp.fill_diagonal(gram, fnp.diag(gram) + ridge_rel * scale)
    cov = ((hs - fnp.mean(hs, axis=0)).T @ ccs) / ns
    beta = fnp.linalg.solve(gram.T, cov.T).T
    return mu - beta @ fnp.mean(cc_full, axis=0)


class Estimator(BaseEstimator):
    VERSION = "pure-0.2.52s-2k"

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
        # Reserve a little for Gram/solve (~256^3) so util stays near 0.40.
        target = max(float(budget) * _UTIL * _UTIL_CALIB, flops_per_sample)
        n_forward = int(max(2 * width, min(target / flops_per_sample, _MAX_SAMPLES)))
        n_frames = max(1, n_forward // (2 * width))

        e_r = _chi_mean_radius(width)
        x = _hadamard_directions(n_frames, width, seed=int(mlp.seed) & 0x7FFFFFFF, e_r=e_r)
        w0 = weights[0]
        a1 = fnp.maximum(x @ w0, 0.0)
        h = a1
        for w in weights[1:]:
            h = fnp.maximum(h @ w, 0.0)
        col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
        e_a1 = col_norm * _INV_SQRT_2PI
        mu_final = _multi_cv_mean(h, a1, e_a1, _RIDGE_REL, beta_n=_BETA_SUB, seed=int(mlp.seed) & 0x7FFFFFFF)

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
    print(f"--- hada+a1cv util={_UTIL} (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

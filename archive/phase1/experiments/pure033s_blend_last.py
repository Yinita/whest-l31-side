"""pure-0.3.3s — Hadamard full MC blended with penult→last diag-Gaussian ReLU.

Same samples: store penult activations, compute MC final mean and analytic
last-layer mean; ŷ = λ·μ_mc + (1-λ)·μ_ana. λ≈0.9 keeps MC dominant.
"""

from __future__ import annotations

import argparse
import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_UTIL = 0.40
_UTIL_CALIB = 1.036
_MATMUL_PER_LAYER = 2.0
_MAX_SAMPLES = 500_000
_BLEND_MC = 0.90


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
_PM = fnp.asarray([-1.0, 1.0], dtype=fnp.float32)


def _hadamard_directions(n_frames: int, dim: int, seed: int, e_r: float) -> fnp.ndarray:
    H = _H256 * _INV_SQRT_256 if dim == 256 else _sylvester_hadamard(dim) * fnp.float32(
        1.0 / math.sqrt(dim)
    )
    rng = fnp.random.default_rng(seed)
    scale = fnp.float32(e_r)
    blocks = []
    for _ in range(max(1, n_frames)):
        perm = rng.permutation(dim)
        signs = rng.choice(_PM, size=dim)
        M = H[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks.append(pos)
        blocks.append(-pos)
    return fnp.concatenate(blocks, axis=0)


class Estimator(BaseEstimator):
    VERSION = "pure-0.3.3s"

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
        target = max(float(budget) * _UTIL * _UTIL_CALIB, flops_per_sample)
        n_forward = int(max(2 * width, min(target / flops_per_sample, _MAX_SAMPLES)))
        n_frames = max(1, n_forward // (2 * width))

        e_r = _chi_mean_radius(width)
        x = _hadamard_directions(
            n_frames, width, seed=int(mlp.seed) & 0x7FFFFFFF, e_r=e_r
        )
        for w in weights[:-1]:
            x = fnp.maximum(x @ w, 0.0)
        pen = x
        x = fnp.maximum(x @ weights[-1], 0.0)
        mu_mc = fnp.mean(x, axis=0)

        mu = fnp.mean(pen, axis=0)
        second = fnp.mean(pen * pen, axis=0)
        var = fnp.maximum(second - mu * mu, 1e-12)
        wL = weights[-1]
        mu_pre = wL.T @ mu
        var_pre = fnp.maximum((wL * wL).T @ var, 1e-12)
        sigma = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma
        phi = flops.stats.norm.pdf(alpha)
        Phi = flops.stats.norm.cdf(alpha)
        mu_ana = mu_pre * Phi + sigma * phi

        lam = fnp.float32(_BLEND_MC)
        mu_final = lam * mu_mc + (1.0 - lam) * mu_ana

        zeros = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zeros, mu_final.reshape(1, width)], axis=0)

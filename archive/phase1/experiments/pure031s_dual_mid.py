"""pure-0.3.1s — Hadamard+rad dual-path blend: full-depth + mid-resample.

Runs a full-depth Hadamard mean (like pure-0.2.1s) on fraction of frames,
and a mid-resample path on the rest; blends ŷ = λ·μ_full + (1-λ)·μ_mid.
Keeps the proven full-depth estimator while testing whether mid-restart
adds uncorrelated signal.
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
_MID = 16
_FULL_FRAC = 0.70
_BLEND_FULL = 0.85


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


def _forward(weights, x):
    for w in weights:
        x = fnp.maximum(x @ w, 0.0)
    return x


class Estimator(BaseEstimator):
    VERSION = "pure-0.3.1s"

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        depth = mlp.depth
        weights = list(mlp.weights)
        mid = min(max(1, _MID), depth - 1)
        n2 = float(width * width)

        flops_per_act = _MATMUL_PER_LAYER * n2 + width * 20.0
        target = max(float(budget) * _UTIL * _UTIL_CALIB, flops_per_act * depth)
        # Full path: n_full pairs × depth; mid path: n1×mid + n2f×(depth-mid).
        # Let n_full = F·N, n_mid=(1-F)·N, n1=n2f=n_mid/2.
        mid_eff = 0.5 * mid + 0.5 * (depth - mid)  # = depth/2
        eff = _FULL_FRAC * depth + (1.0 - _FULL_FRAC) * mid_eff
        n_pair = int(max(2 * width, min(target / (eff * flops_per_act), _MAX_SAMPLES)))
        n_frames = max(2, n_pair // (2 * width))
        n_full = max(1, int(n_frames * _FULL_FRAC))
        n_mid = max(1, n_frames - n_full)
        n1 = max(1, n_mid // 2)
        n2f = max(1, n_mid - n1)

        e_r = _chi_mean_radius(width)
        base = int(mlp.seed) & 0x7FFFFFFF

        # Full-depth path (proven)
        x_full = _hadamard_directions(n_full, width, seed=base, e_r=e_r)
        h_full = _forward(weights, x_full)
        mu_full = fnp.mean(h_full, axis=0)

        # Mid-resample path
        x1 = _hadamard_directions(n1, width, seed=base ^ 0x11111111, e_r=e_r)
        h = _forward(weights[:mid], x1)
        mu = fnp.mean(h, axis=0)
        second = fnp.mean(h * h, axis=0)
        sd = fnp.sqrt(fnp.maximum(second - mu * mu, 1e-12))
        z = _hadamard_directions(n2f, width, seed=base ^ 0xA5A5A5A5, e_r=fnp.float32(1.0))
        z = z * fnp.float32(math.sqrt(float(width)))
        h2 = fnp.maximum(mu + sd * z, 0.0)
        h2 = _forward(weights[mid:], h2)
        mu_mid = fnp.mean(h2, axis=0)

        lam = fnp.float32(_BLEND_FULL)
        mu_final = lam * mu_full + (1.0 - lam) * mu_mid

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
    print(f"--- dual-path mid (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

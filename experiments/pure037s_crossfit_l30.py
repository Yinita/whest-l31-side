"""pure-0.3.7s — cross-fit L30 recenter @ util≈0.40.

Stream A estimates μ_30; stream B paths are recentered to μ_A then
finish L31. Independence avoids pilot–path correlation that killed
same-sample two-pass recenter.
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
_PILOT_FRAC = 0.45  # frames for μ_30 pilot stream
_RECENTER_LI = 30  # 0-indexed; after weights[0..30]


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
    VERSION = "pure-0.3.7s"

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        depth = mlp.depth
        weights = list(mlp.weights)
        n2 = float(width * width)
        # Pilot stops at L30; path does full depth — allocate by FLOP share.
        flops_act = _MATMUL_PER_LAYER * n2 + width * 20.0
        layers_pilot = _RECENTER_LI + 1  # 31
        layers_path = depth  # 32
        # total cost ∝ n_a*31 + n_b*32; set n_a = f N, n_b=(1-f)N
        f = _PILOT_FRAC
        eff = f * layers_pilot + (1.0 - f) * layers_path
        target = max(float(budget) * _UTIL * _UTIL_CALIB, flops_act * depth)
        n_pair = int(max(2 * width, min(target / (eff * flops_act), _MAX_SAMPLES)))
        n_frames = max(2, n_pair // (2 * width))
        n_a = max(1, int(n_frames * f))
        n_b = max(1, n_frames - n_a)

        e_r = _chi_mean_radius(width)
        base = int(mlp.seed) & 0x7FFFFFFF

        # Stream A: pilot μ_30
        xa = _hadamard_directions(n_a, width, seed=base, e_r=e_r)
        for li in range(_RECENTER_LI + 1):
            xa = fnp.maximum(xa @ weights[li], 0.0)
        mu_a = fnp.mean(xa, axis=0)

        # Stream B: path → L30, recenter to μ_a, finish
        xb = _hadamard_directions(n_b, width, seed=base ^ 0xC0FFEE, e_r=e_r)
        for li in range(_RECENTER_LI + 1):
            xb = fnp.maximum(xb @ weights[li], 0.0)
        mu_b = fnp.mean(xb, axis=0)
        xb = xb - (mu_b - mu_a)
        xb = fnp.maximum(xb, 0.0)
        for li in range(_RECENTER_LI + 1, depth):
            xb = fnp.maximum(xb @ weights[li], 0.0)
        mu_final = fnp.mean(xb, axis=0)

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
    print(f"--- crossfit L30 (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

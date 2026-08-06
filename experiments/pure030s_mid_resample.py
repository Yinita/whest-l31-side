"""pure-0.3.0s — Hadamard+rad with mid-depth diagonal resample @ util≈0.40.

Stage 1: Hadamard inputs → forward to layer `mid`, fit diag (μ,σ).
Stage 2: fresh Hadamard noise in whitened coords → μ+σ⊙(r·U) → ReLU
         → forward remaining layers. Splits frame budget to fight
         angular-frame saturation with a second independent draw.
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
_STAGE1_FRAC = 0.45  # fraction of frames for stage-1 path


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
    VERSION = "pure-0.3.0s"

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

        # Allocate frames so total matmul FLOPs ≈ util·B (stage depths differ).
        flops_per_act = _MATMUL_PER_LAYER * n2 + width * 20.0
        target = max(float(budget) * _UTIL * _UTIL_CALIB, flops_per_act * depth)
        # cost ≈ (n1*2w)*mid + (n2*2w)*(depth-mid)  [samples × layers]
        # set n1 = f·N, n2=(1-f)·N → cost = N·2w·(f·mid+(1-f)·(depth-mid))·flops_per_act
        eff_layers = _STAGE1_FRAC * mid + (1.0 - _STAGE1_FRAC) * (depth - mid)
        n_pair = int(max(2 * width, min(target / (eff_layers * flops_per_act), _MAX_SAMPLES)))
        n_frames = max(2, n_pair // (2 * width))
        n1 = max(1, int(n_frames * _STAGE1_FRAC))
        n2f = max(1, n_frames - n1)

        e_r = _chi_mean_radius(width)
        base = int(mlp.seed) & 0x7FFFFFFF

        # Stage 1: input → mid
        x = _hadamard_directions(n1, width, seed=base, e_r=e_r)
        h = _forward(weights[:mid], x)
        mu = fnp.mean(h, axis=0)
        second = fnp.mean(h * h, axis=0)
        var = fnp.maximum(second - mu * mu, 1e-12)
        sd = fnp.sqrt(var)

        # Stage 2: whitened Hadamard around (μ,σ), then mid → end
        # Use unit-sphere Hadamard (e_r / sqrt(d) scale on coords) so each
        # coordinate has approx variance 1 before σ scaling.
        z = _hadamard_directions(
            n2f, width, seed=base ^ 0xA5A5A5A5, e_r=fnp.float32(1.0)
        )
        # z rows have ||z||≈1; want coordinate-wise ~N(0,1)-like: multiply sqrt(d)
        z = z * fnp.float32(math.sqrt(float(width)))
        h2 = mu + sd * z
        h2 = fnp.maximum(h2, 0.0)
        h2 = _forward(weights[mid:], h2)
        mu_final = fnp.mean(h2, axis=0)

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
    print(f"--- mid-resample (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

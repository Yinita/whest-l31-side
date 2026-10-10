"""pure-0.2.1s — randomized Hadamard frames + radius RB @ util≈0.55.

Sylvester Hadamard (d=256) rows are an ONB (up to 1/√d). Random row
permutation + column signs → antipodal union, exact 1st moment 0 and
exact 2nd angular moment. No blend/CV/corrector.
"""

from __future__ import annotations

import argparse
import math

import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_UTIL = 0.55
_UTIL_CALIB = 1.036
_MATMUL_PER_LAYER = 2.0
_MAX_SAMPLES = 500_000


def _chi_mean_radius(dim: int) -> float:
    half = 0.5 * float(dim)
    return math.sqrt(2.0) * math.exp(math.lgamma(half + 0.5) - math.lgamma(half))


def _sylvester_hadamard(dim: int) -> fnp.ndarray:
    """Build ±1 Hadamard via Sylvester; dim must be a power of 2."""
    if dim & (dim - 1) != 0 or dim < 1:
        raise ValueError(f"Hadamard size must be power of 2, got {dim}")
    # Build in Python floats then one asarray — avoid flopscope matmuls.
    h = [[1.0]]
    while len(h) < dim:
        n = len(h)
        top = [row + row for row in h]
        bot = [row + [-x for x in row] for row in h]
        h = top + bot
    return fnp.asarray(h, dtype=fnp.float32)


# Cache base Hadamard at import for width=256 (Phase 1).
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
    idx = fnp.arange(dim)
    for _ in range(n_frames):
        perm = rng.permutation(dim)
        signs = rng.choice(fnp.asarray([-1.0, 1.0], dtype=fnp.float32), size=dim)
        # permute rows, sign-flip columns → still scaled ONB rows
        M = H[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks.append(pos)
        blocks.append(-pos)
    return fnp.concatenate(blocks, axis=0)


class Estimator(BaseEstimator):
    """Hadamard frames + antipodal + radius scale, L31 sample mean."""

    VERSION = "pure-0.2.3s"

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
        x = _hadamard_directions(n_frames, width, seed=int(mlp.seed) & 0x7FFFFFFF, e_r=e_r)
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        mu_final = fnp.mean(x, axis=0)

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
    print(f"--- hadamard+rad util={_UTIL} (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

"""pure-0.2.2s — randomized orthogonal frames + radius RB @ util≈0.10.

Each frame: QR of a Gaussian matrix → Haar orthogonal Q; take ±columns
scaled by E[χ_d] so first moment is 0 and second angular moment matches
the radius-conditioned Gaussian. No blend/CV/corrector.
"""

from __future__ import annotations

import argparse
import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_UTIL = 0.10
_UTIL_CALIB = 1.036
_MATMUL_PER_LAYER = 2.0
_MAX_SAMPLES = 500_000


def _chi_mean_radius(dim: int) -> float:
    half = 0.5 * float(dim)
    return math.sqrt(2.0) * math.exp(math.lgamma(half + 0.5) - math.lgamma(half))


def _ortho_directions(n_frames: int, dim: int, seed: int, e_r: float) -> fnp.ndarray:
    """n_frames randomized ONB frames → (n_frames * 2 * dim, dim) samples (±)."""
    rng = fnp.random.default_rng(seed)
    blocks = []
    scale = fnp.float32(e_r)
    for i in range(n_frames):
        # Independent CP-style stream seed per frame via advance.
        G = rng.standard_normal((dim, dim)).astype(fnp.float32)
        Q, _R = fnp.linalg.qr(G)
        # Columns of Q are orthonormal; scale to E[R].
        pos = (Q * scale).T  # (dim, dim) — one direction per row
        blocks.append(pos)
        blocks.append(-pos)
    return fnp.concatenate(blocks, axis=0)


class Estimator(BaseEstimator):
    """Ortho frames + antipodal + radius scale, L31 sample mean."""

    VERSION = "pure-0.2.2s"

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
        # Each frame costs 2*width forwards (±); leave small QR headroom via calib.
        target = max(float(budget) * _UTIL * _UTIL_CALIB, flops_per_sample)
        n_forward = int(max(2 * width, min(target / flops_per_sample, _MAX_SAMPLES)))
        n_frames = max(1, n_forward // (2 * width))

        e_r = _chi_mean_radius(width)
        x = _ortho_directions(n_frames, width, seed=int(mlp.seed) & 0x7FFFFFFF, e_r=e_r)
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
    print(f"--- ortho+rad util={_UTIL} (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

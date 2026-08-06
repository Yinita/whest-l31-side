"""pure-0.1.0s — high-util pure Roberts RQMC (no blend / CV / corrector).

Target C/B ≈ 0.10. Keep only: Roberts lattice + CP shift, antithetic,
float32 forward, final-layer sample mean.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

# Empirically flops_per_sample underestimate → actual util ≈ 0.965 · target.
# Scale so measured C/B lands near _UTIL.
_UTIL = 0.10
_UTIL_CALIB = 1.036
_MATMUL_PER_LAYER = 2.0
_PPF_EPS = 1e-6
_MAX_SAMPLES = 500_000


def _roberts_alpha(dim: int) -> fnp.ndarray:
    """α_j = φ^{-j} for φ root of x^{dim}=x+1 (j=1..dim)."""
    lo, hi = 1.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        try:
            lhs = math.exp(dim * math.log(mid))
        except OverflowError:
            lhs = float("inf")
        if lhs - mid - 1.0 > 0.0:
            hi = mid
        else:
            lo = mid
    log_phi = math.log(lo)
    alphas = [math.exp(-(j + 1) * log_phi) for j in range(dim)]
    return fnp.asarray(alphas, dtype=fnp.float32)


def _rqmc_standard_normal(n_samples: int, dim: int, seed: int) -> fnp.ndarray:
    """Roberts RQMC → standard normal; cast to float32 immediately after ppf."""
    rng = fnp.random.default_rng(seed)
    shift = rng.random(dim).astype(fnp.float32)
    alpha = _roberts_alpha(dim)
    idx = fnp.arange(1, n_samples + 1, dtype=fnp.float32).reshape(n_samples, 1)
    raw = idx * alpha.reshape(1, dim) + shift.reshape(1, dim)
    u = raw - fnp.floor(raw)
    u = fnp.clip(u, _PPF_EPS, 1.0 - _PPF_EPS)
    # ppf may be float64 internally; do not let that promote the forward path.
    return flops.stats.norm.ppf(u).astype(fnp.float32)


class Estimator(BaseEstimator):
    """Pure Roberts RQMC + antithetic, L31-only sample mean."""

    VERSION = "pure-0.1.0s"

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
        n_samples = int(max(64, min(target / flops_per_sample, _MAX_SAMPLES)))

        n_half = max(32, n_samples // 2)
        x = _rqmc_standard_normal(n_half, width, seed=int(mlp.seed) & 0x7FFFFFFF)
        x = fnp.concatenate([x, -x], axis=0)
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
    print(f"--- pure RQMC util={_UTIL} (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

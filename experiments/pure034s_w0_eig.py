"""pure-0.3.4s — W0-informed Hadamard frames + random Hadamard @ util≈0.40.

Spend one (or few) ONB frames aligned to eigenvectors of W0 W0ᵀ
(input-side energy), rest standard randomized Hadamard+antipodal+E[χ].
Tests whether MLP-adaptive angular design beats Haar-random frames.
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
_N_EIG_FRAMES = 2


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


def _random_hadamard_frames(n_frames: int, dim: int, seed: int, e_r: float) -> fnp.ndarray:
    H = _H256 * _INV_SQRT_256 if dim == 256 else _sylvester_hadamard(dim) * fnp.float32(
        1.0 / math.sqrt(dim)
    )
    rng = fnp.random.default_rng(seed)
    scale = fnp.float32(e_r)
    blocks = []
    for _ in range(max(0, n_frames)):
        perm = rng.permutation(dim)
        signs = rng.choice(_PM, size=dim)
        M = H[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks.append(pos)
        blocks.append(-pos)
    if not blocks:
        return fnp.zeros((0, dim), dtype=fnp.float32)
    return fnp.concatenate(blocks, axis=0)


def _eig_frames(w0: fnp.ndarray, n_frames: int, e_r: float, seed: int) -> fnp.ndarray:
    """Build ONB frames from eigenvectors of W0 W0ᵀ, with random sign flips."""
    # S = W0 @ W0.T — (d,d) PSD
    S = w0 @ w0.T
    # eigh returns ascending eigenvalues; take all eigenvectors as ONB
    evals, evecs = fnp.linalg.eigh(S)
    # evecs columns are eigenvectors; take rows as directions → evecs.T
    base = evecs.T  # (d,d) orthonormal rows
    rng = fnp.random.default_rng(seed)
    scale = fnp.float32(e_r)
    blocks = []
    for i in range(max(0, n_frames)):
        signs = rng.choice(_PM, size=base.shape[0])
        # optional: rotate by a Hadamard row-perm of identity in eigenbasis
        if i == 0:
            M = base * signs.reshape(-1, 1)
        else:
            perm = rng.permutation(base.shape[0])
            M = base[perm] * signs.reshape(-1, 1)
        pos = M * scale
        blocks.append(pos)
        blocks.append(-pos)
    if not blocks:
        return fnp.zeros((0, base.shape[1]), dtype=fnp.float32)
    return fnp.concatenate(blocks, axis=0)


class Estimator(BaseEstimator):
    VERSION = "pure-0.3.4s"

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
        # reserve ~1.5e9 for eigh(W0 W0ᵀ)
        eig_reserve = 1.5e9
        target = max(float(budget) * _UTIL * _UTIL_CALIB - eig_reserve, flops_per_sample)
        n_forward = int(max(2 * width, min(target / flops_per_sample, _MAX_SAMPLES)))
        n_frames = max(1, n_forward // (2 * width))
        n_eig = min(_N_EIG_FRAMES, n_frames)
        n_rand = n_frames - n_eig

        e_r = _chi_mean_radius(width)
        base = int(mlp.seed) & 0x7FFFFFFF
        parts = [
            _eig_frames(weights[0], n_eig, e_r, seed=base ^ 0xE16E16),
        ]
        if n_rand > 0:
            parts.append(_random_hadamard_frames(n_rand, width, seed=base, e_r=e_r))
        x = fnp.concatenate(parts, axis=0)
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
    print(f"--- W0-eig Hadamard (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

"""pure-0.2.1s-e — single-stream probe + accumulate (no dual-mean merge).

d2 scored 3.46e-7: split streams hurt. Here probe frames stay in the
same running sum; fps from public budget_summary_dict; static fallback.
"""

from __future__ import annotations

import argparse
import math

import flopscope
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_UTIL = 0.40
_UTIL_CALIB = 1.036
_MAX_SAMPLES = 500_000
_PROBE_FRAMES = 2
_MATMUL_PER_LAYER = 2.0
_RELU_PER_ELEM = 20.0


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


def _hadamard_directions(n_frames: int, dim: int, rng, e_r: float) -> fnp.ndarray:
    if dim != 256:
        H = _sylvester_hadamard(dim) * fnp.float32(1.0 / math.sqrt(dim))
    else:
        H = _H256 * _INV_SQRT_256
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


def _forward(x: fnp.ndarray, weights: list) -> fnp.ndarray:
    for w in weights:
        x = fnp.maximum(x @ w, 0.0)
    return x


def _flops_used() -> float | None:
    try:
        return float(flopscope.budget_summary_dict()["flops_used"])
    except Exception:
        return None


class Estimator(BaseEstimator):
    VERSION = "pure-0.2.1s-e"

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        depth = mlp.depth
        weights = list(mlp.weights)
        e_r = _chi_mean_radius(width)
        rng = fnp.random.default_rng(int(mlp.seed) & 0x7FFFFFFF)
        target_f = float(budget) * _UTIL * _UTIL_CALIB
        n2 = float(width * width)
        fps_static = depth * _MATMUL_PER_LAYER * n2 + depth * width * _RELU_PER_ELEM

        f0 = _flops_used()
        x = _forward(_hadamard_directions(_PROBE_FRAMES, width, rng, e_r), weights)
        f1 = _flops_used()
        n_tot = int(x.shape[0])
        acc = fnp.sum(x, axis=0)

        if f0 is not None and f1 is not None and f1 > f0:
            fps = max((f1 - f0) / max(n_tot, 1), 1.0)
            spent = f1 - f0
        else:
            fps = fps_static
            spent = fps * n_tot

        remain = max(target_f - spent, 0.0)
        n_more = int(min(remain / fps, _MAX_SAMPLES - n_tot))
        n_frames_more = max(0, n_more // (2 * width))

        if n_frames_more > 0:
            x2 = _forward(_hadamard_directions(n_frames_more, width, rng, e_r), weights)
            acc = acc + fnp.sum(x2, axis=0)
            n_tot += int(x2.shape[0])

        mu = acc / fnp.float32(n_tot)
        zeros = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zeros, mu.reshape(1, width)], axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)
    print(f"--- single-stream probe-fill util={_UTIL} (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

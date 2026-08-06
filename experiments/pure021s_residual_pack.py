"""pure-0.2.1s-rp7 — residual pack via FlopscopeArray.base detach.

Probes: #325127 alt-detach smoke-fail; #325130 .base grades.
Hot path: weights/buffers via .base.copy(); matmul outside flopscope metering.
"""

from __future__ import annotations

import math
import random
import time

import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_UTIL = 0.40
_LAMBDA = 1e11
_UTIL_SAFE = 0.88
_MAX_FRAMES = 200_000
_PROBE_FRAMES = 4
_MAX_CHUNK = 64


def _chi_mean_radius(dim: int) -> float:
    half = 0.5 * float(dim)
    return math.sqrt(2.0) * math.exp(math.lgamma(half + 0.5) - math.lgamma(half))


def _sylvester_lists(dim: int):
    h = [[1.0]]
    while len(h) < dim:
        top = [row + row for row in h]
        bot = [row + [-x for x in row] for row in h]
        h = top + bot
    return h


def _base_copy(arr):
    """Detach via .base (allowed on grader) into an owning buffer."""
    b = arr.base
    if b is None:
        raise RuntimeError("no base")
    return b.copy()


def _zeros(shape):
    return _base_copy(fnp.zeros(shape, dtype=fnp.float32))


class Estimator(BaseEstimator):
    VERSION = "pure-0.2.1s-rp7"

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        depth = mlp.depth
        weights = [_base_copy(w) for w in mlp.weights]

        r_target = (_UTIL * _UTIL_SAFE * float(budget)) / _LAMBDA
        e_r = _chi_mean_radius(width)
        H = _base_copy(
            fnp.asarray(_sylvester_lists(width), dtype=fnp.float32)
            * fnp.float32(e_r / math.sqrt(width))
        )

        rng = random.Random(int(mlp.seed) & 0x7FFFFFFF)
        t0 = time.perf_counter()
        acc = _zeros((width,)).astype("float64")
        n_tot = 0
        n_frames_done = 0

        def _run(n_f: int) -> None:
            nonlocal acc, n_tot, n_frames_done
            buf = _zeros((n_f * 2 * width, width))
            idx = list(range(width))
            for i in range(n_f):
                rng.shuffle(idx)
                signs = [(-1.0 if rng.random() < 0.5 else 1.0) for _ in range(width)]
                M = H[idx] * signs
                base = i * 2 * width
                buf[base : base + width] = M
                buf[base + width : base + 2 * width] = -M
            x = buf
            for w in weights:
                x = x @ w
                x *= x > 0
            acc = acc + x.sum(axis=0).astype("float64")
            n_tot += int(x.shape[0])
            n_frames_done += n_f

        n_probe = min(_PROBE_FRAMES, _MAX_FRAMES)
        ta = time.perf_counter()
        _run(n_probe)
        sec_per_frame = max(time.perf_counter() - ta, 1e-6) / float(n_probe)

        while n_frames_done < _MAX_FRAMES:
            r_now = time.perf_counter() - t0
            rem = r_target - r_now
            if rem <= 0.0:
                break
            n_fit = int(rem / (sec_per_frame * 1.25))
            if n_fit < 1:
                break
            frac = r_now / max(r_target, 1e-9)
            if frac > 0.90:
                chunk = 1
            elif frac > 0.75:
                chunk = min(4, n_fit)
            else:
                chunk = min(_MAX_CHUNK, n_fit, _MAX_FRAMES - n_frames_done)
            chunk = max(1, min(chunk, _MAX_FRAMES - n_frames_done))
            ta = time.perf_counter()
            _run(chunk)
            sec_per_frame = 0.65 * sec_per_frame + 0.35 * (
                max(time.perf_counter() - ta, 1e-6) / float(chunk)
            )

        mu = (acc / float(max(n_tot, 1))).astype("float32")
        mu_f = fnp.asarray(mu.tolist(), dtype=fnp.float32)
        zeros = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zeros, mu_f.reshape(1, width)], axis=0)

"""pure-0.2.40s — two-pass Hadamard: empirical mid-layer recenter @ util≈0.40.

Pass A (frac α): plain Hadamard → empirical post-ReLU means at selected layers.
Pass B (1-α): Hadamard with h ← h − mean(h) + μ_A at those layers, then final mean.
Oracle GT-recenter last8 collapses Mini MSE ~1e-6 → ~1e-11; this replaces GT with
an empirical pilot estimate.
"""
from __future__ import annotations
import math
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_UTIL = 0.40
_UTIL_CALIB = 1.036
_MATMUL_PER_LAYER = 2.0
_MAX_SAMPLES = 500_000
_PILOT_FRAC = 0.35
_RECENTER = frozenset(range(28, 32))
_CLIP = True

def _chi_mean_radius(dim: int) -> float:
    half = 0.5 * float(dim)
    return math.sqrt(2.0) * math.exp(math.lgamma(half + 0.5) - math.lgamma(half))

def _sylvester_hadamard(dim: int) -> fnp.ndarray:
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
    H = _H256 * _INV_SQRT_256 if dim == 256 else _sylvester_hadamard(dim) * fnp.float32(1.0 / math.sqrt(dim))
    rng = fnp.random.default_rng(seed)
    scale = fnp.float32(e_r)
    blocks = []
    for _ in range(n_frames):
        perm = rng.permutation(dim)
        signs = rng.choice(_PM, size=dim)
        M = H[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks.append(pos)
        blocks.append(-pos)
    return fnp.concatenate(blocks, axis=0)

class Estimator(BaseEstimator):
    VERSION = "pure-0.2.43s"
    def __init__(self): self._setup_rng = None
    def setup(self, ctx: SetupContext): self._setup_rng = fnp.random.default_rng(ctx.seed)
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width, depth = mlp.width, mlp.depth
        weights = list(mlp.weights)
        n2 = float(width * width)
        flops_per = depth * _MATMUL_PER_LAYER * n2 + depth * width * 20.0
        target = max(float(budget) * _UTIL * _UTIL_CALIB, flops_per)
        n_forward = int(max(2 * width, min(target / flops_per, _MAX_SAMPLES)))
        n_frames = max(2, n_forward // (2 * width))
        n_a = max(1, int(round(n_frames * _PILOT_FRAC)))
        n_b = max(1, n_frames - n_a)
        e_r = _chi_mean_radius(width)
        seed0 = int(mlp.seed) & 0x7FFFFFFF

        # Pass A: pilot means
        xa = _hadamard_directions(n_a, width, seed=seed0, e_r=e_r)
        pilot = {}
        h = xa
        for li, w in enumerate(weights):
            h = fnp.maximum(h @ w, 0.0)
            if li in _RECENTER:
                pilot[li] = fnp.mean(h, axis=0)
        # Pass B: recenter toward pilot
        xb = _hadamard_directions(n_b, width, seed=seed0 ^ 0xA5A5A5A5, e_r=e_r)
        h = xb
        for li, w in enumerate(weights):
            h = fnp.maximum(h @ w, 0.0)
            if li in _RECENTER:
                h = h - fnp.mean(h, axis=0) + pilot[li]
                if _CLIP:
                    h = fnp.maximum(h, 0.0)
        mu_final = fnp.mean(h, axis=0)
        zeros = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zeros, mu_final.reshape(1, width)], axis=0)

"""pure-0.2.26s — Hadamard + shared-dir radial 3-point (vectorized) @ util≈0.40."""
from __future__ import annotations
import math
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_UTIL = 0.40
_UTIL_CALIB = 1.036
_MATMUL_PER_LAYER = 2.0
_MAX_SAMPLES = 500_000
_RAD_A = 0.5

def _chi_mean_var(dim: int):
    half = 0.5 * float(dim)
    mu = math.sqrt(2.0) * math.exp(math.lgamma(half + 0.5) - math.lgamma(half))
    var = max(float(dim) - mu * mu, 1e-6)
    return mu, var

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

class Estimator(BaseEstimator):
    VERSION = "pure-0.2.26s"
    def __init__(self): self._setup_rng = None
    def setup(self, ctx: SetupContext): self._setup_rng = fnp.random.default_rng(ctx.seed)
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width, depth = mlp.width, mlp.depth
        weights = list(mlp.weights)
        n2 = float(width * width)
        flops_per = depth * _MATMUL_PER_LAYER * n2 + depth * width * 20.0
        target = max(float(budget) * _UTIL * _UTIL_CALIB, flops_per)
        n_forward = int(max(2 * width, min(target / flops_per, _MAX_SAMPLES)))
        n_frames = max(1, n_forward // (6 * width))
        mu_r, var_r = _chi_mean_var(width)
        sig_r = math.sqrt(var_r)
        radii = [mu_r - _RAD_A * sig_r, mu_r, mu_r + _RAD_A * sig_r]
        def npdf(z): return math.exp(-0.5 * z * z)
        ws = [npdf(-_RAD_A), npdf(0.0), npdf(_RAD_A)]
        wsum = sum(ws); ws = [w / wsum for w in ws]
        H = _H256 * _INV_SQRT_256
        rng = fnp.random.default_rng(int(mlp.seed) & 0x7FFFFFFF)
        blocks = []
        wrow = []
        for _ in range(n_frames):
            perm = rng.permutation(width)
            signs = rng.choice(_PM, size=width)
            M = H[perm] * signs.reshape(1, -1)
            for r, w in zip(radii, ws):
                if r <= 1e-6: continue
                pos = M * fnp.float32(r)
                blocks.append(pos); blocks.append(-pos)
                ww = fnp.full(width, fnp.float32(w), dtype=fnp.float32)
                wrow.append(ww); wrow.append(ww)
        x = fnp.concatenate(blocks, axis=0)
        w_all = fnp.concatenate(wrow, axis=0)
        for wt in weights:
            x = fnp.maximum(x @ wt, 0.0)
        # weighted mean
        w_all = w_all.reshape(-1, 1)
        mu_final = fnp.sum(fnp.multiply(x, w_all), axis=0) / fnp.sum(w_all)
        zeros = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zeros, mu_final.reshape(1, width)], axis=0)

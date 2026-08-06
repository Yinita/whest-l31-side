"""hyb-0.3.0s — pure-0.2.1s Hadamard + Mini-retrained residual corrector.

θ from experiments/corrector_hada_v030.npz (5-fold OOS ratio ≈1.029 on Mini).
Features: gain last-layer (α,φ,σ) + Hadamard/gain gap (14-dim, shared across neurons).
No blend/CV; correction is additive in mean space (not σ-scaled).
"""

from __future__ import annotations

import argparse
import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_UTIL = 0.40
_UTIL_CALIB = 1.036
_MATMUL_PER_LAYER = 2.0
_MAX_SAMPLES = 500_000
_GAIN_RESERVE = 0.0

# experiments/corrector_hada_v030.npz (lam=1e-2, Mini 5-fold OOS).
_CORR_BETA = fnp.asarray(
    [
        0.0,
        -4.738207548059788e-05,
        2.569282932033616e-05,
        -6.038033519930855e-05,
        -3.909287106882999e-05,
        3.2296980762043763e-06,
        -0.00011389788631107604,
        -4.66015603057141e-05,
        -3.775531851304512e-05,
        -3.601497547070141e-05,
        -0.00011389788631107473,
        0.00016595808293959547,
        2.5244522593656407e-06,
        -0.0002233789142994992,
    ],
    dtype=fnp.float32,
)
_CORR_MU = fnp.asarray(
    [
        1.0,
        -0.03754740092555671,
        17.125511864177103,
        -1.6739673551435836,
        516.8474862822872,
        0.06389155718739549,
        -0.015906711041912457,
        -0.018668616548516807,
        1.8088204927022782,
        1.8247272037441915,
        -0.015906711041912457,
        -1.4721343092044583,
        0.0006390843481606491,
        0.24853172262795792,
    ],
    dtype=fnp.float32,
)
_CORR_SD = fnp.asarray(
    [
        1.0,
        4.138127854431687,
        14.95206774901911,
        138.48488381173073,
        735.5925361584767,
        0.11807696584258671,
        0.028475593426728184,
        0.00913765115988688,
        2.246150583562552,
        2.2646363003704666,
        0.028475593426728184,
        0.38474974165176823,
        0.09448270430202649,
        0.11232098981509224,
    ],
    dtype=fnp.float32,
)


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
    for _ in range(n_frames):
        perm = rng.permutation(dim)
        signs = rng.choice(_PM, size=dim)
        M = H[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks.append(pos)
        blocks.append(-pos)
    return fnp.concatenate(blocks, axis=0)


def _gain_last_stats(weights: list):
    width = weights[0].shape[0]
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = fnp.eye(width, dtype=fnp.float32)
    alpha = phi = sigma = None
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sigma = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma
        phi = flops.stats.norm.pdf(alpha)
        Phi = flops.stats.norm.cdf(alpha)
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sigma * phi
        var_post = fnp.maximum(ez2 - mu * mu, 0.0)
        gain = fnp.where(sigma > 1e-12, Phi, fnp.zeros_like(Phi))
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
    return mu, alpha, phi, sigma


class Estimator(BaseEstimator):
    """Hadamard+rad @0.40 + Mini-retrained residual θ."""

    VERSION = "hyb-0.3.0s-s0.0"

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
        target = max(float(budget) * _UTIL * _UTIL_CALIB - _GAIN_RESERVE, flops_per_sample)
        n_forward = int(max(2 * width, min(target / flops_per_sample, _MAX_SAMPLES)))
        n_frames = max(1, n_forward // (2 * width))

        e_r = _chi_mean_radius(width)
        x = _hadamard_directions(
            n_frames, width, seed=int(mlp.seed) & 0x7FFFFFFF, e_r=e_r
        )
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        mu_h = fnp.mean(x, axis=0)

        mu_g, alpha, phi, sigma = _gain_last_stats(weights)
        gap = mu_h - mu_g
        glayer = fnp.full(width, float(fnp.mean(gap)), dtype=fnp.float32)
        s = fnp.maximum(sigma, 1e-8)
        ones = fnp.ones(width, dtype=fnp.float32)
        a = alpha
        feats = fnp.stack(
            [
                ones,
                a,
                a * a,
                a * a * a,
                a * a * a * a,
                phi,
                gap / s,
                glayer / s,
                mu_h / s,
                mu_g / s,
                (mu_h - mu_g) / s,
                fnp.log(s + 1e-8),
                phi * a,
                sigma,
            ],
            axis=0,
        )
        Xs = (feats - _CORR_MU.reshape(-1, 1)) / _CORR_SD.reshape(-1, 1)
        mu_final = mu_h + _CORR_SCALE * (_CORR_BETA @ Xs)

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
    print(f"--- hada+corr (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))

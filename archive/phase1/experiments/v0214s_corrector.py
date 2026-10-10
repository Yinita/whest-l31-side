"""v0.2.14s — v0.2.12s + synth-trained radiant-style residual corrector.

Advances on v0.2.12s (#321321, online 4.945e-07):

  Offline ridge on 200 synth MLPs (N_GT=1e5), radiant-ish features, fit on
  noisy predict-time stats. Synth 5-fold OOS ratio ≈1.020; Mini transfer
  ratio ≈1.020 (matched — not Mini-overfit). Embed θ in-file (no extra asset).

Pipeline (L31-only):
  1. Roberts RQMC + antithetic
  2. Multi-CV on [a₁ | Σa₁²]
  3. Blend with 0.992·gain-cov
  4. pred ← pred + σ · (θ · f)  (14 features; σ = sample last-preact std)
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_FLOOR_FRAC = 0.094
_MATMUL_PER_LAYER = 2.0
_PPF_EPS = 1e-6
_BLEND_RQMC = 0.85
_GAIN_SCALE = 0.992
_GAIN_RESERVE = 2.0e9
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_RIDGE_REL = 1.0e-4

# Synth-trained corrector (experiments/corrector_v0214.npz).
_CORR_BETA = fnp.asarray(
    [
        -1.0345499492818470e-04,
        -6.5083034063491711e-04,
        1.1486142885622889e-04,
        6.2409043545687731e-05,
        -7.4132816116160194e-05,
        4.3846623039965851e-05,
        -2.1578946308376326e-03,
        -7.7856700009445263e-04,
        3.9619132891951873e-05,
        -8.1664892303548198e-05,
        1.1549689837697581e-03,
        -1.1549721610475446e-03,
        5.8909164150878796e-04,
        -1.2327134407405933e-04,
    ],
    dtype=fnp.float32,
)
_CORR_MU = fnp.asarray(
    [
        0.0,
        -2.6987235284101293e-02,
        1.7778910804597579e01,
        -1.8020884902869230e00,
        5.5564007078510610e02,
        6.2452121967139333e-02,
        -5.4922482980348012e-04,
        -2.9109729502083863e-04,
        -2.1761927266966847e-06,
        1.5390698384997121e-04,
        -8.8342032902659129e-05,
        5.0060485311511835e-04,
        -1.6823972892955678e-02,
        6.4712459344945378e-02,
    ],
    dtype=fnp.float32,
)
_CORR_SD = fnp.asarray(
    [
        1.0,
        4.2164182166926540e00,
        1.5477415859788026e01,
        1.4575731769366541e02,
        7.8075989315844799e02,
        1.1652090848697132e-01,
        2.1831233228664967e-02,
        6.4827521832910044e-03,
        5.6874752537538435e-03,
        2.2634082429298201e-03,
        2.9936115140142042e-03,
        1.6963751912747151e-02,
        3.4826156517002000e00,
        1.1687906784227733e-01,
    ],
    dtype=fnp.float32,
)


def _roberts_alpha(dim: int) -> fnp.ndarray:
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
    rng = fnp.random.default_rng(seed)
    shift = rng.random(dim).astype(fnp.float32)
    alpha = _roberts_alpha(dim)
    idx = fnp.arange(1, n_samples + 1, dtype=fnp.float32).reshape(n_samples, 1)
    raw = idx * alpha.reshape(1, dim) + shift.reshape(1, dim)
    u = raw - fnp.floor(raw)
    u = fnp.clip(u, _PPF_EPS, 1.0 - _PPF_EPS)
    return flops.stats.norm.ppf(u).astype(fnp.float32)


def _multi_cv_mean(
    h: fnp.ndarray, controls: fnp.ndarray, expectations: fnp.ndarray, ridge_rel: float
) -> fnp.ndarray:
    n = float(h.shape[0])
    mu = fnp.mean(h, axis=0)
    cc = controls - expectations.reshape(1, -1)
    gram = (cc.T @ cc) / n
    scale = float(fnp.mean(fnp.diag(gram))) + 1e-18
    fnp.fill_diagonal(gram, fnp.diag(gram) + ridge_rel * scale)
    cov = ((h - mu).T @ cc) / n
    beta = fnp.linalg.solve(gram.T, cov.T).T
    return mu - beta @ fnp.mean(cc, axis=0)


def _gain_and_last_stats(weights: list):
    """Gain-cov final mean + last-layer (α, φ, σ)."""
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
    """Roberts RQMC + L1+sq CV + gain blend + synth residual corrector."""

    VERSION = "0.2.14s"

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        depth = mlp.depth
        weights = list(mlp.weights)
        n2 = float(width * width)

        blend = _BLEND_RQMC
        reserve = _GAIN_RESERVE if blend < 1.0 else 0.0
        flops_per_sample = depth * _MATMUL_PER_LAYER * n2 + depth * width * 20.0
        target = max(float(budget) * _FLOOR_FRAC - reserve, flops_per_sample)
        n_samples = int(max(64, min(target / flops_per_sample, 200_000)))

        n_half = max(32, n_samples // 2)
        x = _rqmc_standard_normal(n_half, width, seed=int(mlp.seed) & 0x7FFFFFFF)
        x = fnp.concatenate([x, -x], axis=0)

        w0 = weights[0]
        a1 = fnp.maximum(x @ w0, 0.0)
        col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
        e_a1 = col_norm * _INV_SQRT_2PI
        q1 = fnp.sum(a1 * a1, axis=1)
        e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
        controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
        expectations = fnp.concatenate(
            [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0
        )

        h = a1
        for w in weights[1:-1]:
            h = fnp.maximum(h @ w, 0.0)
        zpre = h @ weights[-1]
        last = fnp.maximum(zpre, 0.0)
        mu_cv = _multi_cv_mean(last, controls, expectations, _RIDGE_REL)

        mu_g_raw, alpha_g, phi_g, sig_g = _gain_and_last_stats(weights)
        mu_g = _GAIN_SCALE * mu_g_raw
        pred = blend * mu_cv + (1.0 - blend) * mu_g

        # Sample last-preact moments for features / σ scale.
        zc = zpre - fnp.mean(zpre, axis=0)
        m2 = fnp.maximum(fnp.mean(zc * zc, axis=0), 1e-12)
        m3 = fnp.mean(zc * zc * zc, axis=0)
        m4 = fnp.mean(zc * zc * zc * zc, axis=0)
        s = fnp.sqrt(m2)
        skew = m3 / (m2 * s)
        kurt = m4 / (m2 * m2) - 3.0
        alpha_s = fnp.mean(zpre, axis=0) / s
        phi_s = fnp.exp(-0.5 * alpha_s * alpha_s) * _INV_SQRT_2PI
        st = -(skew / 6.0) * phi_s
        kt = (kurt / 24.0) * (alpha_s * alpha_s - 1.0) * phi_s
        gap = mu_cv - mu_g
        glayer = fnp.full(width, float(fnp.mean(gap)), dtype=fnp.float32)

        ones = fnp.ones(width, dtype=fnp.float32)
        a = alpha_g
        feats = fnp.stack(
            [
                ones,
                a,
                a * a,
                a * a * a,
                a * a * a * a,
                phi_g,
                gap / (sig_g + 1e-8),
                glayer / (sig_g + 1e-8),
                st,
                kt,
                (mu_cv - pred) / (s + 1e-8),
                (mu_g - pred) / (s + 1e-8),
                alpha_s,
                phi_s,
            ],
            axis=0,
        )  # (F, width)
        # standardize per feature, apply θ
        Xs = (feats - _CORR_MU.reshape(-1, 1)) / _CORR_SD.reshape(-1, 1)
        delta = s * (_CORR_BETA @ Xs)
        mu_final = pred + delta

        zeros = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zeros, mu_final.reshape(1, width)], axis=0)


def _load_baseline(name: str) -> type[BaseEstimator]:
    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(f"Could not find baseline `{name}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)
    print(f"--- RQMC+CV+blend+corr (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))
    if args.baseline:
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(
            _load_baseline(args.baseline)(), mlp, estimator_budget=int(2.72e11)
        )

"""v0.2.20s — pooled dual multi-CV + pooled-retrained residual corrector.

Same as v0.2.18s but θ retrained on pooled base (300 synth / N_GT=1.5e5).
Keep λ=0.85 (λ=0.90 Mini↑ online↓).
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
        -1.6441005535427580e-04,
        -2.8080400321692728e-03,
        -2.6380186867668606e-04,
        6.0643308848518985e-04,
        3.4375878308765575e-04,
        7.6345414899407717e-04,
        -4.9362769839015590e-03,
        -6.3333224962219847e-04,
        1.2558853301109762e-04,
        -1.2156152670487382e-04,
        2.5170193445246155e-03,
        -2.5170261295451809e-03,
        2.2351412747262153e-03,
        -8.9983834442884279e-04,
    ],
    dtype=fnp.float32,
)
_CORR_MU = fnp.asarray(
    [
        0.0000000000000000e+00,
        3.5329243710260498e-04,
        1.7569566694363211e+01,
        -1.6614737160572307e-01,
        5.3933688823757984e+02,
        6.2407742580782702e-02,
        -3.5968440360112722e-04,
        -5.1009953218086223e-05,
        -2.1546699415301378e-05,
        1.5587280234636859e-04,
        -6.3532765520261629e-05,
        3.6001900461487331e-04,
        1.2164325519977310e-04,
        6.4687673531527073e-02,
    ],
    dtype=fnp.float32,
)
_CORR_SD = fnp.asarray(
    [
        1.0000000000000000e+00,
        4.1916066911602998e+00,
        1.5187073935213604e+01,
        1.4211313645471157e+02,
        7.4901068271294503e+02,
        1.1640103451822517e-01,
        2.2294748420514540e-02,
        7.2259337092529128e-03,
        5.7980155403325757e-03,
        2.3184147491481619e-03,
        3.0550317279463887e-03,
        1.7311799791696194e-02,
        3.4789117362896631e+00,
        1.1681981978718957e-01,
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

    VERSION = "0.2.20s"

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

        def _stream_parts(n_half: int, seed: int):
            x = _rqmc_standard_normal(n_half, width, seed=seed)
            x = fnp.concatenate([x, -x], axis=0)
            w0 = weights[0]
            a1 = fnp.maximum(x @ w0, 0.0)
            h = a1
            for w in weights[1:-1]:
                h = fnp.maximum(h @ w, 0.0)
            zpre_s = h @ weights[-1]
            last = fnp.maximum(zpre_s, 0.0)
            return a1, last, zpre_s

        seed0 = int(mlp.seed) & 0x7FFFFFFF
        n_half = max(32, n_samples // 4)
        a1_a, last_a, z1 = _stream_parts(n_half, seed0)
        a1_b, last_b, z2 = _stream_parts(n_half, seed0 ^ 0xA5A5A5A5)
        a1 = fnp.concatenate([a1_a, a1_b], axis=0)
        last = fnp.concatenate([last_a, last_b], axis=0)
        zpre = fnp.concatenate([z1, z2], axis=0)
        w0 = weights[0]
        col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
        e_a1 = col_norm * _INV_SQRT_2PI
        q1 = fnp.sum(a1 * a1, axis=1)
        e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
        controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
        expectations = fnp.concatenate(
            [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0
        )
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

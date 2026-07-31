"""v0.2.12s — dual-RQMC floor0.094 + λ=0.85 + Roberts RQMC + antithetic + all-L1+sq multi-CV + gain blend.

Advances on v0.2.8s (#321279, online 4.976e-07):

  * v0.2.9s top-k / block L1: Mini worse than all-256 (e.g. top-128 adj 4.84e-7).
  * Mini GT: append exact L1 sum-of-squares control
      E[Σ_j relu(Z_j)²] = Σ_j σ_j² / 2
    to the all-256 multi-CV → raw ~4.33e-6 → ~4.30e-6; val improves more
    (~4.17e-6 → ~4.03e-6). Extra column is FLOP-cheap vs the 256×256 Gram.

Pipeline (L31-only):
  1. Roberts RQMC + antithetic
  2. Forward; keep L1 activations a₁
  3. Multi-CV on [a₁ | Σa₁²] exact expectations, ridge Gram
  4. ŷ = 0.85·μ + 0.15·(0.992·gain-cov)
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
    rng = fnp.random.default_rng(seed)
    shift = rng.random(dim).astype(fnp.float32)
    alpha = _roberts_alpha(dim)
    idx = fnp.arange(1, n_samples + 1, dtype=fnp.float32).reshape(n_samples, 1)
    raw = idx * alpha.reshape(1, dim) + shift.reshape(1, dim)
    u = raw - fnp.floor(raw)
    u = fnp.clip(u, _PPF_EPS, 1.0 - _PPF_EPS)
    return flops.stats.norm.ppf(u).astype(fnp.float32)


def _gain_cov_final(weights: list) -> fnp.ndarray:
    width = weights[0].shape[0]
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = fnp.eye(width, dtype=fnp.float32)
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
    return mu


def _multi_cv_mean(
    h: fnp.ndarray, controls: fnp.ndarray, expectations: fnp.ndarray, ridge_rel: float
) -> fnp.ndarray:
    """Multi-CV: mean(h) − Β (mean(C) − E[C]), ridge on control Gram."""
    n = float(h.shape[0])
    mu = fnp.mean(h, axis=0)
    cc = controls - expectations.reshape(1, -1)
    gram = (cc.T @ cc) / n
    scale = float(fnp.mean(fnp.diag(gram))) + 1e-18
    fnp.fill_diagonal(gram, fnp.diag(gram) + ridge_rel * scale)
    cov = ((h - mu).T @ cc) / n
    beta = fnp.linalg.solve(gram.T, cov.T).T
    return mu - beta @ fnp.mean(cc, axis=0)


class Estimator(BaseEstimator):
    """Roberts RQMC + antithetic + all-L1+sq multi-CV + gain blend, L31-only."""

    VERSION = "0.2.15s"

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

        def _one_stream(n_half: int, seed: int):
            x = _rqmc_standard_normal(n_half, width, seed=seed)
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
            for w in weights[1:]:
                h = fnp.maximum(h @ w, 0.0)
            return _multi_cv_mean(h, controls, expectations, _RIDGE_REL)

        seed0 = int(mlp.seed) & 0x7FFFFFFF
        n_half = max(32, n_samples // 4)  # two streams share the sample budget
        mu_rq = 0.5 * (
            _one_stream(n_half, seed0) + _one_stream(n_half, seed0 ^ 0xA5A5A5A5)
        )

        if blend < 1.0:
            mu_g = _GAIN_SCALE * _gain_cov_final(weights)
            mu_final = blend * mu_rq + (1.0 - blend) * mu_g
        else:
            mu_final = mu_rq

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
    print(f"--- RQMC+L1+sq-CV+blend (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))
    if args.baseline:
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(
            _load_baseline(args.baseline)(), mlp, estimator_budget=int(2.72e11)
        )

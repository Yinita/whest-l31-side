"""v0.2.8s — Roberts RQMC + antithetic + all-L1 ridge multi-CV + gain blend.

Advances on v0.2.7s (#321259, online 5.101e-07):

  * Scalar L1-sum CV helped (~1.10× Mini). Using *all* L1 ReLU units as
    controls with exact E[relu]=σ/√(2π) and relative ridge≈1e-4 on the
    control Gram cuts Mini raw MSE further (~1.17× vs L1-sum offline).
  * 256×256 Gram/solve pushes util slightly above 0.1 (~0.107 mean); still
    better adjusted than cutting RQMC samples to stay under the floor.

Pipeline (L31-only):
  1. Cranley–Patterson Roberts lattice → N(0,I), antithetic (z,−z)
  2. Forward; keep L1 activations a₁
  3. μ ← mean(h) − Β̂ (mean(a₁) − E[a₁]) with ridge Gram
  4. ŷ = 0.8·μ + 0.2·(0.992·gain-cov)
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_FLOOR_FRAC = 0.098
_MATMUL_PER_LAYER = 2.0
_PPF_EPS = 1e-6
_BLEND_RQMC = 0.80
_GAIN_SCALE = 0.992
_GAIN_RESERVE = 2.0e9
# Multi-CV Gram/solve costs ~3e9 FLOPs and can push util slightly above 0.1.
# Mini showed adjusted is still better than starving RQMC samples to stay under
# the floor (4.45e-07 @ util≈0.107 vs 4.68e-07 @ util=0.1).
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


def _l1_multi_cv_mean(
    h: fnp.ndarray, a1: fnp.ndarray, e_a1: fnp.ndarray, ridge_rel: float
) -> fnp.ndarray:
    """Multi-CV: mean(h) − Β (mean(a₁) − E[a₁]), ridge on control Gram."""
    n = float(h.shape[0])
    mu = fnp.mean(h, axis=0)
    cc = a1 - e_a1.reshape(1, -1)
    gram = (cc.T @ cc) / n
    scale = float(fnp.mean(fnp.diag(gram))) + 1e-18
    # Ridge on diagonal (keeps Gram SPD; avoid as_symmetric validation cost).
    fnp.fill_diagonal(gram, fnp.diag(gram) + ridge_rel * scale)
    cov = ((h - mu).T @ cc) / n
    # Β = Cov @ Gram^{-1}
    beta = fnp.linalg.solve(gram.T, cov.T).T
    return mu - beta @ fnp.mean(cc, axis=0)


class Estimator(BaseEstimator):
    """Roberts RQMC + antithetic + all-L1 ridge multi-CV + gain blend, L31-only."""

    VERSION = "0.2.8s"

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

        x = a1
        for w in weights[1:]:
            x = fnp.maximum(x @ w, 0.0)
        mu_rq = _l1_multi_cv_mean(x, a1, e_a1, _RIDGE_REL)

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
    print(f"--- RQMC+L1-multiCV+blend (v{Estimator.VERSION}) ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=int(2.72e11))
    if args.baseline:
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(
            _load_baseline(args.baseline)(), mlp, estimator_budget=int(2.72e11)
        )

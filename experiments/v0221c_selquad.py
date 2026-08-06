"""v0.2.21c — v0.2.20s + selective quadratic residual corrector (24-dim).

Keep pooled dual multi-CV + λ=0.85; floor=0.093 fnp-only (0.091 under-sampled; 0.094 over-util). Replace 14-dim linear θ with 14 linear
features + 10 OOS-selected pairwise products (synth 5-fold ratio ~1.059 vs
~1.031 linear; scale=1.0). Full 120-dim quad (opus-0.3.0s) failed online.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_FLOOR_FRAC = 0.093
_MATMUL_PER_LAYER = 2.0
_PPF_EPS = 1e-6
_BLEND_RQMC = 0.85
_GAIN_SCALE = 0.992
_GAIN_RESERVE = 2.0e9
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_RIDGE_REL = 1.0e-4
_CORR_SCALE = 1.0

# OOS-selected pairs among the 14 base features (i, j) with i<=j.
_QUAD_PAIRS = (
    (1, 7),
    (7, 12),
    (2, 7),
    (3, 7),
    (4, 7),
    (0, 7),
    (4, 6),
    (4, 11),
    (4, 10),
    (3, 6),
)

# experiments/corrector_selquad_k10_s1.npz (500 synth, lam=1e-4, scale=1).
_CORR_BETA = fnp.asarray(
    [
        0.0000000000000000e+00,
        -1.1450246940357681e-02,
        -2.2873817026891535e-04,
        1.9576922123293733e-03,
        2.7477849956839711e-04,
        6.3693973539998136e-04,
        -1.9209814775038726e-03,
        -1.2531414507968336e-04,
        1.1626683551539653e-04,
        -1.1298931087767637e-04,
        1.1995087441502186e-03,
        -1.1995087441721217e-03,
        9.7421932125668510e-03,
        -8.1548859587123490e-04,
        4.1411651343322179e-04,
        -1.0710581142046745e-03,
        -8.3419142544140278e-04,
        -3.2925183556611681e-04,
        2.0733802219301232e-04,
        -1.2531414508005280e-04,
        -6.4210874400386867e-04,
        -7.4169708099812621e-04,
        7.4169708098659789e-04,
        -9.6411277479698367e-04,
    ],
    dtype=fnp.float32,
)
_CORR_MU = fnp.asarray(
    [
        1.0000000000000000e+00,
        -1.6919450697344366e-03,
        1.7505097221219209e+01,
        -1.9709719551224514e-01,
        5.3856929751550751e+02,
        6.2968612928291534e-02,
        -4.9266740886245118e-04,
        -2.0958119137058957e-04,
        -7.7920217633534613e-06,
        1.5935822002726579e-04,
        -8.2195423376204017e-05,
        4.6577406579854545e-04,
        -1.3558421003051686e-03,
        6.5239344878863362e-02,
        -3.5046097550184499e-05,
        -3.1188082682031964e-05,
        1.4453585033237159e-03,
        -1.0998838223314909e-03,
        1.0621558715623908e-01,
        -2.0958119137058957e-04,
        2.3166280443600382e-01,
        -1.9685425901093404e-01,
        3.4738986884291091e-02,
        3.9603941779159435e-02,
    ],
    dtype=fnp.float32,
)
_CORR_SD = fnp.asarray(
    [
        1.0000000000000000e+00,
        4.1839089806711964e+00,
        1.5236169754606014e+01,
        1.4234014331527072e+02,
        7.5469571092619458e+02,
        1.1673886633306192e-01,
        2.2307437551872384e-02,
        6.8451456180722275e-03,
        5.8107740775487941e-03,
        2.3064012631304020e-03,
        3.0634113031925117e-03,
        1.7359330718090892e-02,
        3.4696107092211848e+00,
        1.1711788166189546e-01,
        2.5158010704781436e-02,
        2.1259946157434872e-02,
        1.3019859788117644e-01,
        7.6282885407659307e-01,
        4.8047299512561858e+00,
        6.8451456180722275e-03,
        1.7449635074387512e+01,
        1.1385559064029765e+01,
        2.0092163054170182e+00,
        2.7225348605515758e+00,
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

    VERSION = "0.2.21c"

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
        base = [
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
        ]
        quad = [base[i] * base[j] for i, j in _QUAD_PAIRS]
        feats = fnp.stack(base + quad, axis=0)  # (24, width)
        Xs = (feats - _CORR_MU.reshape(-1, 1)) / _CORR_SD.reshape(-1, 1)
        delta = _CORR_SCALE * s * (_CORR_BETA @ Xs)
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

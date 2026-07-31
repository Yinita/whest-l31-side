"""opus-0.3.0s — v0.2.20s base + quadratic residual corrector (120-dim).

Aggressive structural change to the corrector: the 14 hand features
(v0.2.14s–v0.2.20s) gain a per-unit sigma-scale feature (s/mean(s)) and
ALL pairwise products -> 120-dim ridge basis. θ refit on the 500-MLP
synth cache (N_GT=1.5e5) with 5-fold OOS model selection:
synth OOS ratio 1.027 (linear) -> 1.073 (quad), a ~4.5% honest held-out
MSE reduction expected to transfer to hidden. scale=1.0 (no
extrapolation) for transfer safety.

Base RQMC+CV+gain blend, λ=0.85, floor=0.094, reserve=2.0e9 unchanged.
"""

from __future__ import annotations

import argparse
import array
import base64
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

# opus-0.3.0s corrector (experiments/corrector_opus030.npz): 120-dim quad
# basis, ridge θ, 5-fold OOS-selected (lam=1e-3, scale=1.0). Arrays inlined
# as base64(float64) so the estimator is self-contained (grader runs only
# estimator.py + imported modules; no data file ships).
_CORR_SCALE = 1.0
_CORR_B64_BETA = (
    "g4ZH5+z4Hr/py3YrXXJhv839rIy5jFy/ljGac/pFXL+5cTPsctAKP11NtL+ZS0O/vGM/Isc76T7zbBtWo9QmP0nAu6HO9EU/jXuzQgZnQr9uCrE2Y+A3PyafkGtn4De/RYXv6P/lZj/JF8Q4/9AyP+8hJhgRoCy/s/6sjLmMXL90Kppz+kVcv0nCMOxy0Ao/bWWa5S1RFz8+NgqPLFVVP0SUCbzH10e/kwRdoSAMIz++xPG40kUjvx0vadeUuz0/vwfQwhfENr/A/2fmGMQ2P+Nl5Big8/6+sux6nkPcST+IkY4YyyNhv4BIMexy0Ao/nL+Z5S1RFz+4aHpSNExEP4ksgrMh2SQ/dk3rxF5jJD+FeqsOJ29Dv5QacLq8mQ2/fIvv036vQT9uJutdnctBP0b+m4qdy0G/C4fknkRPVz86IBwZTUYcv6Hf+pS2ZT2/iHp6UjRMRD+1wk4QSoc4v4cXi1eJX0I/rQxuf+LPI7/s2A2Kbwkzv0ZHUvfKECM/ILgoMryW/L7hg/+08P32vjmp477w/fY+dk3Z+2iAQT85Tb4pkTZCv0kGK7uCOzM/9kz8U5dJQ7/b+eIq05o2P61KVk4mu9++Qx51xEJfID/fn5vlNboyP5MWYHrrtjC/pu0s9ORfFb/kz6T15F8VPyN2Vrp3omE/48ziyYZ1Nb/pOBGvv1gkP8H3Mpio+FC/E/M9khEUOD8bq2v5eyJBv0ttjCXrdyK/Qf366xL4Bj/v/DR2hbomP2i1Haueuia/0n26E4IASb9r829/RXIQv9IOneox3vk+VprSw4OSJj/EKOr9zV0/PySqm+ys+SM/7URCcfTuMD8lEQiVxhQiP8mpWcoGFSK/G0bEMpgXKL/e0WbRQGgTP3mPLLNGwVG/BCG/eFKkCz8lgb9lMcPVvv3x8JT8L/q+WjL8/+lMKb/1QuP7JU8pPwotSsn86kq/S1Cg70zhQj/KV6SqA2NHv1z7vEdoKhg/b72CTTqAED8yokObt7oPv7FsJ4aqvQ8/plRzsQ92Qz+IVWd75Johv8JDdE22zSq/pEC9UEbysL7xkIRUJeQlv94nsfNo6SU/U7YTb7IQNb+l7lQxVdE9P2HlLi7TsPK+7Ir6m2U0J79mwBN1sTYnP4JpRqbWpjY/aT4qyLwSNr9AAUoeoHMzP1BJDz4ZNye/SCm/+9emNr/v2JX31BI2P6NmBMOjczO/0HkgvJUdZT8qkiSZp35Wv4q7dmufW1g/ZpmEZCjWTz8eIQzk4sU4P0RHueAdsi8/"
)
_CORR_B64_MU = (
    "AAAAAAAAAABtzW4viLhbv95vLg1OgTFAaOBtHHs6yb9kFtvrjdSAQCyVMwW2HrA/L9RVNMskQL91DsC5YHgrv4rNC1FPV+C+CKfNryzjJD98KAKfCowVv7kSQ2Fkhj4/Wq6tXtA2Vr+FzKqUhrOwP31//////+8/3m8uDU6BMUBs4G0cezrJv2QW2+uN1IBAdbOtBLoQKcAQamaRnEkxv90QvNjHsVM/huYdvs5fAr81LPgSnBNhv/hLqc2nfri+qll5waiKKD9/RQuJN2JRv+E0QoCS6ixATo1F1AJjL7/8vMC/SQFpv2QW2+uN1IBAdbOtBLoQKcA1pfRVMMnTQBgG4dbvrrA/xJ04aSHBdT/cZS3gRa5XP/T1fJM0E86+s1/xzi5jaz9EzLXbBqtKP3BqS4bP43K/eAaiQgAQw79QvvqdOM6zPwKVYAWyqTVANaX0VTDJ00BBXrBqjQSGwI0H9bujeiW/KOZ3xfdGpD/Cy/ZCPwVSv3Eb7Myee3+//+lJqy8J9D6J4mEf04J4P0hR5eCqXKG/RkG1u1DJekC6vS44VRoavz2Ia6VAQdO/0s01As07KkGAhlnXzLjJP7dKYHQgp80/DFcp2fEwuz8jLQ52HSQCv5EqZVy325w/Pd8g+U7JoT8N4EM2hTLJv+YsODcQ1CLAZ0e+zTRL0z/QJYth/PSGQEggWtvmA5I/waieqUg/Ir/MSlH08j8EvwCYntGSksq+b8PJO1sdIb8gGpAjE5P5vvxn2+OCHSI/U6BMmdWyMb+jG4t5Ci+SP7W6FNG5bKc/iMWf2mNQQD8MvBNhRVQBP8pPT0mdItM+lohTWdn/Y75hyDt7B9gRPzGxlO5fRzm/SPbtPqO+Wz/4WdqvRb0kv8Nd5+UrWSW/4setw82WCD8yUHw3eX2BPkdlO8EF2YO+ot41XpUw0j66UMya08T5voWZdFz+WQC/m7fHzuh+Bb9kBTakl58sv2DW7e3hswE/2/x2zGVlhD46EZfKmbWmPrIBK5r3FdC+qYIjOmo0YL+9SwCG5GrLvg5W2LsotdW+OCsT7gVr1j53Oj/THm5lPpGPmQABXI6+4GwTypXVwb7H0t0M7oogv33x0SPwzCQ/abzx3uKx4z7myisRrOYLv+7o/J/xSDA/jXavQ2h0/L4gBssRygH/vrevKYxkwzM/cJv79wASV794SZzaySckP86CmsyZ9iU/xy/E/Y4TKEAB4HFVuigwv/gFpAx9iWO/tZYPM3Rnkj87YZewaWWoP8dlqPemVvI/"
)
_CORR_B64_SD = (
    "AAAAAAAA8D8dfnGjUrwQQN6STj7reC5Asj9BdOLKYUBnM+XQkJWHQMG0BViZ4r0/1yXFdsPXlj8kG0B4qgl8PykrPHkMzXc/Y9pHHuXkYj9FYfTJdRhpP+hDqKKlxpE/iz/HQ8PBC0Ao3XYqcPu9P8nApKGld9g/3pJOPut4LkCyP0F04sphQGcz5dCQlYdAYollm/OguECWvm5cakS4P2oQ/R6z3bU/OvKiMAbDmT8w/wKEe1dzPwJ9A+trsmg/MibvuYsXhj+ZtTVhBEyvP8/elpeELSdAMI15aPR8uT+9B1C4O+4VQGcz5dCQlYdAYollm/OguEDZeOy6hELjQDXCWuET67g/C0bPcKQa3T/8zGoVWarAPwyrRINB538/NMx9GuQzeT9Vxrk2MSusP2r0p1Tt89M//YN5NHy1W0Awc8LAJPu6P38y1mymQzlA2XjsuoRC40DK1zc4kpETQT2c4SBs3cc/rei5XMDHBUDn6/QTGGnoP1JouF65wYo///xBJqgIkz/RzuskM37UP0p8HL0dCP0/+ENLWFCfgUDEbo6f9wXNP1DiCxXrAGtAaSMTHA3gQEHzPX6B2WfSPx2U7kgbczFAoNOGIQs4E0AYY1qGfQeiP7TxvdtvK6g/GnLqAOASAECNRLv/Z8UmQBz1A3XIhbJAviBGecDC1z+m3K0h5s+TQJPTdcKM26Q/hjrM6eCgaj+gM4OrTzdRP7JWnhR71V0/XPySJHG0Rj/DQnIFw+Q/Pw+SX2tGl2Y/lQEgcM7gtz++9MnCI+WkP+fKTOHsMrY/QFE75z8WUj9tgvqMJKsmP8Y0kVEsUSQ/504WTzx6Dz+stDyM+OsjPxCLEDqROEw/1N38vDdDsj9f4rEWigNrP7brMIDHgpY/MYXvHxvJFT/0DRerjD0IP/pQXhfhl/Q+LSlYJYQK+T7xnghoN7shPzJupIErxZU/bUF6KLxgUT+TSCsyt2B4P0ACNdDYyyE/HEISShn9CD8o2MUMa8v3PjxQ9ksw2SA/ZrDnh2Bicj8yqp65wzReP4MpcWEb9nE/gP6KpzY++z7pQf91RWfiPuH5fJ0JDAo/F9NGHrb0Zj/hC0qHr95GP54M/m4/+Fo/a+mrpBFZ9j5AjnJEt6UfP7ezX9RLxYI/I/aoQBokQD+uQxbyvZpnP+Cvi+5SakY/wChVMX+Xqj8jqsjNtt1mP0pEwGlDuJA/+c/C29bgIUDp6rQ8P9m4P7LSnLVcmBFAWp+6EMb0pD/pyTnf+FW2P5KJb+Q3ve8/"
)


def _decode(b64: str) -> fnp.ndarray:
    # stdlib only — grader has no numpy (only flopscope.numpy as fnp).
    buf = array.array("d")
    buf.frombytes(base64.b64decode(b64))
    return fnp.asarray(buf, dtype=fnp.float32)


_CORR_BETA = _decode(_CORR_B64_BETA)  # (120,)
_CORR_MU = _decode(_CORR_B64_MU)  # (120,)
_CORR_SD = _decode(_CORR_B64_SD)  # (120,)


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

    VERSION = "opus-0.3.0s"

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
        sig_scale = s / (fnp.mean(s) + 1e-12)
        # 15 base features (14 legacy + sigma-scale); order MUST match the
        # trainer (experiments/train_corrector_opus030.py::build_feats).
        base_rows = [
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
            sig_scale,
        ]
        k = len(base_rows)  # 15
        rows = list(base_rows)  # linear part
        for i in range(1, k):
            for j in range(i, k):
                rows.append(base_rows[i] * base_rows[j])  # pairwise products
        feats = fnp.stack(rows, axis=0)  # (120, width)
        # standardize per feature, apply θ
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

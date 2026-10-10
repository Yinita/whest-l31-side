"""Probe: RQMC to L-1, then exact E[relu] under diagonal-Gaussian preacts + CV blend.

Uses same budget/floor as v0.2.12s. Compares on Mini GT (BudgetContext util).
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, load_dataset

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e12", ROOT / "experiments" / "v0212s_floor094.py")
e12 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(e12)

B = int(2.72e11)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _relu_mean_gauss(mu: fnp.ndarray, var: fnp.ndarray) -> fnp.ndarray:
    var = fnp.maximum(var, 1e-12)
    sigma = fnp.sqrt(var)
    alpha = mu / sigma
    return mu * flops.stats.norm.cdf(alpha) + sigma * flops.stats.norm.pdf(alpha)


def predict_analytic_last(mlp: MLP, budget: int, mix: float) -> fnp.ndarray:
    """mix: weight on analytic-last vs full-depth CV blend (0=base, 1=analytic)."""
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    blend = e12._BLEND_RQMC
    reserve = e12._GAIN_RESERVE if blend < 1.0 else 0.0
    flops_per_sample = depth * e12._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(budget) * e12._FLOOR_FRAC - reserve, flops_per_sample)
    n_samples = int(max(64, min(target / flops_per_sample, 200_000)))
    n_half = max(32, n_samples // 2)
    x = e12._rqmc_standard_normal(n_half, width, seed=int(mlp.seed) & 0x7FFFFFFF)
    x = fnp.concatenate([x, -x], axis=0)

    w0 = weights[0]
    a1 = fnp.maximum(x @ w0, 0.0)
    col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
    e_a1 = col_norm * _INV_SQRT_2PI
    q1 = fnp.sum(a1 * a1, axis=1)
    e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
    controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
    expectations = fnp.concatenate([e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0)

    h = a1
    for w in weights[1:-1]:
        h = fnp.maximum(h @ w, 0.0)
    # preacts at last layer
    pre = h @ weights[-1]
    mu_pre = e12._multi_cv_mean(pre, controls, expectations, e12._RIDGE_REL)
    # diagonal var from samples (CV not applied to 2nd moment for speed)
    var_pre = fnp.maximum(fnp.var(pre, axis=0), 1e-12)
    mu_an = _relu_mean_gauss(mu_pre, var_pre)

    # baseline full-depth CV
    post = fnp.maximum(pre, 0.0)
    mu_rq = e12._multi_cv_mean(post, controls, expectations, e12._RIDGE_REL)
    mu_g = e12._GAIN_SCALE * e12._gain_cov_final(weights)
    mu_base = blend * mu_rq + (1.0 - blend) * mu_g
    mu_an_blend = blend * mu_an + (1.0 - blend) * mu_g
    return (1.0 - mix) * mu_base + mix * mu_an_blend


def main() -> None:
    ds = load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")
    mlps, Y = [], []
    for i in range(100):
        r = ds[i]
        W = [fnp.asarray(w, dtype=fnp.float32) for w in r["weights"]]
        mlps.append(MLP(width=256, depth=len(W), weights=W, seed=int(r["mlp_seed"])))
        Y.append(np.asarray(r["final_means"], np.float64))
    Y = np.stack(Y)

    for mix in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mses, fls = [], []
        t0 = time.time()
        for i in range(100):
            with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
                pred = predict_analytic_last(mlps[i], B, mix)
                fls.append(float(ctx.flops_used))
            mses.append(float(np.mean((np.asarray(pred, float) - Y[i]) ** 2)))
        raw = float(np.mean(mses))
        util = float(np.mean(fls)) / B
        adj = raw * max(0.1, util)
        print(
            f"mix={mix:.2f} raw={raw:.4e} adj={adj:.4e} util={util:.4f} t={time.time()-t0:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

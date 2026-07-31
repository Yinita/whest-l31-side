"""Probe pooled dual-stream multi-CV vs mean-of-streams (v0.2.17s)."""
from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, load_dataset

ROOT = Path("/home/sti/Desktop/projects/whest-l31-side")
SPEC = importlib.util.spec_from_file_location(
    "e17", ROOT / "experiments" / "v0217s_dual_corr_retrain.py"
)
e17 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(e17)

B = int(2.72e11)
INV = e17._INV_SQRT_2PI


def _apply_corr(mu_cv, mu_g, pred, zpre, alpha_g, phi_g, sig_g):
    width = pred.shape[0]
    zc = zpre - zpre.mean(0)
    m2 = np.maximum((zc * zc).mean(0), 1e-12)
    m3 = (zc**3).mean(0)
    m4 = (zc**4).mean(0)
    s = np.sqrt(m2)
    skew = m3 / (m2 * s)
    kurt = m4 / (m2 * m2) - 3.0
    alpha_s = zpre.mean(0) / s
    phi_s = np.exp(-0.5 * alpha_s * alpha_s) / math.sqrt(2 * math.pi)
    st = -(skew / 6.0) * phi_s
    kt = (kurt / 24.0) * (alpha_s**2 - 1.0) * phi_s
    gap = mu_cv - mu_g
    glayer = np.full(width, float(gap.mean()))
    a = np.asarray(alpha_g, float)
    feats = np.stack(
        [
            np.ones(width),
            a,
            a**2,
            a**3,
            a**4,
            np.asarray(phi_g, float),
            gap / (np.asarray(sig_g, float) + 1e-8),
            glayer / (np.asarray(sig_g, float) + 1e-8),
            st,
            kt,
            (mu_cv - pred) / (s + 1e-8),
            (mu_g - pred) / (s + 1e-8),
            alpha_s,
            phi_s,
        ],
        0,
    )
    beta = np.asarray(e17._CORR_BETA, float)
    mu = np.asarray(e17._CORR_MU, float)
    sd = np.asarray(e17._CORR_SD, float)
    Xs = (feats - mu.reshape(-1, 1)) / sd.reshape(-1, 1)
    return pred + s * (beta @ Xs)


def forward_streams(mlp, mode: str, blend=0.85):
    """mode: mean_anti | pool_anti | pool_noanti | mean_noanti"""
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    reserve = e17._GAIN_RESERVE
    flops_per = depth * e17._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * e17._FLOOR_FRAC - reserve, flops_per)
    n_samples = int(max(64, min(target / flops_per, 200_000)))
    seed0 = int(mlp.seed) & 0x7FFFFFFF
    anti = "noanti" not in mode
    pool = mode.startswith("pool")

    if anti:
        # 2 streams × (n_half unique + anti) = n_samples forwards
        n_lat = max(16, n_samples // 4)
    else:
        # 2 streams × n_lat unique = n_samples forwards
        n_lat = max(16, n_samples // 2)

    parts_mu, parts_a1, parts_h, parts_z = [], [], [], []
    for j, xor in enumerate((0, 0xA5A5A5A5)):
        seed = seed0 if j == 0 else (seed0 ^ xor)
        x = e17._rqmc_standard_normal(n_lat, width, seed=seed)
        if anti:
            x = fnp.concatenate([x, -x], axis=0)
        w0 = weights[0]
        a1 = fnp.maximum(x @ w0, 0.0)
        h = a1
        for w in weights[1:-1]:
            h = fnp.maximum(h @ w, 0.0)
        zpre = h @ weights[-1]
        last = fnp.maximum(zpre, 0.0)
        if not pool:
            col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
            e_a1 = col_norm * INV
            q1 = fnp.sum(a1 * a1, axis=1)
            e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
            controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
            expectations = fnp.concatenate(
                [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0
            )
            mu_cv = e17._multi_cv_mean(last, controls, expectations, e17._RIDGE_REL)
            parts_mu.append(np.asarray(mu_cv, float))
        else:
            parts_a1.append(a1)
            parts_h.append(last)
        parts_z.append(np.asarray(zpre, float))

    if pool:
        a1 = fnp.concatenate(parts_a1, axis=0)
        last = fnp.concatenate(parts_h, axis=0)
        w0 = weights[0]
        col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
        e_a1 = col_norm * INV
        q1 = fnp.sum(a1 * a1, axis=1)
        e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
        controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
        expectations = fnp.concatenate(
            [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0
        )
        mu_cv = np.asarray(
            e17._multi_cv_mean(last, controls, expectations, e17._RIDGE_REL), float
        )
    else:
        mu_cv = np.mean(np.stack(parts_mu), axis=0)

    zpre = np.concatenate(parts_z, 0)
    mu_g_raw, alpha_g, phi_g, sig_g = e17._gain_and_last_stats(weights)
    mu_g = e17._GAIN_SCALE * np.asarray(mu_g_raw, float)
    pred = blend * mu_cv + (1.0 - blend) * mu_g
    return _apply_corr(mu_cv, mu_g, pred, zpre, alpha_g, phi_g, sig_g)


def main():
    ds = load_dataset(
        "aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini"
    )
    mlps, Y = [], []
    for i in range(100):
        r = ds[i]
        W = [fnp.asarray(w, dtype=fnp.float32) for w in r["weights"]]
        mlps.append(MLP(width=256, depth=len(W), weights=W, seed=int(r["mlp_seed"])))
        Y.append(np.asarray(r["final_means"], np.float64))
    Y = np.stack(Y)
    print("loaded", flush=True)

    t0 = time.time()
    for mode in ("mean_anti", "pool_anti", "pool_noanti", "mean_noanti"):
        mses = []
        for i, mlp in enumerate(mlps):
            with flops.BudgetContext(flop_budget=B, quiet=True):
                pred = forward_streams(mlp, mode)
            mses.append(float(np.mean((pred - Y[i]) ** 2)))
        raw = float(np.mean(mses))
        print(
            f"{mode:12s} raw={raw:.4e} adj≈{raw*0.1001:.4e} t={time.time()-t0:.0f}s",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

"""Probe ridge_rel and k=3 pooled streams on v0.2.20s base+corrector."""
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
SPEC = importlib.util.spec_from_file_location("e20", ROOT / "experiments" / "v0220s_pooled_corr.py")
e = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(e)

B = int(2.72e11)
INV = e._INV_SQRT_2PI
XOR = (0, 0xA5A5A5A5, 0x3C3C3C3C)


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
    beta = np.asarray(e._CORR_BETA, float)
    mu = np.asarray(e._CORR_MU, float)
    sd = np.asarray(e._CORR_SD, float)
    Xs = (feats - mu.reshape(-1, 1)) / sd.reshape(-1, 1)
    return pred + s * (beta @ Xs)


def predict(mlp, k=2, ridge=None, blend=0.85):
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    reserve = e._GAIN_RESERVE
    flops_per = depth * e._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * e._FLOOR_FRAC - reserve, flops_per)
    n_samples = int(max(64, min(target / flops_per, 200_000)))
    # k streams × (n_lat unique + anti) ≈ n_samples
    n_lat = max(16, n_samples // (2 * k))
    seed0 = int(mlp.seed) & 0x7FFFFFFF
    ridge = e._RIDGE_REL if ridge is None else ridge

    parts_a1, parts_last, parts_z = [], [], []
    for j in range(k):
        seed = seed0 if j == 0 else (seed0 ^ XOR[j % len(XOR)] ^ (j * 0x11111111))
        x = e._rqmc_standard_normal(n_lat, width, seed=seed)
        x = fnp.concatenate([x, -x], axis=0)
        a1 = fnp.maximum(x @ weights[0], 0.0)
        h = a1
        for w in weights[1:-1]:
            h = fnp.maximum(h @ w, 0.0)
        zpre = h @ weights[-1]
        last = fnp.maximum(zpre, 0.0)
        parts_a1.append(a1)
        parts_last.append(last)
        parts_z.append(np.asarray(zpre, float))

    a1 = fnp.concatenate(parts_a1, axis=0)
    last = fnp.concatenate(parts_last, axis=0)
    zpre = np.concatenate(parts_z, 0)
    w0 = weights[0]
    col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
    e_a1 = col_norm * INV
    q1 = fnp.sum(a1 * a1, axis=1)
    e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
    controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
    expectations = fnp.concatenate(
        [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0
    )
    mu_cv = np.asarray(e._multi_cv_mean(last, controls, expectations, ridge), float)
    mu_g_raw, alpha_g, phi_g, sig_g = e._gain_and_last_stats(weights)
    mu_g = e._GAIN_SCALE * np.asarray(mu_g_raw, float)
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
    print("loaded", flush=True)

    cfgs = [
        ("k2_r1e-4", dict(k=2, ridge=1e-4)),
        ("k2_r3e-5", dict(k=2, ridge=3e-5)),
        ("k2_r3e-4", dict(k=2, ridge=3e-4)),
        ("k3_r1e-4", dict(k=3, ridge=1e-4)),
        ("k2_r1e-4_l88", dict(k=2, ridge=1e-4, blend=0.88)),
    ]
    t0 = time.time()
    for name, kw in cfgs:
        mses, fls = [], []
        for i, mlp in enumerate(mlps):
            with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
                pred = predict(mlp, **kw)
                fls.append(float(ctx.flops_used))
            mses.append(float(np.mean((pred - Y[i]) ** 2)))
        raw = float(np.mean(mses))
        util = float(np.mean(fls)) / B
        print(
            f"{name:14s} raw={raw:.4e} adj={raw*max(0.1,util):.4e} util={util:.4f} t={time.time()-t0:.0f}s",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

"""Structural probes on v0.2.20s: pool/mean mix, gain scale, stream disagreement."""
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


def _corr(mu_cv, mu_g, pred, zpre, alpha_g, phi_g, sig_g, extra=None):
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
    feats = [
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
    ]
    beta = np.asarray(e._CORR_BETA, float)
    mu = np.asarray(e._CORR_MU, float)
    sd = np.asarray(e._CORR_SD, float)
    # If extra features provided, ignore old θ and return base only for those modes
    if extra is None:
        Xs = (np.stack(feats, 0) - mu.reshape(-1, 1)) / sd.reshape(-1, 1)
        return pred + s * (beta @ Xs), s
    return pred, s


def _cv(last, a1, w0, ridge):
    col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
    e_a1 = col_norm * INV
    q1 = fnp.sum(a1 * a1, axis=1)
    e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
    controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
    expectations = fnp.concatenate(
        [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0
    )
    return np.asarray(e._multi_cv_mean(last, controls, expectations, ridge), float)


def predict(mlp, mode="base", blend=0.85, gscale=0.992, pool_w=1.0, ridge=1e-4):
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    reserve = e._GAIN_RESERVE
    flops_per = depth * e._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * e._FLOOR_FRAC - reserve, flops_per)
    n_samples = int(max(64, min(target / flops_per, 200_000)))
    n_half = max(32, n_samples // 4)
    seed0 = int(mlp.seed) & 0x7FFFFFFF

    def stream(seed):
        x = e._rqmc_standard_normal(n_half, width, seed=seed)
        x = fnp.concatenate([x, -x], axis=0)
        a1 = fnp.maximum(x @ weights[0], 0.0)
        h = a1
        for w in weights[1:-1]:
            h = fnp.maximum(h @ w, 0.0)
        z = h @ weights[-1]
        last = fnp.maximum(z, 0.0)
        return a1, last, np.asarray(z, float)

    a1a, la, z1 = stream(seed0)
    a1b, lb, z2 = stream(seed0 ^ 0xA5A5A5A5)
    mu1 = _cv(la, a1a, weights[0], ridge)
    mu2 = _cv(lb, a1b, weights[0], ridge)
    mu_mean = 0.5 * (mu1 + mu2)
    a1 = fnp.concatenate([a1a, a1b], 0)
    last = fnp.concatenate([la, lb], 0)
    zpre = np.concatenate([z1, z2], 0)
    mu_pool = _cv(last, a1, weights[0], ridge)
    mu_cv = pool_w * mu_pool + (1.0 - pool_w) * mu_mean

    if mode == "shrink_disagree":
        # shrink toward gain where streams disagree
        disagree = np.abs(mu1 - mu2)
        med = float(np.median(disagree)) + 1e-12
        w = 1.0 / (1.0 + disagree / med)
        # will apply after gain
        shrink_w = w
    else:
        shrink_w = None

    mu_g_raw, alpha_g, phi_g, sig_g = e._gain_and_last_stats(weights)
    mu_g = gscale * np.asarray(mu_g_raw, float)
    if shrink_w is not None:
        mu_cv = shrink_w * mu_cv + (1.0 - shrink_w) * mu_g
    pred = blend * mu_cv + (1.0 - blend) * mu_g
    out, _ = _corr(mu_cv, mu_g, pred, zpre, alpha_g, phi_g, sig_g)
    return out


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
        ("base", dict()),
        ("pool0.85", dict(pool_w=0.85)),
        ("pool0.70", dict(pool_w=0.70)),
        ("pool0.50", dict(pool_w=0.50)),
        ("g990", dict(gscale=0.990)),
        ("g994", dict(gscale=0.994)),
        ("g996", dict(gscale=0.996)),
        ("blend0.88", dict(blend=0.88)),
        ("blend0.82", dict(blend=0.82)),
        ("shrink_dis", dict(mode="shrink_disagree")),
        ("ridge3e5", dict(ridge=3e-5)),
    ]
    t0 = time.time()
    for name, kw in cfgs:
        mses, fls = [], []
        for i, mlp in enumerate(mlps):
            with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
                p = predict(mlp, **kw)
                fls.append(float(ctx.flops_used))
            mses.append(float(np.mean((p - Y[i]) ** 2)))
        raw = float(np.mean(mses))
        util = float(np.mean(fls)) / B
        print(
            f"{name:12s} raw={raw:.4e} adj={raw*max(0.1,util):.4e} util={util:.4f} t={time.time()-t0:.0f}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

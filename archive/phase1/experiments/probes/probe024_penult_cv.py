"""Probe penultimate-layer controls vs L1-only for pooled multi-CV (v0.2.20s base)."""
from __future__ import annotations

import importlib.util
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


def predict(mlp, mode="l1"):
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
    blend = e._BLEND_RQMC

    parts_a1, parts_pen, parts_last, parts_z = [], [], [], []
    for j, xor in enumerate((0, 0xA5A5A5A5)):
        seed = seed0 if j == 0 else (seed0 ^ xor)
        x = e._rqmc_standard_normal(n_half, width, seed=seed)
        x = fnp.concatenate([x, -x], axis=0)
        a1 = fnp.maximum(x @ weights[0], 0.0)
        h = a1
        for w in weights[1:-1]:
            h = fnp.maximum(h @ w, 0.0)
        pen = h  # a_{L-1}
        zpre = h @ weights[-1]
        last = fnp.maximum(zpre, 0.0)
        parts_a1.append(a1)
        parts_pen.append(pen)
        parts_last.append(last)
        parts_z.append(np.asarray(zpre, float))

    a1 = fnp.concatenate(parts_a1, 0)
    pen = fnp.concatenate(parts_pen, 0)
    last = fnp.concatenate(parts_last, 0)
    zpre = np.concatenate(parts_z, 0)

    w0 = weights[0]
    col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, 0), 1e-30))
    e_a1 = col_norm * INV
    q1 = fnp.sum(a1 * a1, 1)
    e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))

    if mode == "l1":
        controls = fnp.concatenate([a1, q1.reshape(-1, 1)], 1)
        expectations = fnp.concatenate([e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], 0)
    elif mode == "pen":
        # unknown E[pen] — use sample mean as plug-in (biased CV / regression adjust)
        # Better: use gain-propagated E[pen] from layers 0..L-2
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = fnp.eye(width, dtype=fnp.float32)
        for w in weights[:-1]:
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
        e_pen = mu
        qpen = fnp.sum(pen * pen, 1)
        # E[||pen||^2] ≈ trace(E[pen pen^T]) ≈ sum var + ||mu||^2 under diagonal approx
        e_qpen = float(fnp.sum(fnp.diag(cov) + mu * mu))
        controls = fnp.concatenate([pen, qpen.reshape(-1, 1)], 1)
        expectations = fnp.concatenate([e_pen, fnp.asarray([e_qpen], dtype=e_pen.dtype)], 0)
    elif mode == "l1_pen":
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = fnp.eye(width, dtype=fnp.float32)
        for w in weights[:-1]:
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
        e_pen = mu
        q1 = fnp.sum(a1 * a1, 1)
        controls = fnp.concatenate([a1, pen, q1.reshape(-1, 1)], 1)
        expectations = fnp.concatenate(
            [e_a1, e_pen, fnp.asarray([e_q1], dtype=e_a1.dtype)], 0
        )
    else:
        raise ValueError(mode)

    mu_cv = np.asarray(e._multi_cv_mean(last, controls, expectations, e._RIDGE_REL), float)
    mu_g_raw, alpha_g, phi_g, sig_g = e._gain_and_last_stats(weights)
    mu_g = e._GAIN_SCALE * np.asarray(mu_g_raw, float)
    pred = blend * mu_cv + (1.0 - blend) * mu_g

    # apply v0.2.20s linear corrector for fair compare
    width = pred.shape[0]
    zc = zpre - zpre.mean(0)
    m2 = np.maximum((zc * zc).mean(0), 1e-12)
    m3 = (zc**3).mean(0)
    m4 = (zc**4).mean(0)
    s = np.sqrt(m2)
    skew = m3 / (m2 * s)
    kurt = m4 / (m2 * m2) - 3.0
    alpha_s = zpre.mean(0) / s
    phi_s = np.exp(-0.5 * alpha_s * alpha_s) * INV
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
    t0 = time.time()
    for mode in ("l1", "pen", "l1_pen"):
        mses, fls = [], []
        for i, mlp in enumerate(mlps):
            with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
                p = predict(mlp, mode)
                fls.append(float(ctx.flops_used))
            mses.append(float(np.mean((p - Y[i]) ** 2)))
        raw = float(np.mean(mses))
        util = float(np.mean(fls)) / B
        print(
            f"{mode:8s} raw={raw:.4e} adj={raw*max(0.1,util):.4e} util={util:.4f} t={time.time()-t0:.0f}s",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

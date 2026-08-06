"""Probe CV estimation noise on v0.2.20s base: cross-fit, ridge, blocks, PCA."""
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


def _multi_cv(h, controls, expectations, ridge_rel):
    return np.asarray(e._multi_cv_mean(h, controls, expectations, ridge_rel), float)


def _multi_cv_xfit(h, controls, expectations, ridge_rel, k=2):
    """K-fold cross-fitted CV: fit β on other folds, apply on holdout."""
    n = h.shape[0]
    idx = np.arange(n)
    # deterministic folds for reproducibility
    folds = np.array_split(idx, k)
    out = np.zeros(h.shape[1], dtype=np.float64)
    counts = np.zeros(k, dtype=np.float64)
    exp = np.asarray(expectations, float).reshape(1, -1)
    for i, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        h_tr = fnp.asarray(np.asarray(h)[tr], dtype=fnp.float32)
        c_tr = fnp.asarray(np.asarray(controls)[tr], dtype=fnp.float32)
        # fit β on train
        ntr = float(h_tr.shape[0])
        mu_tr = fnp.mean(h_tr, axis=0)
        cc = c_tr - expectations.reshape(1, -1)
        gram = (cc.T @ cc) / ntr
        scale = float(fnp.mean(fnp.diag(gram))) + 1e-18
        fnp.fill_diagonal(gram, fnp.diag(gram) + ridge_rel * scale)
        cov = ((h_tr - mu_tr).T @ cc) / ntr
        beta = fnp.linalg.solve(gram.T, cov.T).T  # (width, p)
        # apply on test: mean(h_te) - β @ mean(c_te - E)
        h_te = np.asarray(h)[te]
        c_te = np.asarray(controls)[te]
        mu_te = h_te.mean(0)
        cc_te = c_te.mean(0) - exp.ravel()
        out += (mu_te - np.asarray(beta, float) @ cc_te) * len(te)
        counts[i] = len(te)
    return out / n


def _controls_full(a1, w0):
    col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
    e_a1 = col_norm * INV
    q1 = fnp.sum(a1 * a1, axis=1)
    e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
    controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
    expectations = fnp.concatenate(
        [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0
    )
    return controls, expectations, col_norm


def _controls_topk(a1, w0, k):
    col_norm = np.asarray(
        fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30)), float
    )
    top = np.argsort(-col_norm)[:k]
    a = np.asarray(a1)[:, top]
    e_a = col_norm[top] * INV
    q1 = (np.asarray(a1) ** 2).sum(1, keepdims=True)
    e_q1 = 0.5 * float((col_norm**2).sum())
    controls = fnp.asarray(np.concatenate([a, q1], 1), dtype=fnp.float32)
    expectations = fnp.asarray(np.concatenate([e_a, [e_q1]]), dtype=fnp.float32)
    return controls, expectations


def _controls_blocks(a1, w0, n_blocks):
    """Sum L1 units into n_blocks by sorted ‖w‖ groups + Σa²."""
    col_norm = np.asarray(
        fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30)), float
    )
    order = np.argsort(-col_norm)
    a = np.asarray(a1)[:, order]
    cn = col_norm[order]
    width = a.shape[1]
    edges = np.linspace(0, width, n_blocks + 1, dtype=int)
    blocks = []
    e_blocks = []
    for i in range(n_blocks):
        sl = slice(edges[i], edges[i + 1])
        blocks.append(a[:, sl].sum(1))
        e_blocks.append(float((cn[sl] * INV).sum()))
    q1 = (np.asarray(a1) ** 2).sum(1)
    e_q1 = 0.5 * float((col_norm**2).sum())
    controls = fnp.asarray(np.column_stack(blocks + [q1]), dtype=fnp.float32)
    expectations = fnp.asarray(e_blocks + [e_q1], dtype=fnp.float32)
    return controls, expectations


def _controls_pca(a1, w0, k, seed=0):
    """Random Gaussian projection of L1 (budget-cheap proxy for PCA) + Σa²."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a1, float)
    col_norm = np.asarray(
        fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30)), float
    )
    P = rng.standard_normal((a.shape[1], k)).astype(np.float64) / math.sqrt(k)
    # Exact E[a1 @ P] = E[a1] @ P
    e_a = (col_norm * INV) @ P
    proj = a @ P
    q1 = (a**2).sum(1, keepdims=True)
    e_q1 = 0.5 * float((col_norm**2).sum())
    controls = fnp.asarray(np.concatenate([proj, q1], 1), dtype=fnp.float32)
    expectations = fnp.asarray(np.concatenate([e_a, [e_q1]]), dtype=fnp.float32)
    return controls, expectations


def predict(mlp, mode="base", ridge=1e-4, k=64, n_blocks=16):
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
    a1 = fnp.concatenate([a1a, a1b], 0)
    last = fnp.concatenate([la, lb], 0)
    zpre = np.concatenate([z1, z2], 0)
    w0 = weights[0]

    if mode == "base":
        c, exp, _ = _controls_full(a1, w0)
        mu_cv = _multi_cv(last, c, exp, ridge)
    elif mode == "xfit2":
        c, exp, _ = _controls_full(a1, w0)
        mu_cv = _multi_cv_xfit(last, c, exp, ridge, k=2)
    elif mode == "xfit4":
        c, exp, _ = _controls_full(a1, w0)
        mu_cv = _multi_cv_xfit(last, c, exp, ridge, k=4)
    elif mode.startswith("topk"):
        kk = int(mode[4:])
        c, exp = _controls_topk(a1, w0, kk)
        mu_cv = _multi_cv(last, c, exp, ridge)
    elif mode.startswith("blk"):
        nb = int(mode[3:])
        c, exp = _controls_blocks(a1, w0, nb)
        mu_cv = _multi_cv(last, c, exp, ridge)
    elif mode.startswith("rp"):
        kk = int(mode[2:])
        c, exp = _controls_pca(a1, w0, kk, seed=seed0)
        mu_cv = _multi_cv(last, c, exp, ridge)
    elif mode == "nosq":
        col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
        e_a1 = col_norm * INV
        mu_cv = _multi_cv(last, a1, e_a1, ridge)
    else:
        raise ValueError(mode)

    mu_g_raw, alpha_g, phi_g, sig_g = e._gain_and_last_stats(weights)
    mu_g = e._GAIN_SCALE * np.asarray(mu_g_raw, float)
    pred = e._BLEND_RQMC * mu_cv + (1.0 - e._BLEND_RQMC) * mu_g
    return _apply_corr(mu_cv, mu_g, pred, zpre, alpha_g, phi_g, sig_g)


def main():
    import sys

    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    ds = load_dataset(
        "aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini"
    )
    mlps, Y = [], []
    for i in range(n_mlps):
        r = ds[i]
        W = [fnp.asarray(w, dtype=fnp.float32) for w in r["weights"]]
        mlps.append(MLP(width=256, depth=len(W), weights=W, seed=int(r["mlp_seed"])))
        Y.append(np.asarray(r["final_means"], np.float64))
    print(f"loaded n={n_mlps}", flush=True)

    cfgs = [
        ("base", {}),
        ("ridge3e5", {"ridge": 3e-5}),
        ("ridge3e4", {"ridge": 3e-4}),
        ("ridge1e3", {"ridge": 1e-3}),
        ("xfit2", {"mode": "xfit2"}),
        ("xfit4", {"mode": "xfit4"}),
        ("topk64", {"mode": "topk64"}),
        ("topk128", {"mode": "topk128"}),
        ("blk8", {"mode": "blk8"}),
        ("blk16", {"mode": "blk16"}),
        ("blk32", {"mode": "blk32"}),
        ("rp32", {"mode": "rp32"}),
        ("rp64", {"mode": "rp64"}),
        ("rp128", {"mode": "rp128"}),
        ("nosq", {"mode": "nosq"}),
    ]
    t0 = time.time()
    for name, kw in cfgs:
        mses, fls = [], []
        for mlp, y in zip(mlps, Y):
            with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
                p = predict(mlp, **kw)
                fls.append(float(ctx.flops_used))
            mses.append(float(np.mean((p - y) ** 2)))
        raw = float(np.mean(mses))
        util = float(np.mean(fls)) / B
        print(
            f"{name:12s} raw={raw:.4e} adj={raw * max(0.1, util):.4e} "
            f"util={util:.4f} t={time.time() - t0:.0f}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

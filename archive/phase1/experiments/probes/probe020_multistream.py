"""Probe multi-stream RQMC (k=1..4) ± corrector; also λ sweep at k=2.

Uses dual-retrained θ from corrector_v0217.npz. Does NOT mutate estimator.py.
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

ROOT = Path("/home/sti/Desktop/projects/whest-l31-side")
SPEC = importlib.util.spec_from_file_location(
    "e17", ROOT / "experiments" / "v0217s_dual_corr_retrain.py"
)
e17 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(e17)

B = int(2.72e11)
INV = e17._INV_SQRT_2PI
XOR_KEYS = [0x00000000, 0xA5A5A5A5, 0x3C3C3C3C, 0x5A5A5A5A, 0xC3C3C3C3, 0xF0F0F0F0]


def predict_k(mlp: MLP, k: int, blend: float, use_corr: bool):
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    reserve = e17._GAIN_RESERVE if blend < 1.0 else 0.0
    flops_per = depth * e17._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * e17._FLOOR_FRAC - reserve, flops_per)
    n_samples = int(max(64, min(target / flops_per, 200_000)))
    # each stream: n_half lattice → 2*n_half with antithetic
    n_half = max(16, n_samples // (2 * k))
    seed0 = int(mlp.seed) & 0x7FFFFFFF

    mus = []
    zpres = []
    for j in range(k):
        seed = seed0 if j == 0 else (seed0 ^ XOR_KEYS[j])
        x = e17._rqmc_standard_normal(n_half, width, seed=seed)
        x = fnp.concatenate([x, -x], axis=0)
        w0 = weights[0]
        a1 = fnp.maximum(x @ w0, 0.0)
        col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
        e_a1 = col_norm * INV
        q1 = fnp.sum(a1 * a1, axis=1)
        e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
        controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
        expectations = fnp.concatenate(
            [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0
        )
        h = a1
        for w in weights[1:-1]:
            h = fnp.maximum(h @ w, 0.0)
        zpre = h @ weights[-1]
        last = fnp.maximum(zpre, 0.0)
        mu_cv = e17._multi_cv_mean(last, controls, expectations, e17._RIDGE_REL)
        mus.append(np.asarray(mu_cv, float))
        zpres.append(np.asarray(zpre, float))

    mu_cv = np.mean(np.stack(mus), axis=0)
    zpre = np.concatenate(zpres, 0)

    mu_g_raw, alpha_g, phi_g, sig_g = e17._gain_and_last_stats(weights)
    mu_g = e17._GAIN_SCALE * np.asarray(mu_g_raw, float)
    pred = blend * mu_cv + (1.0 - blend) * mu_g

    if not use_corr:
        return pred

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
    phi_g = np.asarray(phi_g, float)
    sig_g = np.asarray(sig_g, float)
    feats = np.stack(
        [
            np.ones(width),
            a,
            a**2,
            a**3,
            a**4,
            phi_g,
            gap / (sig_g + 1e-8),
            glayer / (sig_g + 1e-8),
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

    cfgs = []
    for k in (1, 2, 3, 4):
        for corr in (False, True):
            cfgs.append((k, 0.85, corr, f"k{k}_c{int(corr)}"))
    for lam in (0.80, 0.82, 0.85, 0.88, 0.90, 0.93):
        cfgs.append((2, lam, True, f"k2_l{lam:.2f}_c1"))
        cfgs.append((3, lam, True, f"k3_l{lam:.2f}_c1"))

    t0 = time.time()
    results = []
    for k, lam, corr, tag in cfgs:
        mses = []
        for i, mlp in enumerate(mlps):
            with flops.BudgetContext(flop_budget=B, quiet=True):
                pred = predict_k(mlp, k=k, blend=lam, use_corr=corr)
            mses.append(float(np.mean((pred - Y[i]) ** 2)))
        raw = float(np.mean(mses))
        adj = raw * 0.1001
        results.append((adj, raw, tag, k, lam, corr))
        print(
            f"{tag:16s} raw={raw:.4e} adj≈{adj:.4e} t={time.time()-t0:.0f}s",
            flush=True,
        )

    results.sort()
    print("=== TOP ===", flush=True)
    for adj, raw, tag, k, lam, corr in results[:15]:
        print(f"  {tag:16s} raw={raw:.4e} adj≈{adj:.4e}", flush=True)
    print(
        "baseline k2_c1≈",
        next(r for r in results if r[2] == "k2_c1"),
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

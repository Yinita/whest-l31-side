"""Probe: radiant-style moment shrink + Gauss final (+ optional Edgeworth).

Does NOT use L1 multi-CV mean of final activations. Instead:
  RQMC → layer L-1 samples → shrink (μ,σ) toward analytic gain chain
  → Gaussian ReLU mean (+ sample skew/kurt).
Also compare blends with v0.2.12s base.
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
spec = importlib.util.spec_from_file_location("e12", ROOT / "experiments/v0212s_floor094.py")
e12 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(e12)

B = int(2.72e11)
INV = e12._INV_SQRT_2PI


def gain_chain_last_pre(weights):
    """Return (mu_pre, sigma_pre, mu_post_scaled) at last layer from gain-cov."""
    width = weights[0].shape[0]
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = fnp.eye(width, dtype=fnp.float32)
    for li, w in enumerate(weights):
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sigma = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma
        phi = flops.stats.norm.pdf(alpha)
        Phi = flops.stats.norm.cdf(alpha)
        mu_post = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sigma * phi
        var_post = fnp.maximum(ez2 - mu_post * mu_post, 0.0)
        gain = fnp.where(sigma > 1e-12, Phi, fnp.zeros_like(Phi))
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        mu = mu_post
        if li == len(weights) - 1:
            return (
                np.asarray(mu_pre, np.float64),
                np.asarray(sigma, np.float64),
                e12._GAIN_SCALE * np.asarray(mu_post, np.float64),
            )
    raise RuntimeError("empty weights")


def gauss_relu(mu, sig):
    alpha = mu / sig
    phi = np.asarray(
        flops.stats.norm.pdf(fnp.asarray(alpha.astype(np.float32))), np.float64
    )
    Phi = np.asarray(
        flops.stats.norm.cdf(fnp.asarray(alpha.astype(np.float32))), np.float64
    )
    return mu * Phi + sig * phi, alpha, phi


def edgeworth(mu, sig, m3, k4, alpha, phi):
    m2 = sig * sig
    d_skew = -(m3 / (6.0 * m2)) * phi
    d_kurt = (k4 / (24.0 * (sig ** 3))) * (alpha * alpha - 1.0) * phi
    return d_skew, d_kurt


def run_one(mlp: MLP, anti: bool = True, meanlam=0.95, shrink=0.9):
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    blend = e12._BLEND_RQMC
    reserve = e12._GAIN_RESERVE if blend < 1.0 else 0.0
    flops_per = depth * e12._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * e12._FLOOR_FRAC - reserve, flops_per)
    n_samples = int(max(64, min(target / flops_per, 200_000)))

    if anti:
        n_half = max(32, n_samples // 2)
        x = e12._rqmc_standard_normal(n_half, width, seed=int(mlp.seed) & 0x7FFFFFFF)
        x = fnp.concatenate([x, -x], axis=0)
    else:
        x = e12._rqmc_standard_normal(n_samples, width, seed=int(mlp.seed) & 0x7FFFFFFF)

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

    z = np.asarray(h @ weights[-1], np.float64)
    a = np.maximum(z, 0.0)
    mu_emp = z.mean(0)
    zc = z - mu_emp
    m2 = np.maximum((zc * zc).mean(0), 1e-12)
    m3 = (zc ** 3).mean(0)
    m4 = (zc ** 4).mean(0)
    k4 = m4 - 3.0 * m2 * m2
    sig_emp = np.sqrt(m2)

    mu_a, sig_a, mu_gain = gain_chain_last_pre(weights)
    # radiant shrink
    mu_s = meanlam * mu_emp + (1.0 - meanlam) * mu_a
    # shrink std via variance mix
    var_s = shrink * m2 + (1.0 - shrink) * (sig_a * sig_a)
    sig_s = np.sqrt(np.maximum(var_s, 1e-12))

    g_emp, a_emp, p_emp = gauss_relu(mu_emp, sig_emp)
    g_shr, a_shr, p_shr = gauss_relu(mu_s, sig_s)
    ds_e, dk_e = edgeworth(mu_emp, sig_emp, m3, k4, a_emp, p_emp)
    ds_s, dk_s = edgeworth(mu_s, sig_s, m3, k4, a_shr, p_shr)

    last = fnp.asarray(a.astype(np.float32))
    mu_cv = np.asarray(
        e12._multi_cv_mean(last, controls, expectations, e12._RIDGE_REL), np.float64
    )
    base = blend * mu_cv + (1.0 - blend) * mu_gain

    out = {
        "base": base,
        "hyb_gauss": g_shr,
        "hyb_skew": g_shr + ds_s,
        "hyb_both": g_shr + ds_s + dk_s,
        "emp_gauss": g_emp,
        "emp_skew": g_emp + ds_e,
        "emp_both": g_emp + ds_e + dk_e,
        "mc": a.mean(0),
    }
    for k in ("hyb_gauss", "hyb_skew", "hyb_both", "emp_skew", "emp_both"):
        out[f"mix70_{k}"] = 0.70 * base + 0.30 * out[k]
        out[f"mix85_{k}"] = 0.85 * base + 0.15 * out[k]
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
    Y = np.stack(Y)
    print("loaded", flush=True)

    configs = [
        ("anti+shrink", True, 0.95, 0.9),
        ("noanti+shrink", False, 0.95, 0.9),
        ("anti+noshrink", True, 1.0, 1.0),
        ("anti+m0.90", True, 0.90, 0.9),
    ]
    t0 = time.time()
    for name, anti, ml, sh in configs:
        print(f"=== {name} meanlam={ml} shrink={sh} ===", flush=True)
        acc = None
        for i, mlp in enumerate(mlps):
            with flops.BudgetContext(flop_budget=B, quiet=True):
                preds = run_one(mlp, anti=anti, meanlam=ml, shrink=sh)
            if acc is None:
                acc = {k: [] for k in preds}
            for k, v in preds.items():
                acc[k].append(float(np.mean((v - Y[i]) ** 2)))
            if i % 25 == 0:
                print(f"  {i} t={time.time()-t0:.0f}s", flush=True)
        rows = sorted((float(np.mean(v)), k) for k, v in acc.items())
        for raw, k in rows[:10]:
            print(f"  {k:18s} raw={raw:.4e} adj≈{raw*0.1005:.4e}", flush=True)
        print("  base=", next(r for r in rows if r[1] == "base"), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

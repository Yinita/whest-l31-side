"""Probe: last-layer Gauss + sample Edgeworth vs v0.2.12s CV mean.

Inspired by radiant-allomancer v8a (online ~4.47e-7): RQMC prefix →
Gaussian-conditional final layer + sample skew/kurt Edgeworth.

We already have full RQMC forwards; keep L1+sq CV mean as baseline and
compare / blend with last-preact closures from the same samples.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, load_dataset

ROOT = Path("/home/sti/Desktop/projects/whest-l31-side")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e12", ROOT / "experiments/v0212s_floor094.py")
e12 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(e12)

B = int(2.72e11)
INV = e12._INV_SQRT_2PI
SQRT2PI = math.sqrt(2.0 * math.pi)


def last_layer_preds(h_pre: np.ndarray, W_last: np.ndarray):
    """h_pre: (N,n) post-ReLU of layer L-1. Return several L31 mean estimates."""
    z = h_pre @ W_last  # (N,n) pre-activations
    a = np.maximum(z, 0.0)
    mu_mc = a.mean(0)

    mu = z.mean(0)
    zc = z - mu
    m2 = np.maximum((zc * zc).mean(0), 1e-12)
    m3 = (zc ** 3).mean(0)
    m4 = (zc ** 4).mean(0)
    sig = np.sqrt(m2)
    alpha = mu / sig
    # φ, Φ via flopscope for consistency
    phi = np.asarray(
        flops.stats.norm.pdf(fnp.asarray(alpha.astype(np.float32))), np.float64
    )
    Phi = np.asarray(
        flops.stats.norm.cdf(fnp.asarray(alpha.astype(np.float32))), np.float64
    )
    mu_g = mu * Phi + sig * phi

    # unit-consistent Edgeworth (κ raw central moments)
    # δ_skew = -(κ3/(6 σ²)) φ(α) ; δ_kurt = (κ4/(24 σ³))(α²-1) φ(α)
    k4 = m4 - 3.0 * m2 * m2
    d_skew = -(m3 / (6.0 * m2)) * phi  # = -κ3/(6σ²) φ
    d_kurt = (k4 / (24.0 * (sig ** 3))) * (alpha * alpha - 1.0) * phi
    # radiant literal skew (likely typo): -κ3/(6σ³) φ
    d_skew_lit = -(m3 / (6.0 * (sig ** 3))) * phi

    return {
        "mc": mu_mc,
        "gauss": mu_g,
        "skew": mu_g + d_skew,
        "skew_lit": mu_g + d_skew_lit,
        "both": mu_g + d_skew + d_kurt,
        "both_lit": mu_g + d_skew_lit + d_kurt,
        "mc_skew": mu_mc + d_skew,  # Edgeworth on top of sample mean (odd)
    }


def run_one(mlp: MLP, anti: bool = True):
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
    # h is post-ReLU of layer depth-2 (0-index: after weights[0..depth-2])
    # Wait: weights has depth matrices. After w0 we have L1. After weights[1:-1]
    # we've applied weights[1]..weights[depth-2], so h is activation before last weight.
    # For depth=32, weights[0..31], weights[1:-1]=weights[1..30], so after that
    # we've done 31 matmuls → activation at layer 31 index? 
    # Start x; a1 = relu(x@W0) = layer1; then for w in W[1:-1]: h=relu(h@w)
    # W[1:-1] has depth-2 matrices → total layers applied: 1+(depth-2)=depth-1
    # So h = post-ReLU of layer (depth-1) = input to last weight. Correct.

    W_last = np.asarray(weights[-1], np.float32)
    h_np = np.asarray(h, np.float64)
    preds = last_layer_preds(h_np, W_last)

    # full forward final for CV (same as estimator)
    last = fnp.maximum(h @ weights[-1], 0.0)
    mu_cv = np.asarray(
        e12._multi_cv_mean(last, controls, expectations, e12._RIDGE_REL), np.float64
    )
    mu_g = e12._GAIN_SCALE * np.asarray(
        e12._gain_cov_final(weights), np.float64
    )
    base = blend * mu_cv + (1.0 - blend) * mu_g
    preds["base"] = base
    preds["cv"] = mu_cv
    # blends
    for k in ("gauss", "skew", "both", "skew_lit", "both_lit"):
        preds[f"b70_{k}"] = 0.70 * base + 0.30 * preds[k]
        preds[f"b85_{k}"] = 0.85 * base + 0.15 * preds[k]
        preds[f"b50_{k}"] = 0.50 * base + 0.50 * preds[k]
    return preds


def main():
    ds = load_dataset(
        "aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini"
    )
    Y = []
    mlps = []
    for i in range(100):
        r = ds[i]
        W = [fnp.asarray(w, dtype=fnp.float32) for w in r["weights"]]
        mlps.append(MLP(width=256, depth=len(W), weights=W, seed=int(r["mlp_seed"])))
        Y.append(np.asarray(r["final_means"], np.float64))
    Y = np.stack(Y)
    print("loaded", flush=True)

    keys = None
    acc = None
    t0 = time.time()
    for anti in (True, False):
        print(f"=== anti={anti} ===", flush=True)
        acc = None
        for i, mlp in enumerate(mlps):
            with flops.BudgetContext(flop_budget=B, quiet=True):
                preds = run_one(mlp, anti=anti)
            if keys is None:
                keys = sorted(preds.keys())
                acc = {k: [] for k in keys}
            for k in keys:
                acc[k].append(float(np.mean((preds[k] - Y[i]) ** 2)))
            if i % 25 == 0:
                print(f"  {i} t={time.time()-t0:.0f}s", flush=True)
        rows = []
        for k in keys:
            raw = float(np.mean(acc[k]))
            # util ≈ 0.1 for all (same FLOPs)
            rows.append((raw, k))
        rows.sort()
        print("top:", flush=True)
        for raw, k in rows[:12]:
            print(f"  {k:16s} raw={raw:.4e} adj≈{raw*0.1005:.4e}", flush=True)
        print("base:", next(r for r in rows if r[1] == "base"), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

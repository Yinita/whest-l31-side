"""Faithful radiant hybrid: full emp cov at L-1 → shrink → Gauss last layer."""
from __future__ import annotations

import importlib.util
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


def analytic_Lminus1(weights):
    """Gain-cov through all but last layer; return mu, cov at post-ReLU L-1."""
    width = weights[0].shape[0]
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
    return np.asarray(mu, np.float64), np.asarray(cov, np.float64)


def gauss_last(mu_s, cov_s, W_last):
    """Gaussian-conditional final ReLU mean from shrunk L-1 moments."""
    # μ_pre = W^T μ ; var_pre_i = (W^T cov W)_{ii}
    mu_pre = W_last.T @ mu_s
    # cov_pre diag via (W * (cov @ W)).sum(0) or einsum
    cw = cov_s @ W_last  # (n,n)
    var_pre = np.maximum(np.sum(W_last * cw, axis=0), 1e-12)
    sig = np.sqrt(var_pre)
    alpha = mu_pre / sig
    phi = np.asarray(
        flops.stats.norm.pdf(fnp.asarray(alpha.astype(np.float32))), np.float64
    )
    Phi = np.asarray(
        flops.stats.norm.cdf(fnp.asarray(alpha.astype(np.float32))), np.float64
    )
    return mu_pre * Phi + sig * phi, mu_pre, sig, alpha, phi


def run_one(mlp, anti=True, meanlam=0.95, covshrink=0.9):
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    blend = e12._BLEND_RQMC
    reserve = e12._GAIN_RESERVE
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
    H = np.asarray(h, np.float64)
    n = float(H.shape[0])
    mu_emp = H.mean(0)
    Hc = H - mu_emp
    cov_emp = (Hc.T @ Hc) / n

    mu_a, cov_a = analytic_Lminus1(weights)
    mu_s = meanlam * mu_emp + (1.0 - meanlam) * mu_a
    cov_s = covshrink * cov_emp + (1.0 - covshrink) * cov_a
    # symmetrize
    cov_s = 0.5 * (cov_s + cov_s.T)

    W_last = np.asarray(weights[-1], np.float64)
    hyb, mu_pre, sig, alpha, phi = gauss_last(mu_s, cov_s, W_last)

    # sample Edgeworth on actual last preacts
    z = H @ W_last
    zc = z - z.mean(0)
    m2 = np.maximum((zc * zc).mean(0), 1e-12)
    m3 = (zc ** 3).mean(0)
    m4 = (zc ** 4).mean(0)
    k4 = m4 - 3.0 * m2 * m2
    d_skew = -(m3 / (6.0 * m2)) * phi
    d_kurt = (k4 / (24.0 * (sig ** 3))) * (alpha * alpha - 1.0) * phi

    last = fnp.maximum(h @ weights[-1], 0.0)
    mu_cv = np.asarray(
        e12._multi_cv_mean(last, controls, expectations, e12._RIDGE_REL), np.float64
    )
    mu_g = e12._GAIN_SCALE * np.asarray(e12._gain_cov_final(weights), np.float64)
    base = blend * mu_cv + (1.0 - blend) * mu_g

    return {
        "base": base,
        "hyb": hyb,
        "hyb_skew": hyb + d_skew,
        "hyb_both": hyb + d_skew + d_kurt,
        "mix85": 0.85 * base + 0.15 * hyb,
        "mix85s": 0.85 * base + 0.15 * (hyb + d_skew),
        "mix70": 0.70 * base + 0.30 * hyb,
        "mix70s": 0.70 * base + 0.30 * (hyb + d_skew),
        "mix50b": 0.50 * base + 0.50 * (hyb + d_skew + d_kurt),
    }


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
    for name, anti, ml, cs in [
        ("anti s0.9", True, 0.95, 0.9),
        ("noanti s0.9", False, 0.95, 0.9),
        ("anti s0.7", True, 0.95, 0.7),
    ]:
        print(f"=== {name} ===", flush=True)
        acc = None
        for i, mlp in enumerate(mlps):
            with flops.BudgetContext(flop_budget=B, quiet=True):
                preds = run_one(mlp, anti=anti, meanlam=ml, covshrink=cs)
            if acc is None:
                acc = {k: [] for k in preds}
            for k, v in preds.items():
                acc[k].append(float(np.mean((v - Y[i]) ** 2)))
            if i % 25 == 0:
                print(f"  {i} t={time.time()-t0:.0f}s", flush=True)
        rows = sorted((float(np.mean(vs)), k) for k, vs in acc.items())
        for raw, k in rows:
            print(f"  {k:10s} raw={raw:.4e} adj≈{raw*0.1005:.4e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

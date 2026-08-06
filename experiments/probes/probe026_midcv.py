"""Mid-layer / last-preact control variables on v0.2.20s."""
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
SPEC = importlib.util.spec_from_file_location("e20", ROOT / "experiments" / "v0220s_pooled_corr.py")
e = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(e)

B = int(2.72e11)
INV = e._INV_SQRT_2PI


def _corr(mu_cv, mu_g, pred, zpre, alpha_g, phi_g, sig_g):
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
    feats = np.stack([
        np.ones(width), a, a**2, a**3, a**4, np.asarray(phi_g, float),
        gap / (np.asarray(sig_g, float) + 1e-8), glayer / (np.asarray(sig_g, float) + 1e-8),
        st, kt, (mu_cv - pred) / (s + 1e-8), (mu_g - pred) / (s + 1e-8), alpha_s, phi_s,
    ], 0)
    beta = np.asarray(e._CORR_BETA, float)
    mu = np.asarray(e._CORR_MU, float)
    sd = np.asarray(e._CORR_SD, float)
    return pred + s * (beta @ ((feats - mu.reshape(-1, 1)) / sd.reshape(-1, 1)))


def gain_forward_to(weights, upto):
    """Gain-cov mean/cov after `upto` layers (0-indexed exclusive end)."""
    width = weights[0].shape[0]
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = fnp.eye(width, dtype=fnp.float32)
    for w in weights[:upto]:
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
    return mu, cov


def predict(mlp, mode="base"):
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    flops_per = depth * e._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * e._FLOOR_FRAC - e._GAIN_RESERVE, flops_per)
    n_samples = int(max(64, min(target / flops_per, 200_000)))
    n_half = max(32, n_samples // 4)
    seed0 = int(mlp.seed) & 0x7FFFFFFF

    def stream(seed):
        x = e._rqmc_standard_normal(n_half, width, seed=seed)
        x = fnp.concatenate([x, -x], axis=0)
        acts = []
        h = x
        for li, w in enumerate(weights[:-1]):
            h = fnp.maximum(h @ w, 0.0)
            acts.append(h)
        z = h @ weights[-1]
        last = fnp.maximum(z, 0.0)
        return acts, last, np.asarray(z, float)

    A1, L1, z1 = stream(seed0)
    A2, L2, z2 = stream(seed0 ^ 0xA5A5A5A5)
    # pool
    acts = [fnp.concatenate([A1[i], A2[i]], 0) for i in range(len(A1))]
    last = fnp.concatenate([L1, L2], 0)
    zpre = np.concatenate([z1, z2], 0)
    a1 = acts[0]
    w0 = weights[0]
    col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
    e_a1 = col_norm * INV
    q1 = fnp.sum(a1 * a1, axis=1)
    e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
    controls = [a1, q1.reshape(-1, 1)]
    expectations = [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)]

    if mode == "sumL16":
        # scalar sum of layer-16 activations; E via gain-cov
        li = 15  # 0-indexed after ReLU of layer 16 (weights[:16])
        mu_g, _ = gain_forward_to(weights, 16)
        s = fnp.sum(acts[li], axis=1).reshape(-1, 1)
        e_s = float(fnp.sum(mu_g))
        controls.append(s)
        expectations.append(fnp.asarray([e_s], dtype=fnp.float32))
    elif mode == "sumL8":
        li = 7
        mu_g, _ = gain_forward_to(weights, 8)
        s = fnp.sum(acts[li], axis=1).reshape(-1, 1)
        e_s = float(fnp.sum(mu_g))
        controls.append(s)
        expectations.append(fnp.asarray([e_s], dtype=fnp.float32))
    elif mode == "sumL24":
        li = 23
        mu_g, _ = gain_forward_to(weights, 24)
        s = fnp.sum(acts[li], axis=1).reshape(-1, 1)
        e_s = float(fnp.sum(mu_g))
        controls.append(s)
        expectations.append(fnp.asarray([e_s], dtype=fnp.float32))
    elif mode == "sumLastH":
        # sum of last hidden (pre final linear)
        li = depth - 2
        mu_g, _ = gain_forward_to(weights, depth - 1)
        s = fnp.sum(acts[li], axis=1).reshape(-1, 1)
        e_s = float(fnp.sum(mu_g))
        controls.append(s)
        expectations.append(fnp.asarray([e_s], dtype=fnp.float32))
    elif mode != "base":
        raise ValueError(mode)

    C = fnp.concatenate(controls, axis=1)
    E = fnp.concatenate(expectations, axis=0)
    mu_cv = np.asarray(e._multi_cv_mean(last, C, E, e._RIDGE_REL), float)
    mu_g_raw, alpha_g, phi_g, sig_g = e._gain_and_last_stats(weights)
    mu_g = e._GAIN_SCALE * np.asarray(mu_g_raw, float)
    pred = e._BLEND_RQMC * mu_cv + (1.0 - e._BLEND_RQMC) * mu_g
    return _corr(mu_cv, mu_g, pred, zpre, alpha_g, phi_g, sig_g)


def main():
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
    cfgs = ["base", "sumL8", "sumL16", "sumL24", "sumLastH"]
    t0 = time.time()
    for name in cfgs:
        mses, fls = [], []
        for mlp, y in zip(mlps, Y):
            with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
                p = predict(mlp, mode=name)
                fls.append(float(ctx.flops_used))
            mses.append(float(np.mean((p - y) ** 2)))
        raw = float(np.mean(mses))
        util = float(np.mean(fls)) / B
        print(f"{name:12s} raw={raw:.4e} adj={raw*max(0.1,util):.4e} util={util:.4f} t={time.time()-t0:.0f}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()

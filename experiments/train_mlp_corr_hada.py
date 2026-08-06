#!/usr/bin/env python3
"""P3: tiny shared MLP bias corrector on Hadamard (independent-stream targets).

Target = Y - μ_B; features from stream A. 5-fold Mini OOS.
If ratio ≳1.05 → ship hyb-0.3.2s; else document and stop Mini-fit.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
OUT = ROOT / "experiments" / "mlp_corr_hada_v032.npz"
B = int(2.72e11)
F, H = 10, 32


def _load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def _extract(hada, weights, seed: int):
    width = 256
    depth = 32
    n2 = float(width * width)
    flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
    n_forward = int(max(2 * width, min(target / flops_per, hada._MAX_SAMPLES)))
    n_frames = max(1, n_forward // (2 * width))
    e_r = hada._chi_mean_radius(width)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(n_frames, width, seed=seed, e_r=e_r)
        for w in weights[:-1]:
            x = fnp.maximum(x @ w, 0.0)
        z = x @ weights[-1]
        last = fnp.maximum(z, 0.0)
        mu = np.asarray(fnp.mean(last, axis=0), np.float64)
        z = np.asarray(z, np.float64)
        last = np.asarray(last, np.float64)
    zc = z - z.mean(0)
    m2 = np.maximum((zc * zc).mean(0), 1e-12)
    s = np.sqrt(m2)
    alpha = z.mean(0) / s
    phi = np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
    act = (last > 0).mean(0)
    skew = (zc**3).mean(0) / (m2 * s)
    kurt = (zc**4).mean(0) / (m2 * m2) - 3.0
    feats = np.stack(
        [
            np.ones(width),
            mu / (s + 1e-8),
            s / (s.mean() + 1e-8),
            alpha,
            phi,
            act,
            skew,
            kurt,
            (mu - mu.mean()) / (s + 1e-8),
            np.log(s + 1e-8),
        ],
        0,
    )
    return mu, feats.astype(np.float64), s.astype(np.float64)


def _init_params(rng: np.random.Generator):
    # Xavier-ish
    W1 = rng.normal(0, 1.0 / math.sqrt(F), (H, F))
    b1 = np.zeros(H)
    W2 = rng.normal(0, 1.0 / math.sqrt(H), (1, H))
    b2 = np.zeros(1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def _forward(params, X):
    """X: (N, F) → (N,) prediction of residual/σ."""
    h = np.tanh(X @ params["W1"].T + params["b1"])
    return (h @ params["W2"].T + params["b2"]).reshape(-1)


def _train_mlp(X, y, rng, steps=800, lr=0.05, l2=1e-3, batch=4096):
    params = _init_params(rng)
    n = X.shape[0]
    for t in range(steps):
        idx = rng.integers(0, n, size=min(batch, n))
        xb, yb = X[idx], y[idx]
        # forward
        h_pre = xb @ params["W1"].T + params["b1"]
        h = np.tanh(h_pre)
        pred = (h @ params["W2"].T + params["b2"]).reshape(-1)
        err = pred - yb
        # grads
        dpred = (2.0 / len(yb)) * err
        dW2 = dpred.reshape(-1, 1) * h
        db2 = dpred
        dh = dpred.reshape(-1, 1) * params["W2"]  # (B,H)
        dh_pre = dh * (1.0 - h * h)
        dW1 = dh_pre.T @ xb
        db1 = dh_pre.sum(0)
        # mean over batch already in dpred scale; dW2 need mean
        gW2 = dW2.mean(0, keepdims=True) + l2 * params["W2"]
        gb2 = np.array([db2.mean()]) + l2 * params["b2"]
        gW1 = (dh_pre.T @ xb) / len(yb) + l2 * params["W1"]
        gb1 = dh_pre.mean(0) + l2 * params["b1"]
        # adam-ish momentum
        decay = 0.95
        if t == 0:
            m = {k: np.zeros_like(params[k]) for k in params}
        for k, g in [("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2)]:
            m[k] = decay * m[k] + (1 - decay) * g
            params[k] = params[k] - lr * m[k]
        if (t + 1) % 200 == 0:
            lr *= 0.5
    return params


def main() -> None:
    hada = _load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    n = len(ds)
    width = 256
    Bases = np.zeros((n, width))
    Feats = np.zeros((n, F, width))
    Sig = np.zeros((n, width))
    T = np.zeros((n, width))
    Ys = np.zeros((n, width))

    for i in range(n):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"][-1], np.float64)
        s0 = int(ex["mlp_seed"]) & 0x7FFFFFFF
        muA, fA, sA = _extract(hada, weights, s0)
        muB, _, _ = _extract(hada, weights, s0 ^ 0xA5A5A5A5)
        Bases[i], Feats[i], Sig[i], Ys[i] = muA, fA, sA, Y
        T[i] = Y - muB
        if i % 10 == 0:
            print(f"[{i}] base={np.mean((muA-Y)**2):.3e}", flush=True)

    base = float(np.mean((Bases - Ys) ** 2))
    print(f"base mse={base:.4e}")
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)

    best = None
    for l2 in [1e-2, 1e-3, 1e-4]:
        oos = np.zeros_like(Bases)
        for fi, val in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            mu = Feats[tr].mean((0, 2))
            sd = Feats[tr].std((0, 2)) + 1e-8
            sd[0] = 1.0
            # flatten neurons
            def flat(subset):
                X = (Feats[subset] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
                X = X.transpose(0, 2, 1).reshape(-1, F)
                y = (T[subset] / (Sig[subset] + 1e-8)).reshape(-1)
                return X, y

            Xtr, ytr = flat(tr)
            params = _train_mlp(
                Xtr, ytr, np.random.default_rng(int(rng.integers(1 << 30))), l2=l2
            )
            Xva = (Feats[val] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            for j, vi in enumerate(val):
                Xn = Xva[j].T  # (W,F)
                d = _forward(params, Xn) * (Sig[vi] + 1e-8)
                oos[vi] = Bases[vi] + d
        for sh in [1.0, 0.5, 0.25, 0.1]:
            pred = Bases + sh * (oos - Bases)
            m = float(np.mean((pred - Ys) ** 2))
            print(f"l2={l2:.0e} shrink={sh:.2f} oos={m:.4e} r={base/m:.3f}")
            if best is None or m < best[0]:
                best = (m, l2, sh, oos.copy())

    # refit on all with best l2
    l2, sh = best[1], best[2]
    mu = Feats.mean((0, 2))
    sd = Feats.std((0, 2)) + 1e-8
    sd[0] = 1.0
    X = (Feats - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
    X = X.transpose(0, 2, 1).reshape(-1, F)
    y = (T / (Sig + 1e-8)).reshape(-1)
    params = _train_mlp(X, y, rng, l2=l2)
    np.savez(
        OUT,
        W1=params["W1"],
        b1=params["b1"],
        W2=params["W2"],
        b2=params["b2"],
        feat_mu=mu,
        feat_sd=sd,
        l2=l2,
        shrink=sh,
        oos_mse=best[0],
        base_mse=base,
    )
    print(f"wrote {OUT} best oos={best[0]:.4e} r={base/best[0]:.3f} shrink={sh}")


if __name__ == "__main__":
    main()

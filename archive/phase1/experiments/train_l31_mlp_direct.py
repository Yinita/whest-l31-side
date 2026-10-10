#!/usr/bin/env python3
"""Direct L31 residual MLP on Full feat cache (no path recenter).

Reuses l30_feat_cache_v3.npz: predict δ=(Y31−B31)/σ from Fe.
Gate: Full OOS R²>0.02 → Mini additive ŷ=B31+α·δ̂.
"""
from __future__ import annotations

import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
FEAT_CACHE = ROOT / "experiments" / "results" / "full_cache" / "l30_feat_cache_v3.npz"
OUT = ROOT / "experiments" / "l31_mlp_direct_v1.npz"
LOG = ROOT / "experiments" / "results" / "train_l31_mlp_direct_v1.log"
MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
B = int(2.72e11)
W = 256
H = 128  # deeper than L30 attempt
SEED_PROJ = 42


def _estimator_seed(raw_mlp_seed: int) -> int:
    ss = fnp.random.SeedSequence(int(raw_mlp_seed)).spawn(3)
    return int(ss[2].generate_state(1)[0]) & 0x7FFFFFFF


def _load_hada():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def _init_mlp(rng, F):
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / F), (H, F)),
        "b1": np.zeros(H),
        "W2": rng.normal(0, np.sqrt(2.0 / H), (H // 2, H)),
        "b2": np.zeros(H // 2),
        "W3": rng.normal(0, np.sqrt(2.0 / (H // 2)), (1, H // 2)),
        "b3": np.zeros(1),
    }


def _forward(p, X):
    h1 = np.maximum(0.0, X @ p["W1"].T + p["b1"])
    h2 = np.maximum(0.0, h1 @ p["W2"].T + p["b2"])
    return (h2 @ p["W3"].T + p["b3"]).reshape(-1)


def _train_mlp(X, y, rng, steps=2000, lr=3e-3, l2=1e-4, batch=8192, val=None):
    F = X.shape[1]
    p = _init_mlp(rng, F)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(vv) for k, vv in p.items()}
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    best_state, best_val = None, 1e9
    n = X.shape[0]
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, size=min(batch, n))
        xb, yb = X[idx], y[idx]
        h1_pre = xb @ p["W1"].T + p["b1"]
        h1 = np.maximum(0.0, h1_pre)
        h2_pre = h1 @ p["W2"].T + p["b2"]
        h2 = np.maximum(0.0, h2_pre)
        pred = (h2 @ p["W3"].T + p["b3"]).reshape(-1)
        err = pred - yb
        dpred = (2.0 / len(yb)) * err
        gW3 = (dpred.reshape(-1, 1) * h2).mean(0, keepdims=True) + l2 * p["W3"]
        gb3 = np.array([dpred.mean()]) + l2 * p["b3"]
        dh2 = dpred.reshape(-1, 1) * p["W3"]
        dh2_pre = dh2 * (h2_pre > 0)
        gW2 = (dh2_pre.T @ h1) / len(yb) + l2 * p["W2"]
        gb2 = dh2_pre.mean(0) + l2 * p["b2"]
        dh1 = dh2_pre @ p["W2"]
        dh1_pre = dh1 * (h1_pre > 0)
        gW1 = (dh1_pre.T @ xb) / len(yb) + l2 * p["W1"]
        gb1 = dh1_pre.mean(0) + l2 * p["b1"]
        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2, "W3": gW3, "b3": gb3}
        for k in p:
            g = grads[k]
            gn = np.linalg.norm(g)
            if gn > 5.0:
                g = g * (5.0 / gn)
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * (g * g)
            mh = m[k] / (1 - b1**t)
            vh = v[k] / (1 - b2**t)
            p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)
        if val is not None and t % 50 == 0:
            mv = float(np.mean((_forward(p, val[0]) - val[1]) ** 2))
            if mv < best_val:
                best_val = mv
                best_state = {k: vv.copy() for k, vv in p.items()}
        if t % 400 == 0:
            lr *= 0.5
    return best_state if best_state is not None else p


def main():
    c = np.load(FEAT_CACHE)
    B31 = c["B31"]
    Y31 = c["Y31"]
    Sig = c["Sig"]
    Fe = c["Fe"]  # (n, F, W)
    A30, A31 = c["A30"], c["A31"]
    n, F, _ = Fe.shape
    print(f"cache n={n} F={F} H={H}", flush=True)

    T = Y31 - B31
    base31 = float(np.mean(T**2))
    print(f"base L31 residual MSE={base31:.4e} (Hadamard err)", flush=True)

    # Augment features with B31 normalized
    Fe2 = np.concatenate(
        [
            Fe,
            (B31 / (Sig + 1e-8)).reshape(n, 1, W),
            np.log(np.maximum(np.abs(B31), 1e-8)).reshape(n, 1, W),
        ],
        axis=1,
    )
    F2 = Fe2.shape[1]
    print(f"aug F={F2}", flush=True)

    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(n), 5)
    best = None
    for l2 in [1e-3, 1e-4, 1e-5]:
        oos = np.zeros_like(T)
        t0 = time.time()
        for fi, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            mu = Fe2[tr].mean((0, 2))
            sd = Fe2[tr].std((0, 2)) + 1e-8
            sd[0] = 1.0
            Xtr = ((Fe2[tr] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)).transpose(
                0, 2, 1
            ).reshape(-1, F2)
            ytr = (T[tr] / (Sig[tr] + 1e-8)).reshape(-1)
            ntr = Xtr.shape[0]
            perm = rng.permutation(ntr)
            nv = max(1000, ntr // 10)
            print(f"l2={l2:.0e} fold={fi+1}/5 …", flush=True)
            params = _train_mlp(
                Xtr[perm[nv:]],
                ytr[perm[nv:]],
                np.random.default_rng(int(rng.integers(1 << 30))),
                l2=l2,
                val=(Xtr[perm[:nv]], ytr[perm[:nv]]),
            )
            Xte = (Fe2[te] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            for j, i in enumerate(te):
                oos[i] = _forward(params, Xte[j].T) * (Sig[i] + 1e-8)
            print(f"  fold done t={time.time()-t0:.0f}s", flush=True)
        for sh in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]:
            m = float(np.mean((T - sh * oos) ** 2))
            r2 = 1.0 - m / base31
            print(f"l2={l2:.0e} sh={sh:.2f} δMSE={m:.4e} R²={r2:.4f}", flush=True)
            if best is None or m < best[0]:
                best = (m, l2, sh, oos.copy())

    l2, sh = best[1], best[2]
    r2 = 1 - best[0] / base31
    print(f"BEST R²={r2:.4f} l2={l2} sh={sh}", flush=True)

    # refit all
    mu = Fe2.mean((0, 2))
    sd = Fe2.std((0, 2)) + 1e-8
    sd[0] = 1.0
    X = ((Fe2 - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)).transpose(0, 2, 1).reshape(
        -1, F2
    )
    y = (T / (Sig + 1e-8)).reshape(-1)
    perm = rng.permutation(X.shape[0])
    nv = max(2000, X.shape[0] // 10)
    params = _train_mlp(
        X[perm[nv:]], y[perm[nv:]], rng, l2=l2, val=(X[perm[:nv]], y[perm[:nv]])
    )
    np.savez(
        OUT,
        **params,
        feat_mu=mu,
        feat_sd=sd,
        A30=A30,
        A31=A31,
        l2=l2,
        shrink=sh,
        n_full=n,
        oos_r2=r2,
        F2=F2,
    )
    print(f"wrote {OUT}", flush=True)

    if r2 < 0.02:
        print("R² too low; skip Mini", flush=True)
        return

    # Mini additive
    hada = _load_hada()
    # need extract like train_l30 — import path
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tl", ROOT / "experiments" / "train_l30_mlp_full.py"
    )
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)

    ds = Dataset.load_from_disk(str(MINI))
    eb, ec = [], []
    for i in range(len(ds)):
        ex = ds[i]
        ws = [fnp.asarray(ex["weights"][li], dtype=fnp.float32) for li in range(32)]
        Y = np.asarray(ex["all_layer_means"][-1], np.float64)
        seed = _estimator_seed(int(ex["mlp_seed"]))
        mu30, mu31, s, feats, _ = tl._extract(hada, ws, seed, A30, A31)
        feats2 = np.concatenate(
            [
                feats,
                (mu31 / (s + 1e-8)).reshape(1, -1),
                np.log(np.maximum(np.abs(mu31), 1e-8)).reshape(1, -1),
            ],
            0,
        )
        Xs = ((feats2 - mu.reshape(-1, 1)) / sd.reshape(-1, 1)).T
        dlt = sh * _forward(params, Xs) * (s + 1e-8)
        corr = mu31 + dlt
        eb.append(float(np.mean((mu31 - Y) ** 2)))
        ec.append(float(np.mean((corr - Y) ** 2)))
        if i % 20 == 0:
            print("mini", i, eb[-1], ec[-1], flush=True)
    b, c = float(np.mean(eb)), float(np.mean(ec))
    print(
        f"Mini base={b:.4e} corr={c:.4e} ratio={b/max(c,1e-30):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

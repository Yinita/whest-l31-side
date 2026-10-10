#!/usr/bin/env python3
"""Nonlinear L30 residual on Full shards → Mini soft-recenter.

Per-neuron features = sample stats + random projections of W30 col & W31 row.
Shared MLP (F→H→1) predicts (Y30−μ̂)/σ. Gate: Full OOS R²>0.05 and Mini ratio>1.05.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
SHARD_DIR = ROOT / "experiments" / "results" / "full_cache"
MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
FEAT_CACHE = ROOT / "experiments" / "results" / "full_cache" / "l30_feat_cache_v3.npz"
OUT = ROOT / "experiments" / "l30_mlp_full_v3.npz"
B = int(2.72e11)
L = 30
W = 256
P = 8  # projection dim per weight vector
F = 10 + 2 * P  # sample feats + W30 col proj + W31 row proj
H = 64
SEED_PROJ = 42


def _estimator_seed(raw_mlp_seed: int) -> int:
    """Phase-1 seed_protocol 3.0: parquet mlp_seed → estimator seed."""
    ss = fnp.random.SeedSequence(int(raw_mlp_seed)).spawn(3)
    return int(ss[2].generate_state(1)[0]) & 0x7FFFFFFF


def _load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def _proj_mats(rng: np.random.Generator):
    # Fixed random projections (Gaussian) — share across train/infer
    A30 = rng.normal(0, 1.0 / np.sqrt(W), size=(P, W)).astype(np.float64)
    A31 = rng.normal(0, 1.0 / np.sqrt(W), size=(P, W)).astype(np.float64)
    return A30, A31


def _extract(hada, weights, seed: int, A30, A31):
    depth = 32
    n2 = float(W * W)
    flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * W * 20.0
    target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
    n_forward = int(max(2 * W, min(target / flops_per, hada._MAX_SAMPLES)))
    n_frames = max(1, n_forward // (2 * W))
    e_r = hada._chi_mean_radius(W)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(n_frames, W, seed=seed, e_r=e_r)
        for li in range(L + 1):
            x = fnp.maximum(x @ weights[li], 0.0)
        mu30 = np.asarray(fnp.mean(x, axis=0), np.float64)
        a = np.asarray(x, np.float64)
        x2 = x
        for li in range(L + 1, depth):
            x2 = fnp.maximum(x2 @ weights[li], 0.0)
        mu31 = np.asarray(fnp.mean(x2, axis=0), np.float64)
    s = np.sqrt(np.maximum(((a - mu30) ** 2).mean(0), 1e-12))
    act = (a > 0).mean(0)
    W30 = np.asarray(weights[L], np.float64)
    W31 = np.asarray(weights[L + 1], np.float64)
    # (P, W): proj of each column of W30 / each row of W31
    p30 = A30 @ W30  # (P, W)
    p31 = (A31 @ W31.T)  # (P, W)  — A31 @ row^T for each row → cols
    feats = np.concatenate(
        [
            np.stack(
                [
                    np.ones(W),
                    mu30 / (s + 1e-8),
                    s / (s.mean() + 1e-8),
                    act,
                    np.log(s + 1e-8),
                    (mu30 - mu30.mean()) / (s + 1e-8),
                    np.sqrt(np.maximum((W30 * W30).sum(0), 1e-30))
                    / (np.sqrt((W30 * W30).sum(0)).mean() + 1e-8),
                    np.sqrt(np.maximum((W31 * W31).sum(1), 1e-30))
                    / (np.sqrt((W31 * W31).sum(1)).mean() + 1e-8),
                    W30.mean(0),
                    W31.mean(1),
                ],
                0,
            ),
            p30,
            p31,
        ],
        0,
    )
    return mu30, mu31, s, feats.astype(np.float64), x


def _init_mlp(rng: np.random.Generator):
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / F), (H, F)),
        "b1": np.zeros(H),
        "W2": rng.normal(0, np.sqrt(2.0 / H), (1, H)),
        "b2": np.zeros(1),
    }


def _forward(p, X):
    h = np.maximum(0.0, X @ p["W1"].T + p["b1"])  # ReLU
    return (h @ p["W2"].T + p["b2"]).reshape(-1)


def _train_mlp(X, y, rng, steps=1500, lr=3e-3, l2=1e-4, batch=8192, val=None):
    p = _init_mlp(rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(vv) for k, vv in p.items()}
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    best_state, best_val = None, 1e9
    n = X.shape[0]
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, size=min(batch, n))
        xb, yb = X[idx], y[idx]
        h_pre = xb @ p["W1"].T + p["b1"]
        h = np.maximum(0.0, h_pre)
        pred = (h @ p["W2"].T + p["b2"]).reshape(-1)
        err = pred - yb
        loss = float(np.mean(err * err))
        # grads
        dpred = (2.0 / len(yb)) * err
        gW2 = dpred.reshape(-1, 1).T @ h / 1.0  # wait mean
        gW2 = (dpred.reshape(-1, 1) * h).mean(0, keepdims=True) + l2 * p["W2"]
        gb2 = np.array([dpred.mean()]) + l2 * p["b2"]
        dh = dpred.reshape(-1, 1) * p["W2"]  # (B,H)
        dh_pre = dh * (h_pre > 0)
        gW1 = (dh_pre.T @ xb) / len(yb) + l2 * p["W1"]
        gb1 = dh_pre.mean(0) + l2 * p["b1"]
        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        # Adam
        for k in p:
            g = grads[k]
            # clip
            gn = np.linalg.norm(g)
            if gn > 5.0:
                g = g * (5.0 / gn)
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * (g * g)
            mh = m[k] / (1 - b1**t)
            vh = v[k] / (1 - b2**t)
            p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)
        if val is not None and t % 50 == 0:
            pred_v = _forward(p, val[0])
            mv = float(np.mean((pred_v - val[1]) ** 2))
            if mv < best_val:
                best_val = mv
                best_state = {k: v.copy() for k, v in p.items()}
        if t % 300 == 0:
            lr *= 0.5
    if best_state is not None:
        p = best_state
    return p


def _load_shards():
    shards = sorted(SHARD_DIR.glob("full_shard_*.npz"))
    Ws, Ys, seeds = [], [], []
    for p in shards:
        d = np.load(p)
        Ws.append(d["weights"])
        Ys.append(d["all_layer_means"])
        seeds.append(d["mlp_seed"])
        print("loaded", p.name, d["weights"].shape[0], flush=True)
    return (
        np.concatenate(Ws, 0),
        np.concatenate(Ys, 0),
        np.concatenate(seeds, 0),
    )


def main() -> None:
    hada = _load_hada()
    A30, A31 = _proj_mats(np.random.default_rng(SEED_PROJ))
    weights_all, alm_all, seeds_all = _load_shards()
    n = weights_all.shape[0]
    print(f"Full n={n} F={F} H={H}", flush=True)

    if FEAT_CACHE.is_file():
        c = np.load(FEAT_CACHE)
        if int(c["n"]) == n and int(c["F"]) == F:
            B30, B31, Sig, Fe, Y30, Y31 = (
                c["B30"],
                c["B31"],
                c["Sig"],
                c["Fe"],
                c["Y30"],
                c["Y31"],
            )
            print("feat cache hit", FEAT_CACHE, flush=True)
        else:
            c = None
            print("feat cache shape mismatch; rebuild", flush=True)
    else:
        c = None
        print("no feat cache at", FEAT_CACHE, "; extracting", flush=True)

    if c is None:
        B30 = np.zeros((n, W))
        B31 = np.zeros((n, W))
        Sig = np.zeros((n, W))
        Fe = np.zeros((n, F, W))
        Y30 = np.zeros((n, W))
        Y31 = np.zeros((n, W))
        t0 = time.time()
        for i in range(n):
            ws = [fnp.asarray(weights_all[i, li], dtype=fnp.float32) for li in range(32)]
            seed = _estimator_seed(int(seeds_all[i]))
            mu30, mu31, s, feats, _ = _extract(hada, ws, seed, A30, A31)
            B30[i], B31[i], Sig[i], Fe[i] = mu30, mu31, s, feats
            Y30[i] = alm_all[i, L]
            Y31[i] = alm_all[i, -1]
            if (i + 1) % 25 == 0:
                print(
                    f"[{i+1}/{n}] L30={np.mean((mu30-Y30[i])**2):.3e} t={time.time()-t0:.0f}s",
                    flush=True,
                )
        np.savez(
            FEAT_CACHE,
            B30=B30,
            B31=B31,
            Sig=Sig,
            Fe=Fe,
            Y30=Y30,
            Y31=Y31,
            n=n,
            F=F,
            A30=A30,
            A31=A31,
        )

    T = Y30 - B30
    base30 = float(np.mean(T**2))
    base31 = float(np.mean((B31 - Y31) ** 2))
    print(f"base L30={base30:.4e} L31={base31:.4e}", flush=True)

    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(n), 5)
    best = None
    for l2 in [1e-3, 1e-4]:
        oos_delta = np.zeros_like(T)
        t_l2 = time.time()
        for fi, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            mu = Fe[tr].mean((0, 2))
            sd = Fe[tr].std((0, 2)) + 1e-8
            sd[0] = 1.0
            Xtr = (Fe[tr] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            Xtr = Xtr.transpose(0, 2, 1).reshape(-1, F)
            ytr = (T[tr] / (Sig[tr] + 1e-8)).reshape(-1)
            # small val split inside train
            ntr = Xtr.shape[0]
            perm = rng.permutation(ntr)
            nv = max(1000, ntr // 10)
            va, tr_i = perm[:nv], perm[nv:]
            print(
                f"train l2={l2:.0e} fold={fi+1}/5 ntr={len(tr_i)} …",
                flush=True,
            )
            params = _train_mlp(
                Xtr[tr_i],
                ytr[tr_i],
                np.random.default_rng(int(rng.integers(1 << 30))),
                steps=800,
                lr=5e-3,
                l2=l2,
                val=(Xtr[va], ytr[va]),
            )
            Xva = (Fe[te] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            for j, i in enumerate(te):
                d = _forward(params, Xva[j].T) * (Sig[i] + 1e-8)
                oos_delta[i] = d
            print(
                f"  fold {fi+1} done t={time.time()-t_l2:.0f}s",
                flush=True,
            )
        for sh in [1.0, 0.5, 0.25, 0.1, 0.05]:
            m = float(np.mean((T - sh * oos_delta) ** 2))
            r2 = 1.0 - m / base30
            print(f"l2={l2:.0e} sh={sh:.2f} δMSE={m:.4e} R²={r2:.4f}", flush=True)
            if best is None or m < best[0]:
                best = (m, l2, sh, oos_delta.copy())

    l2, sh = best[1], best[2]
    print(f"BEST R²={1-best[0]/base30:.4f} l2={l2} sh={sh}", flush=True)

    # refit on all
    mu = Fe.mean((0, 2))
    sd = Fe.std((0, 2)) + 1e-8
    sd[0] = 1.0
    X = (Fe - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
    X = X.transpose(0, 2, 1).reshape(-1, F)
    y = (T / (Sig + 1e-8)).reshape(-1)
    nall = X.shape[0]
    perm = rng.permutation(nall)
    nv = max(2000, nall // 10)
    params = _train_mlp(
        X[perm[nv:]],
        y[perm[nv:]],
        rng,
        l2=l2,
        val=(X[perm[:nv]], y[perm[:nv]]),
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
        oos_r2=1 - best[0] / base30,
    )
    print(f"wrote {OUT}", flush=True)

    if 1 - best[0] / base30 < 0.02:
        print("R² too low; skip Mini (gate fail)", flush=True)
        return

    # Mini transfer
    ds = Dataset.load_from_disk(str(MINI))
    eb, ec = [], []
    for i in range(len(ds)):
        ex = ds[i]
        ws = [fnp.asarray(ex["weights"][li], dtype=fnp.float32) for li in range(32)]
        Y = np.asarray(ex["all_layer_means"], np.float64)
        seed = _estimator_seed(int(ex["mlp_seed"]))
        mu30, mu31, s, feats, x = _extract(hada, ws, seed, A30, A31)
        Xs = ((feats - mu.reshape(-1, 1)) / sd.reshape(-1, 1)).T
        dlt = sh * _forward(params, Xs) * (s + 1e-8)
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x2 = x + fnp.asarray(dlt, dtype=fnp.float32)
            x2 = fnp.maximum(x2, 0.0)
            for li in range(L + 1, 32):
                x2 = fnp.maximum(x2 @ ws[li], 0.0)
            corr = np.asarray(fnp.mean(x2, axis=0), np.float64)
        eb.append(np.mean((mu31 - Y[-1]) ** 2))
        ec.append(np.mean((corr - Y[-1]) ** 2))
        if i % 20 == 0:
            print("mini", i, eb[-1], ec[-1], flush=True)
    b, c = float(np.mean(eb)), float(np.mean(ec))
    print(
        f"Mini base={b:.4e} corr={c:.4e} ratio={b/max(c,1e-30):.3f} "
        f"adj≈{b*0.4:.4e}/{c*0.4:.4e}",
        flush=True,
    )


if __name__ == "__main__":
    main()

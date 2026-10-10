#!/usr/bin/env python3
"""Train L30 residual on Full cache; transfer to Mini.

Requires experiments/results/full_cache/full_alm400.npz (or larger).
Uses Hadamard @ util=0.40 features + weight feats → Y30−μ̂30.
Then soft-α additive at L31: ŷ = μ31 + α·δ̂_push — actually recenter soft.
Primary eval: Mini OOS / official-style raw MSE.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "experiments" / "results" / "full_cache" / "full_alm400.npz"
MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
OUT = ROOT / "experiments" / "l30_full_residual.npz"
B = int(2.72e11)
RECENTER_LI = 30
F = 18


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
    depth = len(weights)
    n2 = float(width * width)
    flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
    n_forward = int(max(2 * width, min(target / flops_per, hada._MAX_SAMPLES)))
    n_frames = max(1, n_forward // (2 * width))
    e_r = hada._chi_mean_radius(width)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(n_frames, width, seed=seed, e_r=e_r)
        for li in range(RECENTER_LI + 1):
            x = fnp.maximum(x @ weights[li], 0.0)
        mu30 = np.asarray(fnp.mean(x, axis=0), np.float64)
        a = np.asarray(x, np.float64)
        x31 = x
        for li in range(RECENTER_LI + 1, depth):
            x31 = fnp.maximum(x31 @ weights[li], 0.0)
        mu31 = np.asarray(fnp.mean(x31, axis=0), np.float64)
    s = np.sqrt(np.maximum(((a - mu30) ** 2).mean(0), 1e-12))
    act = (a > 0).mean(0)
    W30 = np.asarray(weights[RECENTER_LI], np.float64)
    W31 = np.asarray(weights[RECENTER_LI + 1], np.float64)
    wfeat = np.stack(
        [
            np.sqrt(np.maximum((W30 * W30).sum(0), 1e-30)),
            np.sqrt(np.maximum((W30 * W30).sum(1), 1e-30)),
            W30.mean(0),
            np.sqrt(np.maximum((W31 * W31).sum(0), 1e-30)),
            np.sqrt(np.maximum((W31 * W31).sum(1), 1e-30)),
            W31.mean(0),
            np.sqrt(np.maximum((W31 * W31).sum(0), 1e-30)) * act,
            W31.mean(1),
        ],
        0,
    )
    feats = np.stack(
        [
            np.ones(width),
            mu30 / (s + 1e-8),
            s / (s.mean() + 1e-8),
            act,
            np.log(s + 1e-8),
            (mu30 - mu30.mean()) / (s + 1e-8),
            wfeat[0] / (wfeat[0].mean() + 1e-8),
            wfeat[1] / (wfeat[1].mean() + 1e-8),
            wfeat[2],
            wfeat[3] / (wfeat[3].mean() + 1e-8),
            wfeat[4] / (wfeat[4].mean() + 1e-8),
            wfeat[5],
            wfeat[6] / (wfeat[6].mean() + 1e-8),
            wfeat[7],
            act * mu30 / (s + 1e-8),
            s,
            mu30,
            wfeat[3] * mu30 / (s + 1e-8),
        ],
        0,
    )
    return mu30, mu31, s, feats.astype(np.float64), x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path, default=FULL)
    ap.add_argument("--max-train", type=int, default=400)
    args = ap.parse_args()
    if not args.full.is_file():
        raise SystemExit(f"missing {args.full}; wait for download")

    hada = _load_hada()
    d = np.load(args.full)
    n = min(args.max_train, d["weights"].shape[0])
    print(f"Full n={n}", flush=True)

    Bases30 = np.zeros((n, 256))
    Bases31 = np.zeros((n, 256))
    Feats = np.zeros((n, F, 256))
    Sig = np.zeros((n, 256))
    Y30 = np.zeros((n, 256))
    Y31 = np.zeros((n, 256))
    t0 = time.time()
    for i in range(n):
        weights = [fnp.asarray(d["weights"][i, l], dtype=fnp.float32) for l in range(32)]
        seed = int(d["mlp_seed"][i]) & 0x7FFFFFFF
        mu30, mu31, s, feats, _ = _extract(hada, weights, seed)
        Bases30[i], Bases31[i], Sig[i], Feats[i] = mu30, mu31, s, feats
        Y30[i] = d["all_layer_means"][i, RECENTER_LI]
        Y31[i] = d["all_layer_means"][i, -1]
        if (i + 1) % 25 == 0:
            print(
                f"[{i+1}/{n}] L30mse={np.mean((mu30-Y30[i])**2):.3e} t={time.time()-t0:.0f}s",
                flush=True,
            )

    T = Y30 - Bases30
    base30 = float(np.mean(T**2))
    base31 = float(np.mean((Bases31 - Y31) ** 2))
    print(f"base L30={base30:.4e} L31={base31:.4e}")

    folds = np.array_split(np.arange(n), 5)
    best = None
    for lam in [1e-1, 3e-2, 1e-2, 1e-3]:
        for sh in [1.0, 0.5, 0.25, 0.1, 0.05]:
            oos30 = np.zeros_like(Bases30)
            for fi, te in enumerate(folds):
                tr = np.concatenate([folds[j] for j in range(5) if j != fi])
                mu = Feats[tr].mean((0, 2))
                sd = Feats[tr].std((0, 2)) + 1e-8
                sd[0] = 1.0
                Xtr = (Feats[tr] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
                y = (T[tr] / (Sig[tr] + 1e-8)).reshape(-1)
                Xt = Xtr.transpose(1, 0, 2).reshape(F, -1)
                beta = np.linalg.solve(
                    Xt @ Xt.T / Xt.shape[1] + lam * np.eye(F), Xt @ y / Xt.shape[1]
                )
                Xva = (Feats[te] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
                for j, i in enumerate(te):
                    oos30[i] = sh * (beta @ Xva[j]) * (Sig[i] + 1e-8)
            m30 = float(np.mean((T - oos30) ** 2))
            # soft additive on L31 using δ30 as proxy (no path change): weak
            # better metric: correlation / residual R^2
            r2 = 1.0 - m30 / base30
            print(f"lam={lam:.0e} sh={sh:.2f} δMSE={m30:.4e} R²={r2:.4f}")
            if best is None or m30 < best[0]:
                best = (m30, lam, sh)

    lam, sh = best[1], best[2]
    mu = Feats.mean((0, 2))
    sd = Feats.std((0, 2)) + 1e-8
    sd[0] = 1.0
    X = (Feats - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
    y = (T / (Sig + 1e-8)).reshape(-1)
    Xt = X.transpose(1, 0, 2).reshape(F, -1)
    beta = np.linalg.solve(
        Xt @ Xt.T / Xt.shape[1] + lam * np.eye(F), Xt @ y / Xt.shape[1]
    )
    np.savez(OUT, beta=beta, feat_mu=mu, feat_sd=sd, lam=lam, shrink=sh)
    print(f"wrote {OUT} best δMSE={best[0]:.4e} R²={1-best[0]/base30:.4f}")

    # Mini transfer: soft recenter with predicted δ
    print("Mini transfer soft-recenter…", flush=True)
    ds = Dataset.load_from_disk(str(MINI))
    eb, ec = [], []
    for i in range(len(ds)):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        mu30, mu31, s, feats, x = _extract(hada, weights, seed)
        dlt = sh * (beta @ ((feats - mu.reshape(-1, 1)) / sd.reshape(-1, 1))) * (s + 1e-8)
        # soft recenter α=1 on δ then finish
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x2 = x + fnp.asarray(dlt, dtype=fnp.float32)
            x2 = fnp.maximum(x2, 0.0)
            for li in range(RECENTER_LI + 1, 32):
                x2 = fnp.maximum(x2 @ weights[li], 0.0)
            corr = np.asarray(fnp.mean(x2, axis=0), np.float64)
        eb.append(np.mean((mu31 - Y[-1]) ** 2))
        ec.append(np.mean((corr - Y[-1]) ** 2))
        if i % 20 == 0:
            print(i, eb[-1], ec[-1], flush=True)
    b, c = float(np.mean(eb)), float(np.mean(ec))
    print(f"Mini base={b:.4e} corr={c:.4e} ratio={b/c:.3f} adj≈{b*0.4:.4e}/{c*0.4:.4e}")


if __name__ == "__main__":
    main()

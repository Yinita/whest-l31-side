#!/usr/bin/env python3
"""Train residual θ using ONLY Hadamard-path features (no gain-cov FLOPs).

Features per neuron from final-layer samples: 1, μ, μ², σ, logσ, μ/σ, σ·μ.
5-fold Mini OOS. If ratio>1.01 write npz + lit candidate.
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
OUT = ROOT / "experiments" / "corrector_hada_cheap_v031.npz"
B = int(2.72e11)


def _load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def _hada_mu_std(hada_mod, weights, seed: int):
    width = weights[0].shape[0]
    depth = len(weights)
    n2 = float(width * width)
    flops_per = depth * hada_mod._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * hada_mod._UTIL * hada_mod._UTIL_CALIB, flops_per)
    n_forward = int(max(2 * width, min(target / flops_per, hada_mod._MAX_SAMPLES)))
    n_frames = max(1, n_forward // (2 * width))
    e_r = hada_mod._chi_mean_radius(width)
    with flops.BudgetContext(flop_budget=B * 2):
        x = hada_mod._hadamard_directions(n_frames, width, seed=seed, e_r=e_r)
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        mu = fnp.mean(x, axis=0)
        second = fnp.mean(x * x, axis=0)
        var = fnp.maximum(second - mu * mu, 0.0)
        sd = fnp.sqrt(var)
    return np.asarray(mu, np.float64), np.asarray(sd, np.float64)


def _features(mu, sd) -> np.ndarray:
    s = np.maximum(sd, 1e-8)
    return np.stack(
        [
            np.ones_like(mu),
            mu,
            mu * mu,
            s,
            np.log(s),
            mu / s,
            s * mu,
            mu * mu * mu,
        ],
        axis=0,
    )


def main() -> None:
    hada = _load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    n = len(ds)
    F = 8
    W = 256
    Mu = np.zeros((n, W), np.float64)
    Sd = np.zeros((n, W), np.float64)
    Y = np.zeros((n, W), np.float64)
    for i in range(n):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        mu, sd = _hada_mu_std(hada, weights, seed)
        Mu[i], Sd[i] = mu, sd
        Y[i] = np.asarray(ex["all_layer_means"][-1], np.float64)
        if i % 10 == 0:
            print(f"[{i}] mse={np.mean((mu - Y[i]) ** 2):.4e}", flush=True)

    Feats = np.stack([_features(Mu[i], Sd[i]) for i in range(n)], axis=0)  # (n,F,W)
    R = Y - Mu
    base_mse = float(np.mean(R * R))
    print("base_mse", base_mse)

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)

    best = None
    for lam in [1e-3, 1e-2, 3e-2, 1e-1, 3e-1]:
        oos = np.zeros_like(R)
        for fi, val in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            mu_f = Feats[tr].mean(axis=(0, 2))
            sd_f = Feats[tr].std(axis=(0, 2)) + 1e-8
            Xtr = (Feats[tr] - mu_f.reshape(1, -1, 1)) / sd_f.reshape(1, -1, 1)
            Xt = Xtr.transpose(1, 0, 2).reshape(F, -1)
            yt = R[tr].reshape(-1)
            gram = Xt @ Xt.T / Xt.shape[1] + lam * np.eye(F)
            beta = np.linalg.solve(gram, Xt @ yt / Xt.shape[1])
            Xva = (Feats[val] - mu_f.reshape(1, -1, 1)) / sd_f.reshape(1, -1, 1)
            for j, vi in enumerate(val):
                oos[vi] = np.einsum("f,fw->w", beta, Xva[j])
        mse = float(np.mean((R - oos) ** 2))
        ratio = base_mse / mse
        print(f"lam={lam:.0e} oos_mse={mse:.6e} ratio={ratio:.4f}")
        if best is None or ratio > best[0]:
            best = (ratio, lam, mse)

    # refit on all with best lam
    lam = best[1]
    mu_f = Feats.mean(axis=(0, 2))
    sd_f = Feats.std(axis=(0, 2)) + 1e-8
    X = (Feats - mu_f.reshape(1, -1, 1)) / sd_f.reshape(1, -1, 1)
    Xt = X.transpose(1, 0, 2).reshape(F, -1)
    yt = R.reshape(-1)
    gram = Xt @ Xt.T / Xt.shape[1] + lam * np.eye(F)
    beta = np.linalg.solve(gram, Xt @ yt / Xt.shape[1])
    np.savez(
        OUT,
        beta=beta.astype(np.float64),
        mu=mu_f.astype(np.float64),
        sd=sd_f.astype(np.float64),
        lam=lam,
        oos_ratio=best[0],
        oos_mse=best[2],
        base_mse=base_mse,
    )
    print("WROTE", OUT, "best_ratio", best[0], "lam", lam)


if __name__ == "__main__":
    main()

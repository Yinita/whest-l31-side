#!/usr/bin/env python3
"""Offline: shrink μ_30 = (1-λ)μ_s + λ μ_wb then recenter paths → L31.

μ_wb candidates: diag-gain; ridge on weight/gain features (5-fold OOS).
Gate: OOS L31 mse must beat Hadamard base before any official Mini.
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
OUT = ROOT / "experiments" / "l30_shrink_probe.npz"
B = int(2.72e11)
RECENTER_LAYER = 30


def _load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def _diag_gain_to_layer(weights: list, stop_li: int):
    width = weights[0].shape[0]
    mu = np.zeros(width, np.float64)
    var = np.ones(width, np.float64)
    alpha = phi = sigma = None
    for li in range(stop_li + 1):
        W = np.asarray(weights[li], np.float64)
        mu_pre = W.T @ mu
        var_pre = np.maximum((W.T ** 2) @ var, 1e-12)
        sigma = np.sqrt(var_pre)
        alpha = mu_pre / sigma
        phi = np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + np.vectorize(math.erf)(alpha / math.sqrt(2.0)))
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sigma * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
    return mu, alpha, phi, sigma, var


def _wb_features(mu_g, alpha, phi, sigma, var, weights) -> np.ndarray:
    """Weight-only features (F, width) — no Hadamard sample stats."""
    s = np.maximum(sigma, 1e-8)
    a = alpha
    # cheap weight fingerprints broadcast per neuron
    wL = np.asarray(weights[RECENTER_LAYER], np.float64)
    col_n = np.sqrt(np.maximum(np.sum(wL * wL, axis=0), 1e-30))
    row_n = np.sqrt(np.maximum(np.sum(wL * wL, axis=1), 1e-30))
    rn_mean = float(row_n.mean())
    return np.stack(
        [
            np.ones_like(mu_g),
            mu_g,
            a,
            a * a,
            a * a * a,
            phi,
            s,
            np.log(s),
            var,
            np.log(var + 1e-12),
            col_n,
            np.full_like(mu_g, rn_mean),
            phi * a,
            mu_g / s,
            a * s,
            np.abs(a),
        ],
        axis=0,
    )


def _hada_cache(hada, weights, seed: int):
    width = weights[0].shape[0]
    depth = len(weights)
    n2 = float(width * width)
    flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
    n_forward = int(max(2 * width, min(target / flops_per, hada._MAX_SAMPLES)))
    n_frames = max(1, n_forward // (2 * width))
    e_r = hada._chi_mean_radius(width)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(n_frames, width, seed=seed, e_r=e_r)
        for li in range(RECENTER_LAYER + 1):
            x = fnp.maximum(x @ weights[li], 0.0)
        mu_s = np.asarray(fnp.mean(x, axis=0), np.float64)
        x_np = np.asarray(x, np.float32)
        # finish baseline without recenter
        x2 = x
        for li in range(RECENTER_LAYER + 1, depth):
            x2 = fnp.maximum(x2 @ weights[li], 0.0)
        mu_base = np.asarray(fnp.mean(x2, axis=0), np.float64)
    return mu_s, mu_base, x_np


def _rollout(x_pre: np.ndarray, weights, mu_s, mu_t):
    """Recentering by mu_s→mu_t then last layer(s)."""
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = fnp.asarray(x_pre)
        x = x - fnp.asarray(mu_s - mu_t, dtype=fnp.float32)
        x = fnp.maximum(x, 0.0)
        for li in range(RECENTER_LAYER + 1, len(weights)):
            x = fnp.maximum(x @ weights[li], 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def main() -> None:
    hada = _load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    n = len(ds)
    W = 256
    F = 16

    MuS = np.zeros((n, W))
    MuG = np.zeros((n, W))
    MuB = np.zeros((n, W))
    Y30 = np.zeros((n, W))
    Y31 = np.zeros((n, W))
    Feats = np.zeros((n, F, W))
    # store activations — 100 * ~80k * 256 * 4 ≈ 8GB too big.
    # Re-forward per λ eval on subset, or cache to disk memmap.
    cache_dir = Path("/dev/shm/l30_act_cache")
    cache_dir.mkdir(exist_ok=True)

    for i in range(n):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        mu_s, mu_base, x_pre = _hada_cache(hada, weights, seed)
        mu_g, alpha, phi, sigma, var = _diag_gain_to_layer(
            [np.asarray(w) for w in weights], RECENTER_LAYER
        )
        MuS[i], MuG[i], MuB[i] = mu_s, mu_g, mu_base
        Y30[i], Y31[i] = Y[RECENTER_LAYER], Y[-1]
        Feats[i] = _wb_features(mu_g, alpha, phi, sigma, var, weights)
        np.save(cache_dir / f"x{i}.npy", x_pre)
        if i % 10 == 0:
            print(
                f"[{i}] L30 s={np.mean((mu_s-Y30[i])**2):.3e} "
                f"g={np.mean((mu_g-Y30[i])**2):.3e} L31={np.mean((mu_base-Y31[i])**2):.3e}",
                flush=True,
            )

    base_l30_s = float(np.mean((MuS - Y30) ** 2))
    base_l30_g = float(np.mean((MuG - Y30) ** 2))
    base_l31 = float(np.mean((MuB - Y31) ** 2))
    print(f"base L30 sample={base_l30_s:.4e} gain={base_l30_g:.4e} L31={base_l31:.4e}")

    # --- shrink with pure gain ---
    print("\n=== shrink toward diag-gain ===")
    for lam in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
        mses = []
        for i in range(n):
            ex = ds[i]
            weights = [
                fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)
            ]
            mu_t = (1.0 - lam) * MuS[i] + lam * MuG[i]
            x_pre = np.load(cache_dir / f"x{i}.npy")
            mu31 = _rollout(x_pre, weights, MuS[i], mu_t)
            mses.append(np.mean((mu31 - Y31[i]) ** 2))
        mse = float(np.mean(mses))
        print(f"lam={lam:.2f} L31={mse:.4e} gain={base_l31/mse:.3f}", flush=True)

    # --- 5-fold ridge μ_wb → Y30 ---
    print("\n=== ridge weight-only → Y30 + shrink ===")
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    best_ridge = None
    for ridgelam in [1e-2, 1e-1, 1.0, 10.0]:
        oos = np.zeros_like(Y30)
        for fi, val in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            mu = Feats[tr].mean(axis=(0, 2))
            sd = Feats[tr].std(axis=(0, 2)) + 1e-8
            sd[0] = 1.0
            Xtr = (Feats[tr] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            # fit to Y30 absolute (not residual)
            yt = Y30[tr].reshape(-1)
            Xt = Xtr.transpose(1, 0, 2).reshape(F, -1)
            beta = np.linalg.solve(
                Xt @ Xt.T / Xt.shape[1] + ridgelam * np.eye(F),
                Xt @ yt / Xt.shape[1],
            )
            Xva = (Feats[val] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            for j, vi in enumerate(val):
                oos[vi] = beta @ Xva[j]
        m30 = float(np.mean((oos - Y30) ** 2))
        print(f"ridgeλ={ridgelam:.0e} OOS L30={m30:.4e} vs sample={base_l30_s:.4e} vs gain={base_l30_g:.4e}")
        if best_ridge is None or m30 < best_ridge[0]:
            best_ridge = (m30, ridgelam, oos.copy())

    MuW = best_ridge[2]
    print(f"best ridge L30 OOS={best_ridge[0]:.4e} lam={best_ridge[1]}")
    print("\n=== shrink toward ridge μ_wb (OOS predictions) ===")
    for lam in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
        mses = []
        for i in range(n):
            ex = ds[i]
            weights = [
                fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)
            ]
            mu_t = (1.0 - lam) * MuS[i] + lam * MuW[i]
            x_pre = np.load(cache_dir / f"x{i}.npy")
            mu31 = _rollout(x_pre, weights, MuS[i], mu_t)
            mses.append(np.mean((mu31 - Y31[i]) ** 2))
        mse = float(np.mean(mses))
        print(f"lam={lam:.2f} L31={mse:.4e} gain={base_l31/mse:.3f}", flush=True)

    # also residual ridge: μ_wb = μ_s + δ(features without μ_s?) — use gain feats only for δ
    print("\n=== residual ridge δ(feats) on Y30-μ_s, OOS ===")
    best_res = None
    for ridgelam in [1e-2, 1e-1, 1.0]:
        oos = np.zeros_like(Y30)
        for fi, val in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            mu = Feats[tr].mean(axis=(0, 2))
            sd = Feats[tr].std(axis=(0, 2)) + 1e-8
            sd[0] = 1.0
            Xtr = (Feats[tr] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            yt = (Y30[tr] - MuS[tr]).reshape(-1)
            Xt = Xtr.transpose(1, 0, 2).reshape(F, -1)
            beta = np.linalg.solve(
                Xt @ Xt.T / Xt.shape[1] + ridgelam * np.eye(F),
                Xt @ yt / Xt.shape[1],
            )
            Xva = (Feats[val] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            for j, vi in enumerate(val):
                oos[vi] = MuS[vi] + beta @ Xva[j]
        m30 = float(np.mean((oos - Y30) ** 2))
        print(f"resλ={ridgelam:.0e} OOS L30={m30:.4e} r={base_l30_s/m30:.3f}")
        if best_res is None or m30 < best_res[0]:
            best_res = (m30, ridgelam, oos.copy())

    MuR = best_res[2]
    print("\n=== hard recenter to residual-corrected μ (OOS) ===")
    mses = []
    for i in range(n):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        x_pre = np.load(cache_dir / f"x{i}.npy")
        mu31 = _rollout(x_pre, weights, MuS[i], MuR[i])
        mses.append(np.mean((mu31 - Y31[i]) ** 2))
    mse = float(np.mean(mses))
    print(f"hard residual-recenter L31={mse:.4e} gain={base_l31/mse:.3f}")

    np.savez(
        OUT,
        base_l31=base_l31,
        base_l30_s=base_l30_s,
        base_l30_g=base_l30_g,
        ridge_l30=best_ridge[0],
        residual_l30=best_res[0],
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()

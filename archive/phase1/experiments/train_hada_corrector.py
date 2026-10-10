#!/usr/bin/env python3
"""Train a final-layer residual corrector on pure-0.2.1s Hadamard predictions.

Uses Mini all_layer_means[-1] as GT. 5-fold by MLP. Writes θ for hyb-0.3.0s.
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
OUT = ROOT / "experiments" / "corrector_hada_v030.npz"
B = int(2.72e11)
INV = 1.0 / math.sqrt(2.0 * math.pi)


def _load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def _gain_last_stats(weights: list):
    width = weights[0].shape[0]
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = fnp.eye(width, dtype=fnp.float32)
    alpha = phi = sigma = None
    for w in weights:
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
    return np.asarray(mu, np.float64), np.asarray(alpha, np.float64), np.asarray(
        phi, np.float64
    ), np.asarray(sigma, np.float64)


def _hada_predict(hada_mod, weights, seed: int) -> np.ndarray:
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
    return np.asarray(mu, np.float64)


def _features(mu_h, mu_g, alpha, phi, sigma) -> np.ndarray:
    """(F, width) feature matrix — mirrors v0.2.20s style, Hadamard-aware."""
    gap = mu_h - mu_g
    glayer = np.full(mu_h.shape[0], gap.mean())
    s = np.maximum(sigma, 1e-8)
    ones = np.ones_like(mu_h)
    a = alpha
    feats = np.stack(
        [
            ones,
            a,
            a * a,
            a * a * a,
            a * a * a * a,
            phi,
            gap / s,
            glayer / s,
            mu_h / s,
            mu_g / s,
            (mu_h - mu_g) / s,
            np.log(s + 1e-8),
            phi * a,
            sigma,
        ],
        axis=0,
    )
    return feats.astype(np.float64)


def main() -> None:
    hada = _load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    n = len(ds)
    width = 256
    F = 14
    Bases = np.zeros((n, width), np.float64)
    Feats = np.zeros((n, F, width), np.float64)
    Ys = np.zeros((n, width), np.float64)

    for i in range(n):
        ex = ds[i]
        weights = [np.asarray(ex["weights"][l], np.float32) for l in range(32)]
        # wrap as list of fnp arrays
        w_fnp = [fnp.asarray(w, dtype=fnp.float32) for w in weights]
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        mu_h = _hada_predict(hada, w_fnp, seed)
        with flops.BudgetContext(flop_budget=B * 2):
            mu_g, alpha, phi, sigma = _gain_last_stats(w_fnp)
        Bases[i] = mu_h
        Feats[i] = _features(mu_h, mu_g, alpha, phi, sigma)
        Ys[i] = np.asarray(ex["all_layer_means"][-1], np.float64)
        if i % 10 == 0:
            mse = np.mean((mu_h - Ys[i]) ** 2)
            print(f"[{i}/{n}] base_mse={mse:.4e}")

    # 5-fold by MLP
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    oos_pred = np.zeros_like(Bases)
    best_lam, best_ratio = None, 1e9
    for lam in [1e-2, 1e-3, 1e-4, 1e-5]:
        oos = np.zeros_like(Bases)
        for fi, val in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            # standardize features on train
            mu = Feats[tr].mean(axis=(0, 2))
            sd = Feats[tr].std(axis=(0, 2)) + 1e-8
            Xtr = (Feats[tr] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            Xva = (Feats[val] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            # residual target scaled by sigma proxy = feat[-1] roughly; use unit
            Rtr = Ys[tr] - Bases[tr]
            # ridge: solve for β in R ≈ β @ X  with X (n,F,w) → flatten neurons
            # (F,) shared across neurons: vec(R) = X_flat @ β
            Xt = Xtr.transpose(1, 0, 2).reshape(F, -1)  # (F, n*w)
            yt = Rtr.reshape(-1)
            gram = Xt @ Xt.T / Xt.shape[1]
            gram = gram + lam * np.eye(F)
            rhs = Xt @ yt / Xt.shape[1]
            beta = np.linalg.solve(gram, rhs)
            for j, vi in enumerate(val):
                delta = beta @ Xva[j]
                oos[vi] = Bases[vi] + delta
        base_mse = np.mean((Bases - Ys) ** 2)
        oos_mse = np.mean((oos - Ys) ** 2)
        ratio = base_mse / max(oos_mse, 1e-30)
        print(f"lam={lam:.0e}  base={base_mse:.4e} oos={oos_mse:.4e} ratio={ratio:.4f}")
        if oos_mse < best_ratio:
            best_ratio = oos_mse
            best_lam = lam
            oos_pred = oos

    # refit on all with best_lam
    lam = best_lam
    mu = Feats.mean(axis=(0, 2))
    sd = Feats.std(axis=(0, 2)) + 1e-8
    X = (Feats - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
    Xt = X.transpose(1, 0, 2).reshape(F, -1)
    yt = (Ys - Bases).reshape(-1)
    gram = Xt @ Xt.T / Xt.shape[1] + lam * np.eye(F)
    beta = np.linalg.solve(gram, Xt @ yt / Xt.shape[1])
    np.savez(
        OUT,
        beta=beta.astype(np.float64),
        feat_mu=mu.astype(np.float64),
        feat_sd=sd.astype(np.float64),
        lam=lam,
        oos_mse=float(np.mean((oos_pred - Ys) ** 2)),
        base_mse=float(np.mean((Bases - Ys) ** 2)),
    )
    print(f"wrote {OUT} lam={lam} oos_mse={np.mean((oos_pred-Ys)**2):.4e}")


if __name__ == "__main__":
    main()

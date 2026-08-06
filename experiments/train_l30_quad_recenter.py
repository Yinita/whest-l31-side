#!/usr/bin/env python3
"""L30 quadratic residual + soft recenter probe (Mini 5-fold OOS).

Predict δ ≈ Y30-μ_s with σ-scaled quadratic features (opus-style).
Rollout: recenter by sc·δ, measure L31. Require OOS L31 gain > 1.02
before shipping.
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
CACHE = Path("/dev/shm/l30_act_cache")
OUT = ROOT / "experiments" / "l30_quad_recenter.npz"
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


def _diag_gain(weights, stop_li):
    width = weights[0].shape[0]
    mu = np.zeros(width, np.float64)
    var = np.ones(width, np.float64)
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
    return mu, alpha, phi, sigma


def _base_feats(mu_s, mu_g, alpha, phi, sigma):
    s = np.maximum(sigma, 1e-8)
    gap = mu_s - mu_g
    gl = np.full_like(mu_s, gap.mean())
    a = alpha
    F = np.stack(
        [
            np.ones_like(mu_s),
            a,
            a * a,
            a * a * a,
            phi,
            gap / s,
            gl / s,
            mu_s / s,
            mu_g / s,
            np.log(s),
            phi * a,
            sigma,
            (mu_s - mu_s.mean()) / s,
            gap,
            s / (s.mean() + 1e-12),
        ],
        axis=0,
    )  # (15, W)
    # quadratic on feats 1..14
    cols = [F[j] for j in range(15)]
    for a_i in range(1, 15):
        for b_i in range(a_i, 15):
            cols.append(F[a_i] * F[b_i])
    return np.stack(cols, axis=0)  # (15+105, W) = 120


def _hada_to_l30(hada, weights, seed):
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
        for li in range(RECENTER_LAYER + 1):
            x = fnp.maximum(x @ weights[li], 0.0)
        mu_s = np.asarray(fnp.mean(x, axis=0), np.float64)
        x2 = x
        for li in range(RECENTER_LAYER + 1, depth):
            x2 = fnp.maximum(x2 @ weights[li], 0.0)
        mu_b = np.asarray(fnp.mean(x2, axis=0), np.float64)
        x_np = np.asarray(x, np.float32)
    return mu_s, mu_b, x_np


def _rollout(x_pre, weights, mu_s, delta):
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = fnp.asarray(x_pre)
        # recenter by delta: μ_t = μ_s + delta → subtract (μ_s - μ_t) = -delta
        x = x + fnp.asarray(delta, dtype=fnp.float32)
        x = fnp.maximum(x, 0.0)
        for li in range(RECENTER_LAYER + 1, len(weights)):
            x = fnp.maximum(x @ weights[li], 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def main():
    hada = _load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    n = len(ds)
    W = 256

    MuS = np.zeros((n, W))
    MuB = np.zeros((n, W))
    Y30 = np.zeros((n, W))
    Y31 = np.zeros((n, W))
    Sig = np.zeros((n, W))
    FM = None

    for i in range(n):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        path = CACHE / f"x{i}.npy"
        if path.exists():
            # still need mu_s, mu_b — recompute cheap from cache mean + finish
            x_pre = np.load(path)
            mu_s = x_pre.mean(axis=0).astype(np.float64)
            with flops.BudgetContext(flop_budget=B * 2, quiet=True):
                x = fnp.asarray(x_pre)
                for li in range(RECENTER_LAYER + 1, 32):
                    x = fnp.maximum(x @ weights[li], 0.0)
                mu_b = np.asarray(fnp.mean(x, axis=0), np.float64)
        else:
            mu_s, mu_b, x_pre = _hada_to_l30(hada, weights, seed)
            np.save(path, x_pre)
        mu_g, alpha, phi, sigma = _diag_gain([np.asarray(w) for w in weights], RECENTER_LAYER)
        feats = _base_feats(mu_s, mu_g, alpha, phi, sigma)
        if FM is None:
            FM = np.zeros((n, feats.shape[0], W))
        FM[i] = feats
        MuS[i], MuB[i] = mu_s, mu_b
        Y30[i], Y31[i] = Y[RECENTER_LAYER], Y[-1]
        Sig[i] = np.maximum(sigma, 1e-8)
        if i % 10 == 0:
            print(f"[{i}] L30={np.mean((mu_s-Y30[i])**2):.3e} L31={np.mean((mu_b-Y31[i])**2):.3e}", flush=True)

    base_l30 = float(np.mean((MuS - Y30) ** 2))
    base_l31 = float(np.mean((MuB - Y31) ** 2))
    print(f"base L30={base_l30:.4e} L31={base_l31:.4e} F={FM.shape[1]}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)

    def fit(tr, lam):
        Xs, ys = [], []
        for i in tr:
            Xs.append(((FM[i].T - 0) ))  # placeholder
        # proper:
        X = np.concatenate([FM[i].T for i in tr], 0)
        y = np.concatenate([(Y30[i] - MuS[i]) / Sig[i] for i in tr], 0)
        mu = X.mean(0)
        sd = X.std(0) + 1e-8
        mu[0] = 0.0
        sd[0] = 1.0
        Xs = (X - mu) / sd
        Fdim = Xs.shape[1]
        beta = np.linalg.solve(
            Xs.T @ Xs + lam * len(y) * np.eye(Fdim), Xs.T @ y
        )
        return beta, mu, sd

    def apply_delta(beta, mu, sd, i, sc):
        Xi = (FM[i].T - mu) / sd
        return sc * Sig[i] * (Xi @ beta)

    best = None
    for lam in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        for sc in [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
            m30s, m31s = [], []
            for fi, val in enumerate(folds):
                tr = np.concatenate([folds[j] for j in range(5) if j != fi])
                beta, mu, sd = fit(tr, lam)
                for i in val:
                    delta = apply_delta(beta, mu, sd, i, sc)
                    m30s.append(np.mean((MuS[i] + delta - Y30[i]) ** 2))
                    ex = ds[int(i)]
                    weights = [
                        fnp.asarray(ex["weights"][l], dtype=fnp.float32)
                        for l in range(32)
                    ]
                    x_pre = np.load(CACHE / f"x{i}.npy")
                    mu31 = _rollout(x_pre, weights, MuS[i], delta)
                    m31s.append(np.mean((mu31 - Y31[i]) ** 2))
            m30 = float(np.mean(m30s))
            m31 = float(np.mean(m31s))
            print(
                f"lam={lam:.0e} sc={sc:.2f} L30={m30:.4e}(r={base_l30/m30:.3f}) "
                f"L31={m31:.4e}(r={base_l31/m31:.3f})",
                flush=True,
            )
            if best is None or m31 < best[0]:
                best = (m31, lam, sc, m30)

    print(
        f"BEST L31={best[0]:.4e} r={base_l31/best[0]:.3f} lam={best[1]} sc={best[2]}",
        flush=True,
    )
    # refit full
    beta, mu, sd = fit(np.arange(n), best[1])
    np.savez(
        OUT,
        beta=beta,
        feat_mu=mu,
        feat_sd=sd,
        lam=best[1],
        scale=best[2],
        oos_l31=best[0],
        base_l31=base_l31,
        oos_l30=best[3],
        base_l30=base_l30,
    )
    print("wrote", OUT, "ship?" , base_l31 / best[0] > 1.02)


if __name__ == "__main__":
    main()

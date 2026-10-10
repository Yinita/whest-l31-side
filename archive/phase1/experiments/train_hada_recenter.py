#!/usr/bin/env python3
"""P3 DAgger-lite: learn penultimate (L30) mean residual on Hadamard path.

Oracle: GT-recenter at L30 (not L31) cuts Mini final MSE ~50×.
Train shared linear θ: features(sample + cheap analytic) → Y[30]-μ̂_30.
Eval: OOS TF residual + rollout (apply δ, recenter, finish L31).
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
OUT = ROOT / "experiments" / "recenter_hada_v032.npz"
B = int(2.72e11)
F = 12
RECENTER_LAYER = 30  # 0-indexed; last pre-final


def _load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def _diag_gain_to_layer(weights: list, stop_li: int):
    """Diagonal gain-cov mean/σ/α/φ up to and including layer stop_li."""
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
        # erf-based Φ, φ without flopscope for speed
        phi = np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
        # Φ via math.erf vectorized
        Phi = 0.5 * (1.0 + np.vectorize(math.erf)(alpha / math.sqrt(2.0)))
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sigma * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
    return mu, alpha, phi, sigma


def _features(mu_s, mu_g, alpha, phi, sigma) -> np.ndarray:
    s = np.maximum(sigma, 1e-8)
    gap = mu_s - mu_g
    glayer = np.full_like(mu_s, gap.mean())
    a = alpha
    return np.stack(
        [
            np.ones_like(mu_s),
            a,
            a * a,
            phi,
            gap / s,
            glayer / s,
            mu_s / s,
            mu_g / s,
            np.log(s + 1e-8),
            phi * a,
            sigma,
            (mu_s - mu_s.mean()) / s,
        ],
        axis=0,
    ).astype(np.float64)


def _hada_to_layer(hada, weights, seed: int, stop_li: int):
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
        for li in range(stop_li + 1):
            x = fnp.maximum(x @ weights[li], 0.0)
        mu = np.asarray(fnp.mean(x, axis=0), np.float64)
        # also finish without recenter for base final
        x2 = x
        for li in range(stop_li + 1, depth):
            x2 = fnp.maximum(x2 @ weights[li], 0.0)
        mu_final_base = np.asarray(fnp.mean(x2, axis=0), np.float64)
    return mu, mu_final_base, x, n_frames, e_r


def _finish_from(hada, x_pre, weights, start_li: int, delta: np.ndarray | None):
    """Continue from activations at start_li (already post-ReLU), optional recenter."""
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = x_pre
        if delta is not None:
            x = x + fnp.asarray(delta, dtype=fnp.float32)
        for li in range(start_li + 1, len(weights)):
            x = fnp.maximum(x @ weights[li], 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def main() -> None:
    hada = _load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    n = len(ds)
    width = 256
    Bases30 = np.zeros((n, width), np.float64)
    Feats = np.zeros((n, F, width), np.float64)
    Y30 = np.zeros((n, width), np.float64)
    Y31 = np.zeros((n, width), np.float64)
    Base31 = np.zeros((n, width), np.float64)
    # stash activations? too big. re-forward in OOS eval.

    for i in range(n):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        mu_s, mu_f, _x, _nf, _er = _hada_to_layer(hada, weights, seed, RECENTER_LAYER)
        mu_g, alpha, phi, sigma = _diag_gain_to_layer(
            [np.asarray(w, np.float32) for w in weights], RECENTER_LAYER
        )
        Bases30[i] = mu_s
        Feats[i] = _features(mu_s, mu_g, alpha, phi, sigma)
        Y30[i] = Y[RECENTER_LAYER]
        Y31[i] = Y[-1]
        Base31[i] = mu_f
        if i % 10 == 0:
            print(
                f"[{i}/{n}] L30 mse={np.mean((mu_s-Y[RECENTER_LAYER])**2):.3e} "
                f"L31 base={np.mean((mu_f-Y[-1])**2):.3e}",
                flush=True,
            )

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)

    base_l30 = float(np.mean((Bases30 - Y30) ** 2))
    base_l31 = float(np.mean((Base31 - Y31) ** 2))
    print(f"base L30 mse={base_l30:.4e}  base L31 mse={base_l31:.4e}")

    best = None
    for lam in [1e-1, 1e-2, 1e-3, 1e-4]:
        oos30 = np.zeros_like(Bases30)
        oos31 = np.zeros_like(Base31)
        for fi, val in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            mu = Feats[tr].mean(axis=(0, 2))
            sd = Feats[tr].std(axis=(0, 2)) + 1e-8
            sd[0] = 1.0
            Xtr = (Feats[tr] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            Rtr = Y30[tr] - Bases30[tr]
            Xt = Xtr.transpose(1, 0, 2).reshape(F, -1)
            yt = Rtr.reshape(-1)
            beta = np.linalg.solve(
                Xt @ Xt.T / Xt.shape[1] + lam * np.eye(F), Xt @ yt / Xt.shape[1]
            )
            Xva = (Feats[val] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
            for j, vi in enumerate(val):
                delta = beta @ Xva[j]
                oos30[vi] = Bases30[vi] + delta
                ex = ds[int(vi)]
                weights = [
                    fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)
                ]
                seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
                _mu_s, _mf, x_pre, _, _ = _hada_to_layer(
                    hada, weights, seed, RECENTER_LAYER
                )
                oos31[vi] = _finish_from(
                    hada, x_pre, weights, RECENTER_LAYER, delta.astype(np.float64)
                )
        m30 = float(np.mean((oos30 - Y30) ** 2))
        m31 = float(np.mean((oos31 - Y31) ** 2))
        print(
            f"lam={lam:.0e}  OOS L30={m30:.4e} (r={base_l30/m30:.3f})  "
            f"rollout L31={m31:.4e} (r={base_l31/m31:.3f})",
            flush=True,
        )
        if best is None or m31 < best[0]:
            best = (m31, lam, oos31)

    lam = best[1]
    mu = Feats.mean(axis=(0, 2))
    sd = Feats.std(axis=(0, 2)) + 1e-8
    sd[0] = 1.0
    X = (Feats - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
    Xt = X.transpose(1, 0, 2).reshape(F, -1)
    yt = (Y30 - Bases30).reshape(-1)
    beta = np.linalg.solve(
        Xt @ Xt.T / Xt.shape[1] + lam * np.eye(F), Xt @ yt / Xt.shape[1]
    )
    np.savez(
        OUT,
        beta=beta.astype(np.float64),
        feat_mu=mu.astype(np.float64),
        feat_sd=sd.astype(np.float64),
        lam=float(lam),
        recenter_layer=RECENTER_LAYER,
        oos_l31_mse=float(best[0]),
        base_l31_mse=base_l31,
        base_l30_mse=base_l30,
    )
    print(
        f"wrote {OUT} lam={lam} oos_L31={best[0]:.4e} "
        f"ratio={base_l31/best[0]:.3f}"
    )


if __name__ == "__main__":
    main()

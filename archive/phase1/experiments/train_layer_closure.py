#!/usr/bin/env python3
"""P3 prototype: shared per-layer linear closure on Mini all_layer_means.

Teacher-forcing: at each layer, features from diagonal gain-cov state → predict
true layer mean residual. 5-fold by MLP. Does NOT yet do rollout/DAgger.
"""

from __future__ import annotations

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
OUT = ROOT / "experiments" / "closure_layer_v030.npz"
INV = 1.0 / math.sqrt(2.0 * math.pi)


def _propagate_collect(weights: list, Y: np.ndarray):
    """Return per-layer analytic mu and features, plus targets Y[l]."""
    width = weights[0].shape[0]
    depth = len(weights)
    # Diagonal-only gain-cov for speed (offline Mini prototype).
    mu = fnp.zeros(width, dtype=fnp.float32)
    var = fnp.ones(width, dtype=fnp.float32)

    feats_all = []
    mu_all = []
    with flops.BudgetContext(flop_budget=int(1e15)):
        for li, w in enumerate(weights):
            W = np.asarray(w, np.float32)
            # var_pre_j = sum_i W_ij^2 * var_i  (diag approx)
            var_np = np.asarray(var, np.float32)
            mu_np0 = np.asarray(mu, np.float32)
            mu_pre = W.T @ mu_np0
            var_pre = np.maximum((W.T ** 2) @ var_np, 1e-12)
            sigma = np.sqrt(var_pre)
            alpha = mu_pre / sigma
            # use flopscope pdf/cdf on arrays
            alpha_f = fnp.asarray(alpha, dtype=fnp.float32)
            phi = np.asarray(flops.stats.norm.pdf(alpha_f), np.float64)
            Phi = np.asarray(flops.stats.norm.cdf(alpha_f), np.float64)
            mu_new = mu_pre * Phi + sigma * phi
            ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sigma * phi
            var_new = np.maximum(ez2 - mu_new * mu_new, 0.0)
            mu = fnp.asarray(mu_new, dtype=fnp.float32)
            var = fnp.asarray(var_new, dtype=fnp.float32)

            a = alpha.astype(np.float64)
            p = phi
            s = sigma.astype(np.float64)
            layer = np.full(width, (li + 1) / depth)
            row_n = np.sqrt(np.maximum(np.sum(W * W, axis=1), 1e-30))
            rn = np.full(width, float(row_n.mean()))
            feats = np.stack(
                [
                    np.ones(width),
                    a,
                    a * a,
                    p,
                    s,
                    np.log(s + 1e-8),
                    mu_new.astype(np.float64),
                    layer,
                    rn,
                    p * a,
                ],
                axis=0,
            )
            feats_all.append(feats)
            mu_all.append(mu_new.astype(np.float64))
    return np.stack(mu_all, 0), np.stack(feats_all, 0)  # (L,W), (L,F,W)


def main() -> None:
    ds = Dataset.load_from_disk(str(MINI))
    n = len(ds)
    L, W, F = 32, 256, 10
    Bases = np.zeros((n, L, W), np.float64)
    Feats = np.zeros((n, L, F, W), np.float64)
    Ys = np.zeros((n, L, W), np.float64)

    for i in range(n):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(L)]
        Y = np.asarray(ex["all_layer_means"], np.float64)
        mu, feats = _propagate_collect(weights, Y)
        Bases[i], Feats[i], Ys[i] = mu, feats, Y
        if i % 5 == 0:
            print(f"[{i}] final base mse={np.mean((mu[-1]-Y[-1])**2):.4e}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)

    def eval_lam(lam: float) -> float:
        oos = np.zeros_like(Bases)
        for fi, val in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            # pool all layers
            mu_f = Feats[tr].mean(axis=(0, 1, 3))
            sd_f = Feats[tr].std(axis=(0, 1, 3)) + 1e-8
            Xtr = (Feats[tr] - mu_f.reshape(1, 1, -1, 1)) / sd_f.reshape(1, 1, -1, 1)
            Rtr = Ys[tr] - Bases[tr]
            Xt = Xtr.transpose(2, 0, 1, 3).reshape(F, -1)
            yt = Rtr.reshape(-1)
            gram = Xt @ Xt.T / Xt.shape[1] + lam * np.eye(F)
            beta = np.linalg.solve(gram, Xt @ yt / Xt.shape[1])
            Xva = (Feats[val] - mu_f.reshape(1, 1, -1, 1)) / sd_f.reshape(1, 1, -1, 1)
            for j, vi in enumerate(val):
                # (L,F,W) → delta (L,W)
                d = np.einsum("f,lfw->lw", beta, Xva[j])
                oos[vi] = Bases[vi] + d
        return float(np.mean((oos[:, -1] - Ys[:, -1]) ** 2))

    base_final = float(np.mean((Bases[:, -1] - Ys[:, -1]) ** 2))
    print(f"analytic final mse={base_final:.4e}")
    best_lam, best_mse = None, 1e9
    for lam in [1e-2, 1e-3, 1e-4, 1e-5]:
        m = eval_lam(lam)
        print(f"lam={lam:.0e} oos_final={m:.4e} ratio={base_final/m:.4f}")
        if m < best_mse:
            best_mse, best_lam = m, lam

    # refit all
    lam = best_lam
    mu_f = Feats.mean(axis=(0, 1, 3))
    sd_f = Feats.std(axis=(0, 1, 3)) + 1e-8
    X = (Feats - mu_f.reshape(1, 1, -1, 1)) / sd_f.reshape(1, 1, -1, 1)
    Xt = X.transpose(2, 0, 1, 3).reshape(F, -1)
    yt = (Ys - Bases).reshape(-1)
    beta = np.linalg.solve(Xt @ Xt.T / Xt.shape[1] + lam * np.eye(F), Xt @ yt / Xt.shape[1])
    np.savez(
        OUT,
        beta=beta.astype(np.float64),
        feat_mu=mu_f.astype(np.float64),
        feat_sd=sd_f.astype(np.float64),
        lam=lam,
        oos_final_mse=best_mse,
        base_final_mse=base_final,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

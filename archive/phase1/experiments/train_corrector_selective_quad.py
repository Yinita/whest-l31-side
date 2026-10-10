"""Selective quadratic residual corrector from pooled cache — OOS gated."""
from __future__ import annotations

import itertools
import numpy as np
from pathlib import Path

CACHE = Path("experiments/corrector_cache_pooled_v0218.npz")
OUT = Path("experiments/corrector_selective_quad.npz")


def expand_quad(F):  # (N, f, W) -> (N, Fq, W)
    n, f, w = F.shape
    # base + all pairs i<=j
    pairs = list(itertools.combinations_with_replacement(range(f), 2))
    Fq = np.empty((n, f + len(pairs), w), dtype=np.float64)
    Fq[:, :f, :] = F
    for k, (i, j) in enumerate(pairs):
        Fq[:, f + k, :] = F[:, i, :] * F[:, j, :]
    return Fq, pairs


def standardize(X, mu=None, sd=None):
    # X: (nfeat, nobs)
    if mu is None:
        mu = X.mean(1)
        sd = X.std(1)
        sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu.reshape(-1, 1)) / sd.reshape(-1, 1), mu, sd


def fit_ridge(X, y, lam):
    # X (p, n), y (n,)
    p, n = X.shape
    G = (X @ X.T) / n + lam * np.eye(p)
    b = (X @ y) / n
    return np.linalg.solve(G, b)


def oos_mse(Fq, Sigmas, bases, Ys, lam, scale, n_folds=5, active=None):
    n, fq, w = Fq.shape
    idx = np.arange(n)
    folds = np.array_split(idx, n_folds)
    pred_err = []
    base_err = []
    for i, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        # flatten train units
        Xtr = Fq[tr][:, active, :].transpose(1, 0, 2).reshape(len(active), -1)
        ytr = ((Ys[tr] - bases[tr]) / np.maximum(Sigmas[tr], 1e-12)).reshape(-1)
        Xtr, mu, sd = standardize(Xtr)
        beta = fit_ridge(Xtr, ytr, lam)
        # test
        Xte = Fq[te][:, active, :].transpose(1, 0, 2).reshape(len(active), -1)
        Xte, _, _ = standardize(Xte, mu, sd)
        delta = scale * Sigmas[te].reshape(-1) * (beta @ Xte)
        pred = bases[te].reshape(-1) + delta
        yt = Ys[te].reshape(-1)
        pred_err.append(np.mean((pred - yt) ** 2))
        base_err.append(np.mean((bases[te].reshape(-1) - yt) ** 2))
    pe, be = float(np.mean(pred_err)), float(np.mean(base_err))
    return pe, be, be / pe if pe > 0 else 0.0


def main():
    z = np.load(CACHE)
    bases, Feats, Sigmas, Ys = z["bases"], z["Feats"], z["Sigmas"], z["Ys"]
    # use first 300 train-like; hold 200 for final? Cache is 500 — use all with 5-fold
    print("cache", bases.shape, "base_mse", np.mean((bases - Ys) ** 2))

    # linear 14 baseline
    F = Feats
    active14 = list(range(14))
    for lam in [1e-3, 3e-3, 1e-2]:
        pe, be, r = oos_mse(F, Sigmas, bases, Ys, lam, 1.0, active=active14)
        print(f"linear14 lam={lam:.0e} oos={pe:.4e} ratio={r:.4f}")

    Fq, pairs = expand_quad(F)
    print("quad dim", Fq.shape[1], "pairs", len(pairs))

    # Stage1: univariate score per feature on fold0
    n, fq, w = Fq.shape
    # correlation of each feature with residual/sigma
    resid = ((Ys - bases) / np.maximum(Sigmas, 1e-12)).reshape(n, 1, w)
    scores = []
    for j in range(fq):
        x = Fq[:, j, :]
        # flatten
        c = np.corrcoef(x.ravel(), resid.ravel())[0, 1]
        scores.append(abs(c) if np.isfinite(c) else 0.0)
    scores = np.asarray(scores)
    order = np.argsort(-scores)

    # Always keep linear 14; add top quad pairs
    lin = list(range(14))
    quad_idx = [j for j in order if j >= 14]
    for k_extra in [0, 5, 10, 20, 40, 60, 106]:  # 106 = all pairs
        active = lin + quad_idx[:k_extra]
        best = None
        for lam in [3e-4, 1e-3, 3e-3, 1e-2]:
            for scale in [0.85, 1.0, 1.15]:
                pe, be, r = oos_mse(Fq, Sigmas, bases, Ys, lam, scale, active=active)
                if best is None or pe < best[0]:
                    best = (pe, be, r, lam, scale)
        print(
            f"k+={k_extra:3d} dim={len(active):3d} best_oos={best[0]:.4e} "
            f"ratio={best[2]:.4f} lam={best[3]} scale={best[4]}"
        )

    # Fit final on best small-k for ship safety (prefer k<=20 with ratio gain)
    # Refit choosing by ratio with k=20
    active = lin + quad_idx[:20]
    pe, be, r, lam, scale = None, None, None, 1e-3, 1.0
    best = None
    for lam_ in [3e-4, 1e-3, 3e-3]:
        for scale_ in [0.9, 1.0, 1.1]:
            pe_, be_, r_ = oos_mse(Fq, Sigmas, bases, Ys, lam_, scale_, active=active)
            if best is None or pe_ < best[0]:
                best = (pe_, be_, r_, lam_, scale_)
    lam, scale = best[3], best[4]
    print("FINAL select", best)

    # Fit on all data
    X = Fq[:, active, :].transpose(1, 0, 2).reshape(len(active), -1)
    y = ((Ys - bases) / np.maximum(Sigmas, 1e-12)).reshape(-1)
    Xs, mu, sd = standardize(X)
    beta = fit_ridge(Xs, y, lam)
    np.savez(
        OUT,
        beta=beta.astype(np.float64),
        mu=mu.astype(np.float64),
        sd=sd.astype(np.float64),
        active=np.asarray(active, np.int32),
        pairs=np.asarray(pairs, np.int32),
        lam=lam,
        scale=scale,
        oos_mse=best[0],
        oos_ratio=best[2],
    )
    print("wrote", OUT, "dim", len(active), "scale", scale, "lam", lam)


if __name__ == "__main__":
    main()

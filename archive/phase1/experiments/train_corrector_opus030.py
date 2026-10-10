"""opus-0.3.0s corrector: 15 base feats (14 + sigma-scale) + pairwise quadratic
interactions (120-dim), ridge, OOS-selected lam/scale on cached 450-MLP synth set.

Reuses experiments/corrector_cache_pooled_v0218.npz (bases/Feats/Sigmas/Ys built
with the pooled dual-stream base = same base as v0.2.20s), so NO re-extraction.

Feature order MUST match estimator.py exactly:
  linear[0..14] = F0..F13 (current 14) then F14 = sig-scale = s/mean(s)
  then products F[a]*F[b] for a in 1..14, b in a..14   -> 105
  total 120.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "corrector_cache_pooled_v0218.npz"
OUT = ROOT / "experiments" / "corrector_opus030.npz"


def build_feats(Feats: np.ndarray, Sig: np.ndarray) -> np.ndarray:
    """(N,14,W) + (N,W) -> (N,120,W). Deterministic order; mirror in estimator."""
    n, k, w = Feats.shape
    sig_scale = (Sig / (Sig.mean(axis=1, keepdims=True) + 1e-12))[:, None, :]  # (N,1,W)
    F = np.concatenate([Feats, sig_scale], axis=1)  # (N,15,W)
    K = F.shape[1]
    cols = [F[:, j, :] for j in range(K)]  # 15 linear
    for a in range(1, K):
        for b in range(a, K):
            cols.append(F[:, a, :] * F[:, b, :])  # 105 quadratic
    return np.stack(cols, axis=1)  # (N,120,W)


def fit(tr, FM, Sig, Ys, bases, lam):
    X = np.concatenate([FM[i].T for i in tr], 0)
    y = np.concatenate([(Ys[i] - bases[i]) / (Sig[i] + 1e-8) for i in tr], 0)
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    mu[0] = 0.0
    sd[0] = 1.0
    Xs = (X - mu) / sd
    F = Xs.shape[1]
    beta = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(F), Xs.T @ y)
    return beta, mu, sd


def apply(beta, mu, sd, i, FM, Sig, bases, sc):
    return bases[i] + sc * Sig[i] * (((FM[i].T - mu) / sd) @ beta)


def main():
    d = np.load(CACHE)
    bases = d["bases"].astype(np.float64)
    Feats = d["Feats"].astype(np.float64)
    Sig = d["Sigmas"].astype(np.float64)
    Ys = d["Ys"].astype(np.float64)
    N = bases.shape[0]
    FM = build_feats(Feats, Sig)
    print(f"N={N} feat_dim={FM.shape[1]} n_gt={int(d['n_gt'])}", flush=True)

    base_mse = float(np.mean([((bases[i] - Ys[i]) ** 2).mean() for i in range(N)]))
    folds = np.array_split(np.arange(N), 5)

    best = None
    for lam in [1e-3, 3e-3, 1e-2, 3e-2]:
        for sc in [0.85, 1.0, 1.15]:
            c = []
            for fi, te in enumerate(folds):
                tr = np.concatenate([folds[j] for j in range(5) if j != fi])
                beta, mu, sd = fit(tr, FM, Sig, Ys, bases, lam)
                for i in te:
                    c.append(((apply(beta, mu, sd, i, FM, Sig, bases, sc) - Ys[i]) ** 2).mean())
            c = float(np.mean(c))
            print(f"OOS lam={lam:.0e} sc={sc:.2f} corr={c:.4e} ratio={base_mse/c:.4f}", flush=True)
            if best is None or c < best[0]:
                best = (c, lam, sc)

    _, lam, sc = best
    print(f"BEST lam={lam:.0e} sc={sc:.2f} oos_ratio={base_mse/best[0]:.4f} base={base_mse:.4e}", flush=True)
    beta, mu, sd = fit(np.arange(N), FM, Sig, Ys, bases, lam)
    np.savez(
        OUT,
        beta=beta.astype(np.float64),
        feat_mu=mu.astype(np.float64),
        feat_sd=sd.astype(np.float64),
        lam=np.asarray(lam),
        scale=np.asarray(sc),
        oos_ratio=np.asarray(base_mse / best[0]),
        base_mse=np.asarray(base_mse),
        feat_dim=np.asarray(FM.shape[1]),
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

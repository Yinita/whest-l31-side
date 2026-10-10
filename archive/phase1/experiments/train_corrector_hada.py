"""Train residual corrector on Hadamard+gain-blend base (pure-0.2.19s).

Synth OOS gate; do NOT Mini-fit. Cache under experiments/corrector_cache_hada.npz.
"""
from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, sample_mlp

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hada", ROOT / "experiments" / "pure_blend_097.py"
)
hada = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(hada)

B = int(2.72e11)
WIDTH, DEPTH = 256, 32
N_GT = 60_000
N_TRAIN = 80
CACHE = ROOT / "experiments" / "corrector_cache_hada.npz"
OUT = ROOT / "experiments" / "corrector_hada.npz"
INV = 1.0 / math.sqrt(2.0 * math.pi)


def _gt_final(mlp: MLP, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, mlp.width)).astype(np.float32)
    for w in mlp.weights:
        x = np.maximum(x @ np.asarray(w, np.float32), 0.0)
    return x.mean(axis=0).astype(np.float64)


def _predict_base_and_feats(mlp: MLP):
    """Return pred (before θ), sigma scale, and 14 radiant-ish features (F, W)."""
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(
        float(B) * hada._UTIL * hada._UTIL_CALIB - hada._GAIN_RESERVE, flops_per
    )
    n_forward = int(max(2 * width, min(target / flops_per, hada._MAX_SAMPLES)))
    n_frames = max(1, n_forward // (2 * width))
    e_r = hada._chi_mean_radius(width)
    seed0 = int(mlp.seed) & 0x7FFFFFFF
    x = hada._hadamard_directions(n_frames, width, seed=seed0, e_r=e_r)
    h = x
    for w in weights[:-1]:
        h = fnp.maximum(h @ w, 0.0)
    zpre = h @ weights[-1]
    last = fnp.maximum(zpre, 0.0)
    mu_h = fnp.mean(last, axis=0)
    mu_g_raw = hada._gain_cov_final(weights)
    # last-layer α,φ,σ from gain path (reuse gain internals cheaply)
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = fnp.eye(width, dtype=fnp.float32)
    alpha_g = phi_g = sig_g = None
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sig_g = fnp.sqrt(var_pre)
        alpha_g = mu_pre / sig_g
        phi_g = flops.stats.norm.pdf(alpha_g)
        Phi = flops.stats.norm.cdf(alpha_g)
        mu = mu_pre * Phi + sig_g * phi_g
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig_g * phi_g
        var_post = fnp.maximum(ez2 - mu * mu, 0.0)
        gain = fnp.where(sig_g > 1e-12, Phi, fnp.zeros_like(Phi))
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
    mu_g = hada._GAIN_SCALE * mu
    pred = hada._BLEND * mu_h + (1.0 - hada._BLEND) * mu_g

    zc = zpre - fnp.mean(zpre, axis=0)
    m2 = fnp.maximum(fnp.mean(zc * zc, axis=0), 1e-12)
    m3 = fnp.mean(zc * zc * zc, axis=0)
    m4 = fnp.mean(zc * zc * zc * zc, axis=0)
    s = fnp.sqrt(m2)
    skew = m3 / (m2 * s)
    kurt = m4 / (m2 * m2) - 3.0
    alpha_s = fnp.mean(zpre, axis=0) / s
    phi_s = fnp.exp(-0.5 * alpha_s * alpha_s) * INV
    st = -(skew / 6.0) * phi_s
    kt = (kurt / 24.0) * (alpha_s * alpha_s - 1.0) * phi_s
    gap = mu_h - mu_g
    glayer = fnp.full(width, float(fnp.mean(gap)), dtype=fnp.float32)
    a = alpha_g
    feats = fnp.stack(
        [
            fnp.ones(width, dtype=fnp.float32),
            a,
            a * a,
            a * a * a,
            a * a * a * a,
            phi_g,
            gap / (sig_g + 1e-8),
            glayer / (sig_g + 1e-8),
            st,
            kt,
            (mu_h - pred) / (s + 1e-8),
            (mu_g - pred) / (s + 1e-8),
            alpha_s,
            phi_s,
        ],
        axis=0,
    )
    return (
        np.asarray(pred, np.float64),
        np.asarray(s, np.float64),
        np.asarray(feats, np.float64),
    )


def _fit_ridge(X, y, lam):
    p, n = X.shape
    G = (X @ X.T) / n + lam * np.eye(p)
    b = (X @ y) / n
    return np.linalg.solve(G, b)


def _stdize(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean(1)
        sd = X.std(1)
        sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu.reshape(-1, 1)) / sd.reshape(-1, 1), mu, sd


def oos_eval(Feats, Sigmas, bases, Ys, lam, scale, n_folds=5):
    n = bases.shape[0]
    folds = np.array_split(np.arange(n), n_folds)
    pe, be = [], []
    for i, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        Xtr = Feats[tr].transpose(1, 0, 2).reshape(Feats.shape[1], -1)
        ytr = ((Ys[tr] - bases[tr]) / np.maximum(Sigmas[tr], 1e-12)).reshape(-1)
        Xtr, mu, sd = _stdize(Xtr)
        beta = _fit_ridge(Xtr, ytr, lam)
        Xte = Feats[te].transpose(1, 0, 2).reshape(Feats.shape[1], -1)
        Xte, _, _ = _stdize(Xte, mu, sd)
        pred = bases[te].ravel() + scale * Sigmas[te].ravel() * (beta @ Xte)
        pe.append(np.mean((pred - Ys[te].ravel()) ** 2))
        be.append(np.mean((bases[te].ravel() - Ys[te].ravel()) ** 2))
    pe, be = float(np.mean(pe)), float(np.mean(be))
    return pe, be, be / pe if pe > 0 else 0.0


def build_cache():
    if CACHE.exists():
        z = np.load(CACHE)
        print("loaded cache", z["bases"].shape, flush=True)
        return z
    print(f"building cache N_TRAIN={N_TRAIN} N_GT={N_GT}", flush=True)
    bases, Feats, Sigmas, Ys = [], [], [], []
    t0 = time.time()
    for i in range(N_TRAIN):
        seed = 10_000 + i * 9973
        mlp = sample_mlp(width=WIDTH, depth=DEPTH, seed=seed)
        # ensure MLP object
        W = [fnp.asarray(w, dtype=fnp.float32) for w in mlp.weights]
        mlp = MLP(width=WIDTH, depth=DEPTH, weights=W, seed=seed)
        pred, s, feats = _predict_base_and_feats(mlp)
        gt = _gt_final(mlp, N_GT, seed=seed + 1)
        bases.append(pred)
        Sigmas.append(s)
        Feats.append(feats)
        Ys.append(gt)
        if (i + 1) % 10 == 0:
            print(f"  cache {i+1}/{N_TRAIN} t={time.time()-t0:.0f}", flush=True)
    bases = np.stack(bases)
    Feats = np.stack(Feats)
    Sigmas = np.stack(Sigmas)
    Ys = np.stack(Ys)
    np.savez(CACHE, bases=bases, Feats=Feats, Sigmas=Sigmas, Ys=Ys, n_gt=N_GT)
    print("wrote", CACHE, "base_mse", float(np.mean((bases - Ys) ** 2)), flush=True)
    return np.load(CACHE)


def main():
    print("start train_corrector_hada", flush=True)
    z = build_cache()
    bases, Feats, Sigmas, Ys = z["bases"], z["Feats"], z["Sigmas"], z["Ys"]
    print("base_mse", float(np.mean((bases - Ys) ** 2)), flush=True)
    best = None
    for lam in [3e-4, 1e-3, 3e-3, 1e-2]:
        for scale in [0.85, 1.0, 1.15]:
            pe, be, r = oos_eval(Feats, Sigmas, bases, Ys, lam, scale)
            print(f"lam={lam:.0e} scale={scale:.2f} oos={pe:.4e} ratio={r:.4f}", flush=True)
            if best is None or pe < best[0]:
                best = (pe, be, r, lam, scale)
    print("BEST", best, flush=True)
    lam, scale = best[3], best[4]
    X = Feats.transpose(1, 0, 2).reshape(Feats.shape[1], -1)
    y = ((Ys - bases) / np.maximum(Sigmas, 1e-12)).reshape(-1)
    Xs, mu, sd = _stdize(X)
    beta = _fit_ridge(Xs, y, lam)
    np.savez(
        OUT,
        beta=beta,
        mu=mu,
        sd=sd,
        lam=lam,
        scale=scale,
        oos_mse=best[0],
        oos_ratio=best[2],
    )
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

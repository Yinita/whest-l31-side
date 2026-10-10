"""v0.2.15s training: larger synth cache + λ/α/feature ablations.

Caches (bases, Feats, Sigmas, Ys) so sweeps are cheap after the first extract.
Gate on synth 5-fold OOS; Mini transfer only for winners.
"""
from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, load_dataset, sample_mlp

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "base12", ROOT / "experiments" / "v0212s_floor094.py"
)
base = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(base)

B = int(2.72e11)
WIDTH, DEPTH = 256, 32
N_GT = 200_000
N_TRAIN = 400
CACHE = ROOT / "experiments" / "corrector_cache_v0215.npz"
OUT = ROOT / "experiments" / "corrector_v0215.npz"
INV = base._INV_SQRT_2PI
FEAT_NAMES = [
    "1",
    "a",
    "a2",
    "a3",
    "a4",
    "phi_g",
    "gap_s",
    "glayer_s",
    "st",
    "kt",
    "cv_gap",
    "g_gap",
    "alpha_s",
    "phi_s",
]


def _gt_final(mlp: MLP, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, mlp.width)).astype(np.float32)
    for w in mlp.weights:
        x = np.maximum(x @ np.asarray(w, np.float32), 0.0)
    return x.mean(axis=0).astype(np.float64)


def _extract(mlp: MLP):
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    blend = base._BLEND_RQMC
    reserve = base._GAIN_RESERVE
    flops_per = depth * base._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * base._FLOOR_FRAC - reserve, flops_per)
    n_samples = int(max(64, min(target / flops_per, 200_000)))
    n_half = max(32, n_samples // 2)

    x = base._rqmc_standard_normal(n_half, width, seed=int(mlp.seed) & 0x7FFFFFFF)
    x = fnp.concatenate([x, -x], axis=0)
    w0 = weights[0]
    a1 = fnp.maximum(x @ w0, 0.0)
    col_norm = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-30))
    e_a1 = col_norm * INV
    q1 = fnp.sum(a1 * a1, axis=1)
    e_q1 = 0.5 * float(fnp.sum(col_norm * col_norm))
    controls = fnp.concatenate([a1, q1.reshape(-1, 1)], axis=1)
    expectations = fnp.concatenate(
        [e_a1, fnp.asarray([e_q1], dtype=e_a1.dtype)], axis=0
    )
    h = a1
    for w in weights[1:-1]:
        h = fnp.maximum(h @ w, 0.0)
    zpre = h @ weights[-1]
    last = fnp.maximum(zpre, 0.0)
    mu_cv = np.asarray(
        base._multi_cv_mean(last, controls, expectations, base._RIDGE_REL), float
    )

    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = fnp.eye(width, dtype=fnp.float32)
    alpha_g = phi_g = sig_g = None
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
        alpha_g = np.asarray(alpha, float)
        phi_g = np.asarray(phi, float)
        sig_g = np.asarray(sigma, float)
    mu_g = base._GAIN_SCALE * np.asarray(mu, float)
    pred = blend * mu_cv + (1.0 - blend) * mu_g

    zp = np.asarray(zpre, float)
    zc = zp - zp.mean(0)
    m2 = np.maximum((zc**2).mean(0), 1e-12)
    m3 = (zc**3).mean(0)
    m4 = (zc**4).mean(0)
    s = np.sqrt(m2)
    skew = m3 / (m2 * s)
    kurt = m4 / (m2 * m2) - 3.0
    alpha_s = zp.mean(0) / s
    phi_s = np.exp(-0.5 * alpha_s * alpha_s) / math.sqrt(2 * math.pi)
    st = -(skew / 6.0) * phi_s
    kt = (kurt / 24.0) * (alpha_s**2 - 1.0) * phi_s
    gap = mu_cv - mu_g
    glayer = np.full(width, float(gap.mean()))
    a = alpha_g
    feats = np.stack(
        [
            np.ones(width),
            a,
            a**2,
            a**3,
            a**4,
            phi_g,
            gap / (sig_g + 1e-8),
            glayer / (sig_g + 1e-8),
            st,
            kt,
            (mu_cv - pred) / (s + 1e-8),
            (mu_g - pred) / (s + 1e-8),
            alpha_s,
            phi_s,
        ],
        0,
    )
    return pred, feats, s


def _build_cache():
    if CACHE.is_file():
        d = np.load(CACHE)
        if int(d["n_train"]) == N_TRAIN and int(d["n_gt"]) == N_GT:
            print(f"loaded cache {CACHE}", flush=True)
            return d["bases"], d["Feats"], d["Sigmas"], d["Ys"]
    t0 = time.time()
    print(f"building cache N={N_TRAIN} GT={N_GT}", flush=True)
    bases, Feats, Sigmas, Ys = [], [], [], []
    for i in range(N_TRAIN):
        rng = fnp.random.default_rng(80_000 + i)
        mlp = sample_mlp(width=WIDTH, depth=DEPTH, rng=rng, seed=90_000 + i)
        y = _gt_final(mlp, N_GT, seed=100_000 + i)
        with flops.BudgetContext(flop_budget=B, quiet=True):
            pred, feats, sig = _extract(mlp)
        bases.append(pred)
        Feats.append(feats)
        Sigmas.append(sig)
        Ys.append(y)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{N_TRAIN} t={time.time()-t0:.0f}s", flush=True)
    bases = np.stack(bases)
    Feats = np.stack(Feats)
    Sigmas = np.stack(Sigmas)
    Ys = np.stack(Ys)
    np.savez(
        CACHE,
        bases=bases,
        Feats=Feats,
        Sigmas=Sigmas,
        Ys=Ys,
        n_train=np.asarray(N_TRAIN),
        n_gt=np.asarray(N_GT),
    )
    print(f"wrote {CACHE} base_mse={float(np.mean((bases-Ys)**2)):.4e}", flush=True)
    return bases, Feats, Sigmas, Ys


def _fit(tr_idx, Feats, Sigmas, Ys, bases, lam, feat_idx):
    X, y = [], []
    for i in tr_idx:
        X.append(Feats[i][feat_idx].T)
        y.append((Ys[i] - bases[i]) / (Sigmas[i] + 1e-8))
    X = np.concatenate(X, 0)
    y = np.concatenate(y, 0)
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    if 0 in feat_idx:
        # intercept column index within selected feats
        j0 = list(feat_idx).index(0)
        mu[j0] = 0.0
        sd[j0] = 1.0
    Xs = (X - mu) / sd
    F = Xs.shape[1]
    beta = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(F), Xs.T @ y)
    return beta, mu, sd


def _apply(beta, mu, sd, base_i, feats_i, sig_i, feat_idx, scale):
    Xi = (feats_i[feat_idx].T - mu) / sd
    return base_i + scale * sig_i * (Xi @ beta)


def _oos(bases, Feats, Sigmas, Ys, lam, feat_idx, scale, n_folds=5):
    idx = np.arange(len(bases))
    folds = np.array_split(idx, n_folds)
    oos_b, oos_c = [], []
    for fi, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        beta, mu, sd = _fit(tr, Feats, Sigmas, Ys, bases, lam, feat_idx)
        for i in te:
            oos_b.append(float(np.mean((bases[i] - Ys[i]) ** 2)))
            oos_c.append(
                float(
                    np.mean(
                        (
                            _apply(beta, mu, sd, bases[i], Feats[i], Sigmas[i], feat_idx, scale)
                            - Ys[i]
                        )
                        ** 2
                    )
                )
            )
    b, c = float(np.mean(oos_b)), float(np.mean(oos_c))
    return b, c, b / c


def main():
    bases, Feats, Sigmas, Ys = _build_cache()
    F = Feats.shape[1]
    all_idx = np.arange(F)

    # feature subsets
    subsets = {
        "all": all_idx,
        "no_a34": np.array([i for i in all_idx if i not in (3, 4)]),
        "no_edge": np.array([i for i in all_idx if i not in (8, 9)]),
        "no_sample": np.array([i for i in all_idx if i not in (12, 13)]),
        "core": np.array([0, 1, 2, 5, 6, 7, 10, 11]),  # 1,a,a2,phi,gaps
        "radiant_min": np.array([0, 1, 2, 3, 4, 5, 6, 7]),  # α powers + gaps
    }

    print("=== sweeps ===", flush=True)
    rows = []
    for name, feat_idx in subsets.items():
        for lam in [3e-3, 1e-2, 3e-2, 1e-1]:
            for scale in [0.5, 0.7, 0.85, 1.0]:
                b, c, ratio = _oos(bases, Feats, Sigmas, Ys, lam, feat_idx, scale)
                rows.append((c, ratio, name, lam, scale, b))
    rows.sort()
    print("top 12 by OOS corr mse:", flush=True)
    for c, ratio, name, lam, scale, b in rows[:12]:
        print(
            f"  {name:12s} lam={lam:.0e} α={scale:.2f} "
            f"base={b:.4e} corr={c:.4e} ratio={ratio:.4f}",
            flush=True,
        )

    best = rows[0]
    _, ratio, name, lam, scale, _ = best
    feat_idx = subsets[name]
    print(f"BEST {name} lam={lam} scale={scale} ratio={ratio:.4f}", flush=True)

    beta, mu, sd = _fit(np.arange(len(bases)), Feats, Sigmas, Ys, bases, lam, feat_idx)

    # Mini transfer for top-3 distinct configs
    ds = load_dataset(
        "aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini"
    )
    mini_rows = []
    for i in range(100):
        r = ds[i]
        W = [fnp.asarray(w, dtype=fnp.float32) for w in r["weights"]]
        mlp = MLP(width=256, depth=len(W), weights=W, seed=int(r["mlp_seed"]))
        y = np.asarray(r["final_means"], np.float64)
        with flops.BudgetContext(flop_budget=B, quiet=True):
            pred, feats, sig = _extract(mlp)
        mini_rows.append((pred, feats, sig, y))
        if (i + 1) % 25 == 0:
            print(f"  mini extract {i+1}", flush=True)

    def mini_eval(tag, feat_idx, lam, scale):
        beta, mu, sd = _fit(np.arange(len(bases)), Feats, Sigmas, Ys, bases, lam, feat_idx)
        mb, mc = [], []
        for pred, feats, sig, y in mini_rows:
            corr = _apply(beta, mu, sd, pred, feats, sig, feat_idx, scale)
            mb.append(float(np.mean((pred - y) ** 2)))
            mc.append(float(np.mean((corr - y) ** 2)))
        b, c = float(np.mean(mb)), float(np.mean(mc))
        print(
            f"MINI {tag:20s} base={b:.4e} corr={c:.4e} ratio={b/c:.4f} adj≈{c*0.1001:.4e}",
            flush=True,
        )
        return c, beta, mu, sd

    # evaluate best + a few runners-up with distinct (name,lam,scale)
    seen = set()
    winners = []
    for c, ratio, name, lam, scale, b in rows:
        key = (name, lam, scale)
        if key in seen:
            continue
        seen.add(key)
        mc, beta, mu, sd = mini_eval(f"{name}/l{lam}/a{scale}", subsets[name], lam, scale)
        winners.append((mc, name, lam, scale, beta, mu, sd, subsets[name], ratio))
        if len(winners) >= 5:
            break
    winners.sort()
    mc, name, lam, scale, beta, mu, sd, feat_idx, oos_r = winners[0]
    print(
        f"SHIP candidate {name} lam={lam} scale={scale} mini_corr={mc:.4e} oos_r={oos_r:.4f}",
        flush=True,
    )

    # expand beta/mu/sd back to full-F with zeros for dropped feats
    F = Feats.shape[1]
    beta_full = np.zeros(F)
    mu_full = np.zeros(F)
    sd_full = np.ones(F)
    for j, fi in enumerate(feat_idx):
        beta_full[fi] = beta[j]
        mu_full[fi] = mu[j]
        sd_full[fi] = sd[j]
    # intercept handling
    mu_full[0] = 0.0
    sd_full[0] = 1.0

    np.savez(
        OUT,
        beta=beta_full,
        feat_mu=mu_full,
        feat_sd=sd_full,
        feat_idx=feat_idx,
        lam=np.asarray(lam),
        scale=np.asarray(scale),
        subset=np.asarray(name),
        oos_ratio=np.asarray(oos_r),
        mini_corr=np.asarray(mc),
        feat_names=np.asarray(FEAT_NAMES, dtype=object),
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

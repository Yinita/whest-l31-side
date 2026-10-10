"""Train radiant-style offline corrector on synth MLPs; transfer-eval Mini.

Fits θ on noisy predict-time features (not converged). Gate on synth
5-fold OOS before trusting Mini transfer.
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
    "base", ROOT / "experiments" / "v0212s_floor094.py"
)
base = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(base)

B = int(2.72e11)
WIDTH, DEPTH = 256, 32
N_GT = 100_000
N_TRAIN = 200
RIDGE = 3e-2
OUT = ROOT / "experiments" / "corrector_v0214.npz"
INV = base._INV_SQRT_2PI


def _gt_final(mlp: MLP, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, mlp.width)).astype(np.float32)
    for w in mlp.weights:
        x = np.maximum(x @ np.asarray(w, np.float32), 0.0)
    return x.mean(axis=0).astype(np.float64)


def _extract(mlp: MLP):
    """Base pred + radiant-ish features from same budgeted forward."""
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

    # gain path + last alpha
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
    )  # (F, width)
    return pred, feats, s


def _fit(tr_idx, Feats, Sigmas, Ys, bases, lam):
    X, y = [], []
    for i in tr_idx:
        X.append(Feats[i].T)
        y.append((Ys[i] - bases[i]) / (Sigmas[i] + 1e-8))
    X = np.concatenate(X, 0)
    y = np.concatenate(y, 0)
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    mu[0] = 0.0
    sd[0] = 1.0
    Xs = (X - mu) / sd
    F = Xs.shape[1]
    beta = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(F), Xs.T @ y)
    return beta, mu, sd


def _apply(beta, mu, sd, base_i, feats_i, sig_i):
    Xi = (feats_i.T - mu) / sd
    return base_i + sig_i * (Xi @ beta)


def main():
    t0 = time.time()
    print(f"synth {N_TRAIN} MLPs GT={N_GT}", flush=True)
    bases, Feats, Sigmas, Ys = [], [], [], []
    for i in range(N_TRAIN):
        rng = fnp.random.default_rng(50_000 + i)
        mlp = sample_mlp(width=WIDTH, depth=DEPTH, rng=rng, seed=60_000 + i)
        y = _gt_final(mlp, N_GT, seed=70_000 + i)
        with flops.BudgetContext(flop_budget=B, quiet=True):
            pred, feats, sig = _extract(mlp)
        bases.append(pred)
        Feats.append(feats)
        Sigmas.append(sig)
        Ys.append(y)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N_TRAIN} t={time.time()-t0:.0f}s", flush=True)
    bases = np.stack(bases)
    Feats = np.stack(Feats)
    Sigmas = np.stack(Sigmas)
    Ys = np.stack(Ys)
    print("base mse", float(np.mean((bases - Ys) ** 2)), flush=True)

    folds = np.array_split(np.arange(N_TRAIN), 5)
    best = None
    for lam in [1e-2, 3e-2, 1e-1]:
        oos_b, oos_c = [], []
        for fi, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            beta, mu, sd = _fit(tr, Feats, Sigmas, Ys, bases, lam)
            for i in te:
                oos_b.append(float(np.mean((bases[i] - Ys[i]) ** 2)))
                oos_c.append(
                    float(np.mean((_apply(beta, mu, sd, bases[i], Feats[i], Sigmas[i]) - Ys[i]) ** 2))
                )
        b, c = float(np.mean(oos_b)), float(np.mean(oos_c))
        ratio = b / c
        print(f"OOS lam={lam:.0e} base={b:.4e} corr={c:.4e} ratio={ratio:.4f}", flush=True)
        if best is None or c < best[0]:
            best = (c, lam, ratio)

    lam = best[1]
    print("chosen lam", lam, "best OOS ratio", best[2], flush=True)
    beta, mu, sd = _fit(np.arange(N_TRAIN), Feats, Sigmas, Ys, bases, lam)

    # Mini transfer
    ds = load_dataset(
        "aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini"
    )
    mb, mc = [], []
    for i in range(100):
        r = ds[i]
        W = [fnp.asarray(w, dtype=fnp.float32) for w in r["weights"]]
        mlp = MLP(width=256, depth=len(W), weights=W, seed=int(r["mlp_seed"]))
        y = np.asarray(r["final_means"], np.float64)
        with flops.BudgetContext(flop_budget=B, quiet=True):
            pred, feats, sig = _extract(mlp)
        corr = _apply(beta, mu, sd, pred, feats, sig)
        mb.append(float(np.mean((pred - y) ** 2)))
        mc.append(float(np.mean((corr - y) ** 2)))
        if (i + 1) % 25 == 0:
            print(f"  mini {i+1}", flush=True)
    b, c = float(np.mean(mb)), float(np.mean(mc))
    print(
        f"MINI transfer base={b:.4e} corr={c:.4e} ratio={b/c:.4f} "
        f"adj≈{c*0.1005:.4e}",
        flush=True,
    )
    np.savez(
        OUT,
        beta=beta,
        feat_mu=mu,
        feat_sd=sd,
        lam=np.asarray(lam),
        oos_ratio=np.asarray(best[2]),
        mini_ratio=np.asarray(b / c),
    )
    print(f"wrote {OUT} t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

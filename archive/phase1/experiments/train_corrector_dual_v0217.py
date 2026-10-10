"""Train corrector on dual-stream v0.2.15s base features (for v0.2.17s)."""
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
# dual base without corrector
SPEC = importlib.util.spec_from_file_location(
    "dual", ROOT / "experiments" / "v0215s_dual.py"
)
dual = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(dual)

B = int(2.72e11)
WIDTH, DEPTH = 256, 32
N_GT = 150_000
N_TRAIN = 300
CACHE = ROOT / "experiments" / "corrector_cache_dual_v0217.npz"
OUT = ROOT / "experiments" / "corrector_v0217.npz"
INV = dual._INV_SQRT_2PI


def _gt_final(mlp: MLP, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, mlp.width)).astype(np.float32)
    for w in mlp.weights:
        x = np.maximum(x @ np.asarray(w, np.float32), 0.0)
    return x.mean(axis=0).astype(np.float64)


def _extract_dual(mlp: MLP):
    """Mirror v0.2.16s feature path but return pred BEFORE corrector."""
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    blend = dual._BLEND_RQMC
    reserve = dual._GAIN_RESERVE
    flops_per = depth * dual._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * dual._FLOOR_FRAC - reserve, flops_per)
    n_samples = int(max(64, min(target / flops_per, 200_000)))
    n_half = max(32, n_samples // 4)
    seed0 = int(mlp.seed) & 0x7FFFFFFF

    def _stream(n_half: int, seed: int):
        x = dual._rqmc_standard_normal(n_half, width, seed=seed)
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
            dual._multi_cv_mean(last, controls, expectations, dual._RIDGE_REL), float
        )
        return mu_cv, np.asarray(zpre, float)

    mu1, z1 = _stream(n_half, seed0)
    mu2, z2 = _stream(n_half, seed0 ^ 0xA5A5A5A5)
    mu_cv = 0.5 * (mu1 + mu2)
    zpre = np.concatenate([z1, z2], 0)

    # gain path
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
    mu_g = dual._GAIN_SCALE * np.asarray(mu, float)
    pred = blend * mu_cv + (1.0 - blend) * mu_g

    zc = zpre - zpre.mean(0)
    m2 = np.maximum((zc**2).mean(0), 1e-12)
    m3 = (zc**3).mean(0)
    m4 = (zc**4).mean(0)
    s = np.sqrt(m2)
    skew = m3 / (m2 * s)
    kurt = m4 / (m2 * m2) - 3.0
    alpha_s = zpre.mean(0) / s
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


def _fit(tr, Feats, Sigmas, Ys, bases, lam):
    X = np.concatenate([Feats[i].T for i in tr], 0)
    y = np.concatenate([(Ys[i] - bases[i]) / (Sigmas[i] + 1e-8) for i in tr], 0)
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    mu[0] = 0.0
    sd[0] = 1.0
    Xs = (X - mu) / sd
    F = Xs.shape[1]
    beta = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(F), Xs.T @ y)
    return beta, mu, sd


def _apply(beta, mu, sd, base_i, feats_i, sig_i, scale=1.0):
    Xi = (feats_i.T - mu) / sd
    return base_i + scale * sig_i * (Xi @ beta)


def main():
    t0 = time.time()
    if CACHE.is_file():
        d = np.load(CACHE)
        if int(d["n_train"]) == N_TRAIN and int(d["n_gt"]) == N_GT:
            bases, Feats, Sigmas, Ys = d["bases"], d["Feats"], d["Sigmas"], d["Ys"]
            print("loaded cache", flush=True)
        else:
            d = None
    else:
        d = None
    if d is None:
        print(f"building dual cache N={N_TRAIN} GT={N_GT}", flush=True)
        bases, Feats, Sigmas, Ys = [], [], [], []
        for i in range(N_TRAIN):
            rng = fnp.random.default_rng(120_000 + i)
            mlp = sample_mlp(width=WIDTH, depth=DEPTH, rng=rng, seed=130_000 + i)
            y = _gt_final(mlp, N_GT, seed=140_000 + i)
            with flops.BudgetContext(flop_budget=B, quiet=True):
                pred, feats, sig = _extract_dual(mlp)
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
        print(f"cache base_mse={float(np.mean((bases-Ys)**2)):.4e}", flush=True)

    folds = np.array_split(np.arange(N_TRAIN), 5)
    best = None
    for lam in [3e-3, 1e-2, 3e-2, 1e-1]:
        for scale in [0.7, 0.85, 1.0]:
            oos_b, oos_c = [], []
            for fi, te in enumerate(folds):
                tr = np.concatenate([folds[j] for j in range(5) if j != fi])
                beta, mu, sd = _fit(tr, Feats, Sigmas, Ys, bases, lam)
                for i in te:
                    oos_b.append(float(np.mean((bases[i] - Ys[i]) ** 2)))
                    oos_c.append(
                        float(
                            np.mean(
                                (
                                    _apply(beta, mu, sd, bases[i], Feats[i], Sigmas[i], scale)
                                    - Ys[i]
                                )
                                ** 2
                            )
                        )
                    )
            b, c = float(np.mean(oos_b)), float(np.mean(oos_c))
            ratio = b / c
            print(
                f"OOS lam={lam:.0e} α={scale:.2f} base={b:.4e} corr={c:.4e} ratio={ratio:.4f}",
                flush=True,
            )
            if best is None or c < best[0]:
                best = (c, lam, scale, ratio)

    lam, scale = best[1], best[2]
    print(f"BEST lam={lam} scale={scale} oos_ratio={best[3]:.4f}", flush=True)
    beta, mu, sd = _fit(np.arange(N_TRAIN), Feats, Sigmas, Ys, bases, lam)

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
            pred, feats, sig = _extract_dual(mlp)
        corr = _apply(beta, mu, sd, pred, feats, sig, scale)
        mb.append(float(np.mean((pred - y) ** 2)))
        mc.append(float(np.mean((corr - y) ** 2)))
        if (i + 1) % 25 == 0:
            print(f"  mini {i+1}", flush=True)
    b, c = float(np.mean(mb)), float(np.mean(mc))
    print(
        f"MINI dual-base={b:.4e} corr={c:.4e} ratio={b/c:.4f} adj≈{c*0.1001:.4e} "
        f"(dual+oldcorr Mini was 4.045e-7)",
        flush=True,
    )
    np.savez(
        OUT,
        beta=beta,
        feat_mu=mu,
        feat_sd=sd,
        lam=np.asarray(lam),
        scale=np.asarray(scale),
        oos_ratio=np.asarray(best[3]),
        mini_corr=np.asarray(c),
        mini_base=np.asarray(b),
    )
    print(f"wrote {OUT} t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

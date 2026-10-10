#!/usr/bin/env python3
"""Hadamard + last-layer statistic ridge corrector (discourse #18085 style).

Base: pure-0.2.1s MC mean.
Features per neuron (from last preacts z): α, α²..α⁴, gauss_gap/σ,
  skew γ1, excess kurt γ2, and optional μ_mc/σ.
Target: (Y − μ_mc) / σ_z.
Gate: Full 5-fold OOS R²>0.02 and Mini ratio>1.05 before shipping.
"""
from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
SHARD_DIR = ROOT / "experiments" / "results" / "full_cache"
FEAT = ROOT / "experiments" / "results" / "full_cache" / "l31_statcorr_feat_v1.npz"
OUT = ROOT / "experiments" / "l31_statcorr_v1.npz"
MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
B = int(2.72e11)
W = 256
NF = 50
F = 9  # feature dim
SQRT2 = math.sqrt(2.0)
ISP = 1.0 / math.sqrt(2.0 * math.pi)


def _load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def _estimator_seed(raw: int) -> int:
    ss = fnp.random.SeedSequence(int(raw)).spawn(3)
    return int(ss[2].generate_state(1)[0]) & 0x7FFFFFFF


def _phi(a: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * a * a) * ISP


def _Phi(a: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(a / SQRT2))


def _gauss_relu(mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    sig = np.maximum(sig, 1e-12)
    a = mu / sig
    return mu * _Phi(a) + sig * _phi(a)


def _extract(hada, weights, seed: int):
    e_r = hada._chi_mean_radius(W)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
        for w in weights[:-1]:
            x = fnp.maximum(x @ w, 0.0)
        z = np.asarray(x @ weights[-1], np.float64)
        mc = np.maximum(z, 0.0).mean(0)
    mu = z.mean(0)
    zc = z - mu
    m2 = np.mean(zc * zc, 0)
    m3 = np.mean(zc**3, 0)
    m4 = np.mean(zc**4, 0)
    sig = np.sqrt(np.maximum(m2, 1e-18))
    a = mu / sig
    g = _gauss_relu(mu, sig)
    g1 = m3 / (sig**3 + 1e-18)  # skewness
    g2 = m4 / (sig**4 + 1e-18) - 3.0  # excess kurtosis
    feats = np.stack(
        [
            np.ones(W),
            a,
            a * a,
            a**3,
            a**4,
            (g - mc) / (sig + 1e-8),
            g1,
            g2,
            mc / (sig + 1e-8),
        ],
        0,
    )  # (F, W)
    return mc, sig, feats


def main() -> None:
    hada = _load_hada()
    shards = sorted(SHARD_DIR.glob("full_shard_*.npz"))
    Ws, Ys, seeds = [], [], []
    for p in shards:
        d = np.load(p)
        Ws.append(d["weights"])
        Ys.append(d["all_layer_means"][:, -1])
        seeds.append(d["mlp_seed"])
        print("loaded", p.name, flush=True)
    Wa = np.concatenate(Ws, 0)
    Ya = np.concatenate(Ys, 0)
    Sa = np.concatenate(seeds, 0)
    n = Wa.shape[0]
    print(f"Full n={n}", flush=True)

    if FEAT.is_file():
        c = np.load(FEAT)
        if int(c["n"]) == n and int(c["F"]) == F:
            Mc, Sig, Fe, Y = c["Mc"], c["Sig"], c["Fe"], c["Y"]
            print("feat cache hit", FEAT, flush=True)
        else:
            c = None
    else:
        c = None

    if c is None:
        Mc = np.zeros((n, W))
        Sig = np.zeros((n, W))
        Fe = np.zeros((n, F, W))
        Y = np.zeros((n, W))
        t0 = time.time()
        for i in range(n):
            ws = [fnp.asarray(Wa[i, li], dtype=fnp.float32) for li in range(32)]
            seed = _estimator_seed(int(Sa[i]))
            mc, sig, feats = _extract(hada, ws, seed)
            Mc[i], Sig[i], Fe[i], Y[i] = mc, sig, feats, Ya[i]
            if (i + 1) % 25 == 0:
                print(
                    f"[{i+1}/{n}] mse={np.mean((mc-Ya[i])**2):.3e} t={time.time()-t0:.0f}s",
                    flush=True,
                )
        np.savez(FEAT, Mc=Mc, Sig=Sig, Fe=Fe, Y=Y, n=n, F=F)
        print("wrote", FEAT, flush=True)

    T = Y - Mc
    base = float(np.mean(T**2))
    print(f"base residual MSE={base:.6e}", flush=True)

    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(n), 5)
    best = None
    for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0]:
        oos = np.zeros_like(T)
        for fi, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            # standardize features on train
            mu = Fe[tr].mean((0, 2))
            sd = Fe[tr].std((0, 2)) + 1e-8
            sd[0] = 1.0
            Xtr = ((Fe[tr] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)).transpose(
                0, 2, 1
            ).reshape(-1, F)
            ytr = (T[tr] / (Sig[tr] + 1e-8)).reshape(-1)
            # subsample for speed/stability
            idx = rng.choice(len(ytr), size=min(50000, len(ytr)), replace=False)
            beta = np.linalg.solve(
                Xtr[idx].T @ Xtr[idx] + lam * np.eye(F), Xtr[idx].T @ ytr[idx]
            )
            Xte = ((Fe[te] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)).transpose(
                0, 2, 1
            ).reshape(-1, F)
            oos[te] = (Xte @ beta).reshape(len(te), W) * (Sig[te] + 1e-8)
        for sh in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]:
            m = float(np.mean((T - sh * oos) ** 2))
            r2 = 1.0 - m / base
            print(f"lam={lam:.0e} sh={sh:.2f} R²={r2:.4f} mse={m:.4e}", flush=True)
            if best is None or m < best[0]:
                best = (m, lam, sh, oos.copy(), None)

    r2_best = 1.0 - best[0] / base
    print(f"BEST R²={r2_best:.4f} lam={best[1]} sh={best[2]}", flush=True)

    # refit on all
    lam, sh = best[1], best[2]
    mu = Fe.mean((0, 2))
    sd = Fe.std((0, 2)) + 1e-8
    sd[0] = 1.0
    X = ((Fe - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)).transpose(0, 2, 1).reshape(
        -1, F
    )
    y = (T / (Sig + 1e-8)).reshape(-1)
    idx = rng.choice(len(y), size=min(80000, len(y)), replace=False)
    beta = np.linalg.solve(X[idx].T @ X[idx] + lam * np.eye(F), X[idx].T @ y[idx])
    np.savez(OUT, beta=beta, feat_mu=mu, feat_sd=sd, lam=lam, shrink=sh, oos_r2=r2_best)
    print("wrote", OUT, flush=True)

    if r2_best < 0.02:
        print("R² too low; skip Mini", flush=True)
        return

    # Mini transfer
    ds = Dataset.load_from_disk(str(MINI))
    eb, ec = [], []
    for i in range(len(ds)):
        ex = ds[i]
        ws = [fnp.asarray(ex["weights"][li], dtype=fnp.float32) for li in range(32)]
        Ym = np.asarray(ex["final_means"], np.float64)
        seed = _estimator_seed(int(ex["mlp_seed"]))
        mc, sig, feats = _extract(hada, ws, seed)
        Xs = ((feats - mu.reshape(-1, 1)) / sd.reshape(-1, 1)).T  # (W,F)
        delta = sh * (Xs @ beta) * (sig + 1e-8)
        corr = mc + delta
        eb.append(float(np.mean((mc - Ym) ** 2)))
        ec.append(float(np.mean((corr - Ym) ** 2)))
        if i % 20 == 0:
            print("mini", i, eb[-1], ec[-1], flush=True)
    b, c = float(np.mean(eb)), float(np.mean(ec))
    print(
        f"Mini base={b:.4e} corr={c:.4e} ratio={b / max(c, 1e-30):.3f} "
        f"adj≈{b * 0.408:.4e}/{c * 0.408:.4e}",
        flush=True,
    )


if __name__ == "__main__":
    main()

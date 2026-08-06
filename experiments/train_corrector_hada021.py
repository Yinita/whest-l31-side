"""Train residual corrector on pure-0.2.1s Hadamard base (sample features only)."""
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
    "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
)
hada = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(hada)

B = int(2.72e11)
WIDTH, DEPTH = 256, 32
N_GT = 120_000
N_TRAIN = 250
CACHE = ROOT / "experiments" / "corrector_cache_hada021.npz"
OUT = ROOT / "experiments" / "corrector_hada021.npz"
INV = 1.0 / math.sqrt(2.0 * math.pi)


def _gt_final(mlp: MLP, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, mlp.width)).astype(np.float32)
    for w in mlp.weights:
        x = np.maximum(x @ np.asarray(w, np.float32), 0.0)
    return x.mean(axis=0).astype(np.float64)


def _extract(mlp: MLP):
    """Hadamard forward; return pred + sample features (F, width)."""
    width = mlp.width
    depth = mlp.depth
    weights = list(mlp.weights)
    n2 = float(width * width)
    flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
    n_forward = int(max(2 * width, min(target / flops_per, hada._MAX_SAMPLES)))
    n_frames = max(1, n_forward // (2 * width))
    e_r = hada._chi_mean_radius(width)
    x = hada._hadamard_directions(
        n_frames, width, seed=int(mlp.seed) & 0x7FFFFFFF, e_r=e_r
    )
    # keep penultimate for zpre
    h = x
    for w in weights[:-1]:
        h = fnp.maximum(h @ w, 0.0)
    zpre = h @ weights[-1]
    last = fnp.maximum(zpre, 0.0)
    pred = np.asarray(fnp.mean(last, axis=0), float)
    z = np.asarray(zpre, float)
    a = np.asarray(last, float)

    zc = z - z.mean(0)
    m2 = np.maximum((zc * zc).mean(0), 1e-12)
    m3 = (zc**3).mean(0)
    m4 = (zc**4).mean(0)
    s = np.sqrt(m2)
    skew = m3 / (m2 * s)
    kurt = m4 / (m2 * m2) - 3.0
    alpha_s = z.mean(0) / s
    phi_s = np.exp(-0.5 * alpha_s * alpha_s) * INV
    st = -(skew / 6.0) * phi_s
    kt = (kurt / 24.0) * (alpha_s**2 - 1.0) * phi_s
    act_rate = (a > 0).mean(0)
    # L1 column norms (cheap exact stats)
    w0 = np.asarray(weights[0], float)
    col_norm = np.sqrt(np.maximum((w0 * w0).sum(0), 1e-30))
    e_a1 = col_norm * INV

    feats = np.stack(
        [
            np.ones(width),
            pred / (s + 1e-8),
            s / (s.mean() + 1e-8),
            alpha_s,
            phi_s,
            st,
            kt,
            act_rate,
            e_a1 / (e_a1.mean() + 1e-8),
            skew,
            kurt,
            (pred - pred.mean()) / (s + 1e-8),
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
    start_i = 0
    bases_l, Feats_l, Sigmas_l, Ys_l = [], [], [], []
    if CACHE.is_file():
        d = np.load(CACHE)
        if int(d["n_gt"]) == N_GT and "bases" in d.files:
            bases_l = [np.asarray(x) for x in d["bases"]]
            Feats_l = [np.asarray(x) for x in d["Feats"]]
            Sigmas_l = [np.asarray(x) for x in d["Sigmas"]]
            Ys_l = [np.asarray(x) for x in d["Ys"]]
            start_i = len(bases_l)
            if start_i >= N_TRAIN:
                bases = np.stack(bases_l[:N_TRAIN])
                Feats = np.stack(Feats_l[:N_TRAIN])
                Sigmas = np.stack(Sigmas_l[:N_TRAIN])
                Ys = np.stack(Ys_l[:N_TRAIN])
                print(f"loaded full cache n={N_TRAIN}", flush=True)
                start_i = N_TRAIN
            else:
                print(f"resume cache from {start_i}/{N_TRAIN}", flush=True)
        else:
            print("cache mismatch; rebuilding", flush=True)

    if start_i < N_TRAIN:
        print(f"building Hadamard cache N={N_TRAIN} GT={N_GT} from={start_i}", flush=True)
        for i in range(start_i, N_TRAIN):
            rng = fnp.random.default_rng(220_000 + i)
            mlp = sample_mlp(width=WIDTH, depth=DEPTH, rng=rng, seed=230_000 + i)
            y = _gt_final(mlp, N_GT, seed=240_000 + i)
            with flops.BudgetContext(flop_budget=B, quiet=True):
                pred, feats, sig = _extract(mlp)
            bases_l.append(pred)
            Feats_l.append(feats)
            Sigmas_l.append(sig)
            Ys_l.append(y)
            if (i + 1) % 25 == 0 or (i + 1) == N_TRAIN:
                np.savez(
                    CACHE,
                    bases=np.stack(bases_l),
                    Feats=np.stack(Feats_l),
                    Sigmas=np.stack(Sigmas_l),
                    Ys=np.stack(Ys_l),
                    n_train=np.asarray(len(bases_l)),
                    n_gt=np.asarray(N_GT),
                )
                print(f"  {i+1}/{N_TRAIN} t={time.time()-t0:.0f}s (ckpt)", flush=True)
        bases = np.stack(bases_l[:N_TRAIN])
        Feats = np.stack(Feats_l[:N_TRAIN])
        Sigmas = np.stack(Sigmas_l[:N_TRAIN])
        Ys = np.stack(Ys_l[:N_TRAIN])
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
    else:
        bases = np.stack(bases_l[:N_TRAIN])
        Feats = np.stack(Feats_l[:N_TRAIN])
        Sigmas = np.stack(Sigmas_l[:N_TRAIN])
        Ys = np.stack(Ys_l[:N_TRAIN])

    folds = np.array_split(np.arange(N_TRAIN), 5)
    best = None
    for lam in [1e-3, 3e-3, 1e-2, 3e-2]:
        for scale in [0.5, 0.7, 0.85, 1.0]:
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
            print(
                f"OOS lam={lam:.0e} α={scale:.2f} base={b:.4e} corr={c:.4e} ratio={b/c:.4f}",
                flush=True,
            )
            if best is None or c < best[0]:
                best = (c, lam, scale, b / c)

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
            pred, feats, sig = _extract(mlp)
        corr = np.maximum(_apply(beta, mu, sd, pred, feats, sig, scale), 0.0)
        mb.append(float(np.mean((pred - y) ** 2)))
        mc.append(float(np.mean((corr - y) ** 2)))
        if (i + 1) % 25 == 0:
            print(f"  mini {i+1}", flush=True)
    b, c = float(np.mean(mb)), float(np.mean(mc))
    # approx adj with util 0.40
    print(
        f"MINI hada-base={b:.4e} corr={c:.4e} ratio={b/c:.4f} "
        f"adj≈{c*0.40:.4e} (pure-0.2.1s Mini adj ~3.009e-7)",
        flush=True,
    )
    np.savez(
        OUT,
        beta=beta * scale,  # fold scale into beta
        feat_mu=mu,
        feat_sd=sd,
        lam=np.asarray(lam),
        scale=np.asarray(1.0),
        oos_ratio=np.asarray(best[3]),
        mini_corr=np.asarray(c),
        mini_base=np.asarray(b),
    )
    print(f"wrote {OUT} t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

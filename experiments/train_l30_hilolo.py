#!/usr/bin/env python3
"""Synth-train L30 residual: μ_hi − μ_lo, then recenter path (DAgger-lite).

Offline: Hadamard @ util·HI_MULT as pseudo-GT for layer 30.
Predict δ = μ_hi_30 − μ_lo_30 from lo-stream + weight features.
Eval: OOS synth rollout + Mini transfer (true all_layer_means).
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
from whestbench import sample_mlp

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments" / "cache_l30_hilolo.npz"
OUT = ROOT / "experiments" / "l30_hilolo_theta.npz"
MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
B = int(2.72e11)
WIDTH, DEPTH = 256, 32
N_TRAIN = 120
HI_MULT = 4.0
RECENTER_LI = 30
F = 16


def _load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def _n_frames(hada, util_scale: float) -> int:
    n2 = float(WIDTH * WIDTH)
    flops_per = DEPTH * hada._MATMUL_PER_LAYER * n2 + DEPTH * WIDTH * 20.0
    target = max(float(B) * hada._UTIL * hada._UTIL_CALIB * util_scale, flops_per)
    n_forward = int(max(2 * WIDTH, min(target / flops_per, hada._MAX_SAMPLES)))
    return max(1, n_forward // (2 * WIDTH))


def _forward_to(hada, weights, seed: int, n_frames: int, stop_li: int):
    e_r = hada._chi_mean_radius(WIDTH)
    with flops.BudgetContext(flop_budget=int(B * max(HI_MULT, 2) * 2), quiet=True):
        x = hada._hadamard_directions(n_frames, WIDTH, seed=seed, e_r=e_r)
        for li in range(stop_li + 1):
            x = fnp.maximum(x @ weights[li], 0.0)
        mu = np.asarray(fnp.mean(x, axis=0), np.float64)
        # sample stats at stop
        a = np.asarray(x, np.float64)
    s = np.sqrt(np.maximum(((a - mu) ** 2).mean(0), 1e-12))
    act = (a > 0).mean(0)
    return mu, s, act, x


def _finish(hada, x, weights, start_li: int, delta: np.ndarray | None):
    with flops.BudgetContext(flop_budget=int(B * 2), quiet=True):
        if delta is not None:
            x = x + fnp.asarray(delta, dtype=fnp.float32)
            x = fnp.maximum(x, 0.0)
        for li in range(start_li + 1, len(weights)):
            x = fnp.maximum(x @ weights[li], 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def _weight_feats(weights) -> np.ndarray:
    """(6, width) from W[30] and W[31]."""
    W30 = np.asarray(weights[RECENTER_LI], np.float64)
    W31 = np.asarray(weights[RECENTER_LI + 1], np.float64)
    return np.stack(
        [
            np.sqrt(np.maximum((W30 * W30).sum(0), 1e-30)),  # col norms → L30
            np.sqrt(np.maximum((W30 * W30).sum(1), 1e-30)),  # row norms
            W30.mean(0),
            np.sqrt(np.maximum((W31 * W31).sum(0), 1e-30)),  # col norms L31
            np.sqrt(np.maximum((W31 * W31).sum(1), 1e-30)),
            W31.mean(0),
        ],
        0,
    )


def _features(mu_lo, s, act, wfeat, mu_hi_proxy=None) -> np.ndarray:
    # mu_hi_proxy unused at train feature time; keep API simple
    gap = mu_lo  # placeholder scale
    return np.stack(
        [
            np.ones(WIDTH),
            mu_lo / (s + 1e-8),
            s / (s.mean() + 1e-8),
            act,
            np.log(s + 1e-8),
            (mu_lo - mu_lo.mean()) / (s + 1e-8),
            wfeat[0] / (wfeat[0].mean() + 1e-8),
            wfeat[1] / (wfeat[1].mean() + 1e-8),
            wfeat[2],
            wfeat[3] / (wfeat[3].mean() + 1e-8),
            wfeat[4] / (wfeat[4].mean() + 1e-8),
            wfeat[5],
            act * mu_lo / (s + 1e-8),
            wfeat[0] * act,
            s,
            mu_lo,
        ],
        0,
    ).astype(np.float64)


def _fit(Feats, Targets, Sig, tr, lam: float):
    mu = Feats[tr].mean((0, 2))
    sd = Feats[tr].std((0, 2)) + 1e-8
    sd[0] = 1.0
    X = (Feats[tr] - mu.reshape(1, -1, 1)) / sd.reshape(1, -1, 1)
    y = (Targets[tr] / (Sig[tr] + 1e-8)).reshape(-1)
    Xt = X.transpose(1, 0, 2).reshape(F, -1)
    beta = np.linalg.solve(
        Xt @ Xt.T / Xt.shape[1] + lam * np.eye(F), Xt @ y / Xt.shape[1]
    )
    return beta, mu, sd


def _apply(beta, mu, sd, feats, sig, shrink: float):
    X = (feats - mu.reshape(-1, 1)) / sd.reshape(-1, 1)
    return shrink * (beta @ X) * (sig + 1e-8)


def main() -> None:
    hada = _load_hada()
    t0 = time.time()
    n_lo = _n_frames(hada, 1.0)
    n_hi = _n_frames(hada, HI_MULT)
    print(f"frames lo={n_lo} hi={n_hi} HI_MULT={HI_MULT}", flush=True)

    start = 0
    stores = {
        k: []
        for k in [
            "mu_lo30",
            "mu_hi30",
            "sig_lo",
            "Feats",
            "mu_lo31",
            "mu_hi31",
        ]
    }
    if CACHE.is_file():
        d = np.load(CACHE)
        if float(d["hi_mult"]) == HI_MULT and int(d["n_lo"]) == n_lo:
            for k in stores:
                stores[k] = [np.asarray(x) for x in d[k]]
            start = len(stores["mu_lo30"])
            print(f"resume cache {start}/{N_TRAIN}", flush=True)
        else:
            print("cache mismatch; rebuild", flush=True)

    while start < N_TRAIN:
        i = start
        rng = fnp.random.default_rng(310_000 + i)
        mlp = sample_mlp(width=WIDTH, depth=DEPTH, rng=rng, seed=320_000 + i)
        weights = list(mlp.weights)
        seed = int(mlp.seed) & 0x7FFFFFFF
        mu_lo, s_lo, act_lo, x_lo = _forward_to(
            hada, weights, seed, n_lo, RECENTER_LI
        )
        mu_hi, _, _, x_hi = _forward_to(
            hada, weights, seed ^ 0x11111111, n_hi, RECENTER_LI
        )
        mu_lo31 = _finish(hada, x_lo, weights, RECENTER_LI, None)
        mu_hi31 = _finish(hada, x_hi, weights, RECENTER_LI, None)
        wfeat = _weight_feats(weights)
        feats = _features(mu_lo, s_lo, act_lo, wfeat)
        stores["mu_lo30"].append(mu_lo)
        stores["mu_hi30"].append(mu_hi)
        stores["sig_lo"].append(s_lo)
        stores["Feats"].append(feats)
        stores["mu_lo31"].append(mu_lo31)
        stores["mu_hi31"].append(mu_hi31)
        start += 1
        if start % 10 == 0 or start == N_TRAIN:
            np.savez(
                CACHE,
                **{k: np.stack(v) for k, v in stores.items()},
                hi_mult=np.asarray(HI_MULT),
                n_lo=np.asarray(n_lo),
                n_hi=np.asarray(n_hi),
            )
            gap = np.mean((mu_hi - mu_lo) ** 2)
            print(
                f"[{start}/{N_TRAIN}] gap30={gap:.3e} "
                f"lo31_vs_hi={np.mean((mu_lo31-mu_hi31)**2):.3e} "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )

    mu_lo30 = np.stack(stores["mu_lo30"][:N_TRAIN])
    mu_hi30 = np.stack(stores["mu_hi30"][:N_TRAIN])
    Sig = np.stack(stores["sig_lo"][:N_TRAIN])
    Feats = np.stack(stores["Feats"][:N_TRAIN])
    mu_lo31 = np.stack(stores["mu_lo31"][:N_TRAIN])
    mu_hi31 = np.stack(stores["mu_hi31"][:N_TRAIN])
    Targets = mu_hi30 - mu_lo30

    base_gap = float(np.mean(Targets**2))
    base_l31 = float(np.mean((mu_lo31 - mu_hi31) ** 2))
    print(f"synth base |δ30|^2={base_gap:.4e}  |lo31-hi31|^2={base_l31:.4e}")

    # Need x_lo for rollout — re-forward in OOS (accept cost)
    folds = np.array_split(np.arange(N_TRAIN), 5)
    best = None
    for lam in [1e-1, 3e-2, 1e-2, 1e-3]:
        for shrink in [1.0, 0.5, 0.25, 0.1]:
            oos_gap = []
            oos_l31 = []
            for fi, te in enumerate(folds):
                tr = np.concatenate([folds[j] for j in range(5) if j != fi])
                beta, mu, sd = _fit(Feats, Targets, Sig, tr, lam)
                for i in te:
                    d = _apply(beta, mu, sd, Feats[i], Sig[i], shrink)
                    oos_gap.append(float(np.mean((Targets[i] - d) ** 2)))
                    # rollout: recreate lo path
                    rng = fnp.random.default_rng(310_000 + int(i))
                    mlp = sample_mlp(
                        width=WIDTH, depth=DEPTH, rng=rng, seed=320_000 + int(i)
                    )
                    weights = list(mlp.weights)
                    seed = int(mlp.seed) & 0x7FFFFFFF
                    _m, _s, _a, x_lo = _forward_to(
                        hada, weights, seed, n_lo, RECENTER_LI
                    )
                    pred31 = _finish(hada, x_lo, weights, RECENTER_LI, d)
                    oos_l31.append(float(np.mean((pred31 - mu_hi31[i]) ** 2)))
            g = float(np.mean(oos_gap))
            m = float(np.mean(oos_l31))
            print(
                f"lam={lam:.0e} sh={shrink:.2f}  "
                f"δ OOS={g:.4e} (r={base_gap/max(g,1e-30):.3f})  "
                f"L31 OOS={m:.4e} (r={base_l31/max(m,1e-30):.3f})",
                flush=True,
            )
            if best is None or m < best[0]:
                best = (m, lam, shrink)

    lam, shrink = best[1], best[2]
    beta, mu, sd = _fit(Feats, Targets, Sig, np.arange(N_TRAIN), lam)
    np.savez(
        OUT,
        beta=beta,
        feat_mu=mu,
        feat_sd=sd,
        lam=lam,
        shrink=shrink,
        recenter_li=RECENTER_LI,
        hi_mult=HI_MULT,
        synth_oos_l31=best[0],
        synth_base_l31=base_l31,
    )
    print(f"wrote {OUT} best L31 OOS={best[0]:.4e} r={base_l31/best[0]:.3f}")

    # Mini transfer: predict δ, recenter, vs true Y
    print("Mini transfer…", flush=True)
    ds = Dataset.load_from_disk(str(MINI))
    errs_base, errs_corr = [], []
    for i in range(len(ds)):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        mu_lo, s_lo, act_lo, x_lo = _forward_to(
            hada, weights, seed, n_lo, RECENTER_LI
        )
        wfeat = _weight_feats(weights)
        feats = _features(mu_lo, s_lo, act_lo, wfeat)
        d = _apply(beta, mu, sd, feats, s_lo, shrink)
        base31 = _finish(hada, x_lo, weights, RECENTER_LI, None)
        # need fresh x for corr path
        _, _, _, x2 = _forward_to(hada, weights, seed, n_lo, RECENTER_LI)
        corr31 = _finish(hada, x2, weights, RECENTER_LI, d)
        errs_base.append(np.mean((base31 - Y[-1]) ** 2))
        errs_corr.append(np.mean((corr31 - Y[-1]) ** 2))
        if i % 20 == 0:
            print(
                f"  [{i}] base={errs_base[-1]:.3e} corr={errs_corr[-1]:.3e}",
                flush=True,
            )
    b, c = float(np.mean(errs_base)), float(np.mean(errs_corr))
    print(
        f"Mini raw base={b:.4e} corr={c:.4e} ratio={b/c:.3f} "
        f"proxy_adj_base={b*0.4:.4e} proxy_adj_corr={c*0.4:.4e}"
    )


if __name__ == "__main__":
    main()

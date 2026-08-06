#!/usr/bin/env python3
"""Quick Mini h2h: pure-0.2.1s mean vs last-layer diag-Gaussian (pure-0.3.2s).

Also tests blend: (1-λ)*sample_mean + λ*gauss_last @ λ∈{0.05,0.1,0.2}.
Protocol 3.0; one forward shared where possible.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset
from whestbench.domain import MLP

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "probes" / "results"
OUT.mkdir(parents=True, exist_ok=True)
MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
B = int(2.72e11)
W = 256
NF = 50
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def phi_cdf(alpha: np.ndarray) -> np.ndarray:
    s = math.sqrt(2.0)
    return np.asarray(
        [0.5 * (1.0 + math.erf(float(a) / s)) for a in np.asarray(alpha).ravel()],
        dtype=np.float64,
    ).reshape(np.asarray(alpha).shape)


def phi_pdf(alpha: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * np.asarray(alpha, dtype=np.float64) ** 2)


def e_relu_gauss(mu_pre: np.ndarray, var_pre: np.ndarray) -> np.ndarray:
    sig = np.sqrt(np.maximum(var_pre, 1e-12))
    a = mu_pre / sig
    return sig * phi_pdf(a) + mu_pre * phi_cdf(a)


def main():
    hada = load_mod("hada", "experiments/pure021s_hada_u40.py")
    ds = Dataset.load_from_disk(str(MINI))
    e_r = hada._chi_mean_radius(W)
    kinds = ["mean", "gauss_last", "blend05", "blend10", "blend20", "more_nf_gauss"]
    errs = {k: [] for k in kinds}
    print(f"NF={NF}", flush=True)

    for i in range(len(ds)):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        weights = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF

        # baseline full forward
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
            for w in weights:
                x = fnp.maximum(x @ w, 0.0)
            mean_pred = np.asarray(fnp.mean(x, axis=0), np.float64)
        errs["mean"].append(float(np.mean((mean_pred - Y) ** 2)))

        # penult + gaussian last (same NF)
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
            for w in weights[:-1]:
                x = fnp.maximum(x @ w, 0.0)
            X = np.asarray(x, np.float64)
            mu = X.mean(0)
            var = np.maximum(X.var(0), 1e-12)  # population? use ddof=0
            wL = np.asarray(weights[-1], np.float64)
            mu_pre = wL.T @ mu
            var_pre = (wL * wL).T @ var
            g = e_relu_gauss(mu_pre, var_pre)
        errs["gauss_last"].append(float(np.mean((g - Y) ** 2)))
        for lam, k in [(0.05, "blend05"), (0.1, "blend10"), (0.2, "blend20")]:
            # blend needs sample mean of last layer — use mean_pred from same seed full fwd
            b = (1 - lam) * mean_pred + lam * g
            errs[k].append(float(np.mean((b - Y) ** 2)))

        # more frames with gauss last: NF*32/31 ≈ +3% frames
        nf2 = int(NF * 32 / 31)
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(nf2, W, seed=seed, e_r=e_r)
            for w in weights[:-1]:
                x = fnp.maximum(x @ w, 0.0)
            X = np.asarray(x, np.float64)
            mu = X.mean(0)
            var = np.maximum(X.var(0), 1e-12)
            wL = np.asarray(weights[-1], np.float64)
            g2 = e_relu_gauss(wL.T @ mu, (wL * wL).T @ var)
        errs["more_nf_gauss"].append(float(np.mean((g2 - Y) ** 2)))

        if i % 10 == 0:
            means = {k: float(np.mean(errs[k])) for k in kinds}
            best = min(means.items(), key=lambda kv: kv[1])
            print(
                f"[{i}] mean={means['mean']:.4e} best={best[0]}={best[1]:.4e} "
                f"r={means['mean']/best[1]:.4f}",
                flush=True,
            )
            (OUT / "hada_last_gauss_partial.json").write_text(
                json.dumps({"n": i + 1, **means})
            )

    base = float(np.mean(errs["mean"]))
    ranking = []
    print(f"\n=== Mini n={len(ds)} base={base:.6e} ===", flush=True)
    for m, k in sorted((float(np.mean(errs[k])), k) for k in kinds):
        print(
            f"{k:14s} mse={m:.6e} ratio={base/m:.4f} proxy_adj={m*0.408:.6e}",
            flush=True,
        )
        ranking.append({"kind": k, "mse": m, "ratio": base / m, "proxy_adj": m * 0.408})
    (OUT / "hada_last_gauss_final.json").write_text(
        json.dumps({"base": base, "n": len(ds), "ranking": ranking}, indent=2)
    )


if __name__ == "__main__":
    main()

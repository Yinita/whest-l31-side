#!/usr/bin/env python3
"""James-Stein / shrink post-process on Hadamard L31 mean (near-zero FLOP).

If μ̂ ≈ N(μ, σ²I), JS can dominate the sample mean. Probe several targets.
Protocol 3.0. Durable checkpoints.
"""
from __future__ import annotations

import importlib.util
import json
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
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def gain_analytic(wL: np.ndarray) -> np.ndarray:
    """He-Gaussian last-layer rough: E[relu] ≈ ||col||/√(2π) if input ~N(0,I) — crude."""
    # Use column norms of W31 (weight acts on right: x@W → cols are input weights per out)
    # Actually x@W: out_j = sum_i x_i W_ij, so column j.
    s = np.sqrt(np.maximum((wL * wL).sum(0), 1e-30))
    return s * _INV_SQRT_2PI


def js_shrink(mu: np.ndarray, target: np.ndarray, sigma2: float) -> np.ndarray:
    """Positive-part James-Stein toward target."""
    d = mu.size
    r2 = float(np.sum((mu - target) ** 2))
    if r2 < 1e-30:
        return mu.copy()
    # classic factor (d-2)σ²/||μ-t||²; clamp to [0,1]
    c = max(0.0, 1.0 - (d - 2) * sigma2 / r2)
    return target + c * (mu - target)


def soft_shrink(mu: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - lam) * mu + lam * target


def main():
    hada = load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    e_r = hada._chi_mean_radius(W)
    kinds = [
        "mean",
        "js_zero",
        "js_gmean",
        "js_gain",
        "soft05_gmean",
        "soft02_gain",
        "soft05_gain",
        "scale099",
        "scale098",
    ]
    errs = {k: [] for k in kinds}
    # estimate σ² from dual half-split on first pass? use proxy: mse of mean ≈ σ²/N_eff
    # For JS use per-MLP estimate: half-sample variance
    print(f"NF={NF} kinds={kinds}", flush=True)

    for i in range(len(ds)):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        weights = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF

        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
            for w in weights:
                x = fnp.maximum(x @ w, 0.0)
            X = np.asarray(x, np.float64)
            mu = X.mean(0)
            # unbiased-ish per-coord var of the mean: s²/N
            n = X.shape[0]
            s2_coord = X.var(0) / n
            sigma2 = float(np.mean(s2_coord))

        wL = np.asarray(weights[-1], np.float64)
        g = gain_analytic(wL)
        gmean = np.full(W, float(mu.mean()))

        preds = {
            "mean": mu,
            "js_zero": js_shrink(mu, np.zeros(W), sigma2),
            "js_gmean": js_shrink(mu, gmean, sigma2),
            "js_gain": js_shrink(mu, g, sigma2),
            "soft05_gmean": soft_shrink(mu, gmean, 0.05),
            "soft02_gain": soft_shrink(mu, g, 0.02),
            "soft05_gain": soft_shrink(mu, g, 0.05),
            "scale099": mu * 0.99,
            "scale098": mu * 0.98,
        }
        for k, p in preds.items():
            errs[k].append(float(np.mean((p - Y) ** 2)))

        if i % 10 == 0:
            means = {k: float(np.mean(errs[k])) for k in kinds}
            best = min(means.items(), key=lambda kv: kv[1])
            print(
                f"[{i}] mean={means['mean']:.4e} best={best[0]}={best[1]:.4e} "
                f"r={means['mean']/best[1]:.4f} σ²={sigma2:.3e}",
                flush=True,
            )
            (OUT / "hada_js_shrink_partial.json").write_text(
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
    (OUT / "hada_js_shrink_final.json").write_text(
        json.dumps({"base": base, "n": len(ds), "ranking": ranking}, indent=2)
    )


if __name__ == "__main__":
    main()

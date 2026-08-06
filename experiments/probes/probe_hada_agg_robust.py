#!/usr/bin/env python3
"""L31 aggregation: mean vs trimmed/winsor/median (same Hadamard samples).

Variance-reduction via robust aggregation — may trade bias. Protocol 3.0.
Processes one MLP at a time; writes durable checkpoints (prior probes died ~n=21).
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


def load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def trimmed_mean(x: np.ndarray, frac: float) -> np.ndarray:
    """x: (N, d) — per-coordinate trimmed mean."""
    n = x.shape[0]
    k = int(n * frac)
    if k <= 0:
        return x.mean(0)
    xs = np.sort(x, axis=0)
    return xs[k : n - k].mean(0)


def winsor_mean(x: np.ndarray, frac: float) -> np.ndarray:
    n = x.shape[0]
    k = int(n * frac)
    if k <= 0:
        return x.mean(0)
    xs = np.sort(x, axis=0)
    lo = xs[k]
    hi = xs[n - 1 - k]
    return np.clip(x, lo, hi).mean(0)


def main():
    hada = load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    e_r = hada._chi_mean_radius(W)
    kinds = ["mean", "trim05", "trim10", "win05", "win10", "median"]
    errs = {k: [] for k in kinds}
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
            X = np.asarray(x, np.float64)  # (N, d)

        preds = {
            "mean": X.mean(0),
            "trim05": trimmed_mean(X, 0.05),
            "trim10": trimmed_mean(X, 0.10),
            "win05": winsor_mean(X, 0.05),
            "win10": winsor_mean(X, 0.10),
            "median": np.median(X, axis=0),
        }
        for k, p in preds.items():
            errs[k].append(float(np.mean((p - Y) ** 2)))

        if i % 10 == 0 or i == len(ds) - 1:
            means = {k: float(np.mean(errs[k])) for k in kinds}
            best = min(means.items(), key=lambda kv: kv[1])
            print(
                f"[{i}] mean={means['mean']:.4e} best={best[0]}={best[1]:.4e} "
                f"r={means['mean'] / best[1]:.4f}",
                flush=True,
            )
            (OUT / "hada_agg_robust_partial.json").write_text(
                json.dumps({"n": i + 1, **means})
            )

    base = float(np.mean(errs["mean"]))
    ranking = []
    print(f"\n=== Mini n={len(ds)} base={base:.6e} ===", flush=True)
    for m, k in sorted((float(np.mean(errs[k])), k) for k in kinds):
        print(
            f"{k:8s} mse={m:.6e} ratio={base / m:.4f} proxy_adj={m * 0.408:.6e}",
            flush=True,
        )
        ranking.append({"kind": k, "mse": m, "ratio": base / m, "proxy_adj": m * 0.408})
    (OUT / "hada_agg_robust_final.json").write_text(
        json.dumps({"base": base, "n": len(ds), "ranking": ranking}, indent=2)
    )


if __name__ == "__main__":
    main()

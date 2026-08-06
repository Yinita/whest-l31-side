#!/usr/bin/env python3
"""Soft α recenter toward GT / noisy targets; measure sample μ30 error."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
B = int(2.72e11)


def load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def run(hada, weights, seed, target30, alpha: float):
    width, depth = 256, 32
    n2 = float(width * width)
    flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
    n_forward = int(max(2 * width, min(target / flops_per, hada._MAX_SAMPLES)))
    n_frames = max(1, n_forward // (2 * width))
    e_r = hada._chi_mean_radius(width)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(n_frames, width, seed=seed, e_r=e_r)
        mu30 = None
        for li, w in enumerate(weights):
            x = fnp.maximum(x @ w, 0.0)
            if li == 30 and alpha > 0.0:
                mu = fnp.mean(x, axis=0)
                mu30 = np.asarray(mu, np.float64)
                y = fnp.asarray(target30, dtype=fnp.float32)
                x = x + fnp.float32(alpha) * (y - mu)
        return np.asarray(fnp.mean(x, axis=0), np.float64), mu30


def main():
    hada = load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    ids = list(range(0, 100, 2))
    alphas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    # relative sample error of μ30 vs GT
    rel_errs = []
    # soft toward GT
    soft = {a: [] for a in alphas}
    # soft toward sample+noise at rel_err≈median sample error — skip
    # soft toward independent pilot (half frames) — later
    for i in ids:
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        rms = float(np.sqrt(np.mean(Y[30] ** 2)) + 1e-12)
        for a in alphas:
            pred, mu30 = run(hada, weights, seed, Y[30], a)
            soft[a].append(float(np.mean((pred - Y[-1]) ** 2)))
            if a == 0.0 and mu30 is None:
                # need mu30: run once more with alpha tiny to capture? recompute
                pass
        # dedicated sample error
        _, mu30 = run(hada, weights, seed, Y[30], 1e-9)  # near-zero soft to grab mu
        # better: alpha=0 path doesn't return mu30 — fix by always recording
        # recompute mu via alpha=0 with side channel
        width, depth = 256, 32
        n2 = float(width * width)
        flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * width * 20.0
        target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
        n_forward = int(max(2 * width, min(target / flops_per, hada._MAX_SAMPLES)))
        n_frames = max(1, n_forward // (2 * width))
        e_r = hada._chi_mean_radius(width)
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(n_frames, width, seed=seed, e_r=e_r)
            for li, w in enumerate(weights):
                x = fnp.maximum(x @ w, 0.0)
                if li == 30:
                    mu30 = np.asarray(fnp.mean(x, axis=0), np.float64)
                    break
        rel = float(np.sqrt(np.mean((mu30 - Y[30]) ** 2)) / rms)
        rel_errs.append(rel)
        if i % 10 == 0:
            print(
                f"[{i}] soft1={soft[1.0][-1]:.3e} soft0={soft[0.0][-1]:.3e} "
                f"rel_mu30={rel:.3f}",
                flush=True,
            )

    base = float(np.mean(soft[0.0]))
    print(f"=== soft α → GT L30 (n={len(ids)}) base={base:.4e} ===", flush=True)
    for a in alphas:
        m = float(np.mean(soft[a]))
        print(
            f"alpha={a:.2f} mse={m:.4e} ratio={base / max(m, 1e-30):.3f} "
            f"proxy_adj={m * 0.40:.4e}",
            flush=True,
        )
    rel_errs = np.asarray(rel_errs)
    print(
        f"sample μ30 rel RMSE / rms(Y30): mean={rel_errs.mean():.3f} "
        f"p50={np.median(rel_errs):.3f} p90={np.quantile(rel_errs,0.9):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

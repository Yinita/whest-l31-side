#!/usr/bin/env python3
"""How accurate must L30 mean targets be for GT-style recenter to help?"""
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


def run(hada, weights, seed, target30, do_recenter):
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
            if do_recenter and li == 30:
                mu = fnp.mean(x, axis=0)
                y = fnp.asarray(target30, dtype=fnp.float32)
                x = x + (y - mu)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def main():
    hada = load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    scales = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, None]
    ids = list(range(0, 100, 2))  # 50 MLPs
    errs = {str(s): [] for s in scales}
    for i in ids:
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        rms = float(np.sqrt(np.mean(Y[30] ** 2)) + 1e-12)
        # also compare sample μ30 error scale
        rng = np.random.default_rng(i + 123)
        for s in scales:
            if s is None:
                pred = run(hada, weights, seed, Y[30], False)
            else:
                noise = rng.standard_normal(256) * rms * float(s)
                pred = run(hada, weights, seed, Y[30] + noise, True)
            errs[str(s)].append(float(np.mean((pred - Y[-1]) ** 2)))
        if i % 10 == 0:
            print(
                f"[{i}] none={errs['None'][-1]:.3e} gt={errs['0.0'][-1]:.3e}",
                flush=True,
            )

    base = float(np.mean(errs["None"]))
    print(f"=== L30 target noise sensitivity (n={len(ids)}) ===", flush=True)
    print(f"baseline (no recenter) mse={base:.4e}", flush=True)
    for s in scales:
        m = float(np.mean(errs[str(s)]))
        print(
            f"scale={s!s:6} mse={m:.4e} ratio={base / max(m, 1e-30):.3f} "
            f"proxy_adj={m * 0.40:.4e}",
            flush=True,
        )


if __name__ == "__main__":
    main()

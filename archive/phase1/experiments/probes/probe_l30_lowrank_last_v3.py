#!/usr/bin/env python3
"""P4-lite: Hadamard to L30, low-rank Gaussian resample for last layer.

Keep full-path MC mean; optionally blend with last-layer mean under
a ~ N(μ, U diag(λ) U' + ε I) fit on L30 activations (rank k).
Extra last-layer samples are cheap (1 matmul); early path unchanged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset
from whestbench.domain import MLP

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hada", ROOT / "experiments/pure021s_hada_u40.py")
hada = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hada)

MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/"
    "snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
B = int(2.72e11)
W = 256
D = 32
NF = 50
e_r = hada._chi_mean_radius(W)


def path(weights, seed: int):
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
        for li, w in enumerate(weights):
            x = fnp.maximum(x @ w, 0.0)
            if li == D - 2:
                a30 = np.asarray(x, np.float64)
        mc = np.asarray(fnp.mean(x, axis=0), np.float64)
    return mc, a30


def lowrank_last(a30, w31, k: int, n_extra: int, seed: int, ridge: float = 1e-3):
    rng = np.random.default_rng(seed)
    mu = a30.mean(0)
    xc = a30 - mu
    # economy SVD on (N,W) centered — use cov eig via SVD of xc
    # xc shape (N,W); for N>=W use xc.T@xc
    if a30.shape[0] >= W:
        cov = (xc.T @ xc) / max(a30.shape[0] - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]
        U = evecs[:, :k]
        lam = np.maximum(evals[:k], 0.0)
        # residual variance for isotropic tail
        tail = float(np.mean(np.maximum(evals[k:], 0.0)))
    else:
        # thin SVD
        _, s, Vt = np.linalg.svd(xc / np.sqrt(max(a30.shape[0] - 1, 1)), full_matrices=False)
        U = Vt[:k].T
        lam = s[:k] ** 2
        tail = float(np.mean(s[k:] ** 2)) if s.size > k else ridge

    # sample: mu + U @ (sqrt(lam)*g) + sqrt(tail+ridge)*h
    g = rng.normal(size=(n_extra, k))
    h = rng.normal(size=(n_extra, W))
    z = mu + (g * np.sqrt(lam + 1e-18)) @ U.T + np.sqrt(max(tail, ridge)) * h
    z = np.maximum(z, 0.0)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        y = fnp.maximum(fnp.asarray(z, dtype=fnp.float32) @ w31, 0.0)
        return np.asarray(fnp.mean(y, axis=0), np.float64)


def main() -> None:
    ds = Dataset.load_from_disk(str(MINI))
    # n_extra chosen ~ free budget if we had cut 5 frames: 5*32*512 ≈ 80k
    specs = [
        ("mc", None, None, None),
        ("lr8_80k_b05", 8, 80000, 0.5),
        ("lr16_80k_b05", 16, 80000, 0.5),
        ("lr32_80k_b05", 32, 80000, 0.5),
        ("lr16_80k_b02", 16, 80000, 0.2),
        ("lr16_20k_b05", 16, 20000, 0.5),
    ]
    errs = {s[0]: [] for s in specs}

    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        ws = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF
        mc, a30 = path(ws, seed)
        for name, k, n_ex, blend in specs:
            if k is None:
                pred = mc
            else:
                extra = lowrank_last(a30, ws[-1], k, n_ex, seed ^ 0x5555)
                pred = (1.0 - blend) * mc + blend * extra
            errs[name].append(float(np.mean((pred - Y) ** 2)))
        if i % 20 == 0:
            print(
                i,
                {k: float(np.mean(errs[k])) for k in ("mc", "lr16_80k_b05", "lr16_80k_b02")},
                flush=True,
            )

    base = float(np.mean(errs["mc"]))
    print("mc", base, "ref", 7.3833e-07, flush=True)
    for name, *_ in specs:
        m = float(np.mean(errs[name]))
        print(f"{name:14s} mse={m:.6e} r={base / m:.4f} adj≈{m * 0.408:.6e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

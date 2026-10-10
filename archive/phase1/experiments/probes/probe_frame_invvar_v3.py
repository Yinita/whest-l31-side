#!/usr/bin/env python3
"""Inverse-variance Hadamard frame weighting (no learned θ), seed_protocol 3.0.

For each of NF frames (512 antipodal samples), compute per-neuron mean & var,
then combine frames with w_k ∝ 1/(v_k + ε) (normalized), vs uniform mean.
Also: scalar JS shrink of each frame mean toward grand mean.
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
NF = 50
SPS = 2 * W  # samples per frame (antipodal)
e_r = hada._chi_mean_radius(W)


def frame_stats(weights, seed: int):
    """Return (NF, W) means and (NF, W) vars of per-frame sample means."""
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        last = np.asarray(x, np.float64)  # (NF*SPS, W)
    last = last.reshape(NF, SPS, W)
    means = last.mean(1)
    # var of the mean estimator per frame
    vars_ = last.var(1, ddof=1) / SPS
    return means, vars_


def combine(means, vars_, mode: str):
    if mode == "uniform":
        return means.mean(0)
    if mode == "invvar":
        # per-neuron precision weights across frames
        prec = 1.0 / (vars_ + 1e-18)
        w = prec / prec.sum(0, keepdims=True)
        return (w * means).sum(0)
    if mode == "invvar_shrink":
        # shrink precisions toward uniform (half)
        prec = 1.0 / (vars_ + 1e-18)
        prec = 0.5 * prec + 0.5 * prec.mean(0, keepdims=True)
        w = prec / prec.sum(0, keepdims=True)
        return (w * means).sum(0)
    if mode == "js_frames":
        # James-Stein: shrink frame means toward grand mean using mean var
        mu = means.mean(0)
        # classic: (1 - (d-2)σ²/||θ||²)+ but per frame vector
        out = np.zeros(W)
        for k in range(means.shape[0]):
            diff = means[k] - mu
            nrm2 = float(np.sum(diff * diff)) + 1e-30
            sigma2 = float(np.mean(vars_[k]))
            coef = max(0.0, 1.0 - (W - 2) * sigma2 / nrm2)
            out += mu + coef * diff
        return out / means.shape[0]
    raise ValueError(mode)


def main() -> None:
    ds = Dataset.load_from_disk(str(MINI))
    modes = ["uniform", "invvar", "invvar_shrink", "js_frames"]
    errs = {m: [] for m in modes}
    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        ws = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF
        means, vars_ = frame_stats(ws, seed)
        for m in modes:
            pred = combine(means, vars_, m)
            errs[m].append(float(np.mean((pred - Y) ** 2)))
        if i % 20 == 0:
            print(i, {m: float(np.mean(errs[m])) for m in modes}, flush=True)

    base = float(np.mean(errs["uniform"]))
    print("uniform", base, "ref", 7.3833e-07, flush=True)
    for m in modes:
        mse = float(np.mean(errs[m]))
        print(
            f"{m:14s} mse={mse:.6e} r={base / mse:.4f} adj≈{mse * 0.408:.6e}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

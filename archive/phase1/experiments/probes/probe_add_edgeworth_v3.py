#!/usr/bin/env python3
"""Additive sample-Edgeworth on top of Hadamard MC mean (not a replacement).

#18085 stage 2.5 helps analytic-suffix residuals; we already killed *replacing*
MC with Gauss+Edgeworth. Here: pred = mc + λ δ_skew (+ κ δ_kurt), λ small.
Protocol 3.0 Mini-100.
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
e_r = hada._chi_mean_radius(W)


def path_pre_and_mc(weights, seed: int):
    """Return L31 mc mean and last-layer pre-activations z (N,W)."""
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
        for li, w in enumerate(weights):
            if li == len(weights) - 1:
                z = x @ w  # pre-activation
                x = fnp.maximum(z, 0.0)
            else:
                x = fnp.maximum(x @ w, 0.0)
        mc = np.asarray(fnp.mean(x, axis=0), np.float64)
        z = np.asarray(z, np.float64)
    return mc, z


def edgeworth_delta(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = z.mean(0)
    sig = np.sqrt(np.maximum(z.var(0, ddof=1), 1e-18))
    alpha = mu / sig
    af = fnp.asarray(alpha.astype(np.float32), dtype=fnp.float32)
    phi = np.asarray(flops.stats.norm.pdf(af), np.float64)
    zc = z - mu
    k3 = (zc**3).mean(0)
    k4 = (zc**4).mean(0) - 3.0 * (zc**2).mean(0) ** 2
    # dimensionally natural forms (probe_edgeworth_last_v3)
    d_skew = -(k3 / (6.0 * sig**2)) * phi
    d_kurt = (k4 / 24.0) * ((alpha**2 - 1.0) / (sig**3)) * phi
    return d_skew, d_kurt


def main() -> None:
    ds = Dataset.load_from_disk(str(MINI))
    cfgs = {
        "mc": [],
        "s001": [],
        "s005": [],
        "s01": [],
        "s02": [],
        "s05": [],
        "sk001": [],
        "sk01": [],
        "sk02": [],
    }
    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF
        mc, z = path_pre_and_mc(list(mlp.weights), seed)
        ds_, dk = edgeworth_delta(z)
        preds = {
            "mc": mc,
            "s001": mc + 0.01 * ds_,
            "s005": mc + 0.05 * ds_,
            "s01": mc + 0.1 * ds_,
            "s02": mc + 0.2 * ds_,
            "s05": mc + 0.5 * ds_,
            "sk001": mc + 0.01 * ds_ + 0.01 * dk,
            "sk01": mc + 0.1 * ds_ + 0.1 * dk,
            "sk02": mc + 0.2 * ds_ + 0.2 * dk,
        }
        for k, p in preds.items():
            cfgs[k].append(float(np.mean((p - Y) ** 2)))
        if i % 20 == 0:
            print(i, {k: float(np.mean(cfgs[k])) for k in ("mc", "s01", "sk01")}, flush=True)

    base = float(np.mean(cfgs["mc"]))
    print("mc", base, "ref", 7.3833e-07, flush=True)
    for k, errs in cfgs.items():
        m = float(np.mean(errs))
        print(f"{k:8s} mse={m:.6e} r={base/m:.4f} adj≈{m*0.408:.6e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

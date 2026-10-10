#!/usr/bin/env python3
"""Hadamard path + Gaussian/Edgeworth last-layer (proto 3.0).

Inspired by discourse #18085: measure preact moments from samples,
pred = μΦ(α)+σφ(α) + δ_skew + δ_kurt, vs plain MC mean of ReLU.
"""
from __future__ import annotations

import importlib.util
import math
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
SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def phi(a):
    return np.exp(-0.5 * a * a) * INV_SQRT_2PI


def Phi(a):
    return 0.5 * (1.0 + np.vectorize(math.erf)(a / SQRT2))


def gauss_relu(mu, sig):
    sig = np.maximum(sig, 1e-12)
    a = mu / sig
    return mu * Phi(a) + sig * phi(a)


def edgeworth_deltas(z, form: str):
    """z: (N,W) pre-activations. Returns (skew_delta, kurt_delta) shape (W,)."""
    mu = z.mean(0)
    zc = z - mu
    m2 = np.mean(zc * zc, 0)
    m3 = np.mean(zc**3, 0)
    m4 = np.mean(zc**4, 0)
    sig = np.sqrt(np.maximum(m2, 1e-18))
    k3 = m3  # centered third moment = κ3
    k4 = m4 - 3.0 * m2 * m2  # excess kurtosis * σ^4
    a = mu / sig
    ph = phi(a)
    if form == "a":
        # literal discourse latex (κ3/σ³)×φ  — dimeless; scale by σ
        d_skew = -(k3 / (6.0 * sig**3)) * ph * sig
        d_kurt = (k4 / 24.0) * ((a * a - 1.0) / (sig**3)) * ph
    elif form == "b":
        # dimensionally natural: κ3/(6σ²) φ
        d_skew = -(k3 / (6.0 * sig**2)) * ph
        d_kurt = (k4 / 24.0) * ((a * a - 1.0) / (sig**2)) * ph
    else:
        raise ValueError(form)
    return d_skew, d_kurt


def main() -> None:
    ds = Dataset.load_from_disk(str(MINI))
    keys = [
        "mc",
        "gauss",
        "gauss_s_a",
        "gauss_sk_a",
        "gauss_s_b",
        "gauss_sk_b",
        "blend_mc_gsk_b",
    ]
    errs = {k: [] for k in keys}

    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        ws = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
            for w in ws[:-1]:
                x = fnp.maximum(x @ w, 0.0)
            # preact of last layer
            z = np.asarray(x @ ws[-1], np.float64)
            mc = np.maximum(z, 0.0).mean(0)

        mu = z.mean(0)
        sig = z.std(0, ddof=1)
        g = gauss_relu(mu, sig)
        ds_a, dk_a = edgeworth_deltas(z, "a")
        ds_b, dk_b = edgeworth_deltas(z, "b")

        preds = {
            "mc": mc,
            "gauss": g,
            "gauss_s_a": g + ds_a,
            "gauss_sk_a": g + ds_a + dk_a,
            "gauss_s_b": g + ds_b,
            "gauss_sk_b": g + ds_b + dk_b,
            "blend_mc_gsk_b": 0.7 * mc + 0.3 * (g + ds_b + dk_b),
        }
        for k, p in preds.items():
            errs[k].append(float(np.mean((p - Y) ** 2)))
        if i % 20 == 0:
            print(i, {k: float(np.mean(errs[k])) for k in ("mc", "gauss_sk_b", "blend_mc_gsk_b")}, flush=True)

    base = float(np.mean(errs["mc"]))
    print("mc", base, "ref", 7.3833e-07, flush=True)
    for k in keys:
        m = float(np.mean(errs[k]))
        print(f"{k:16s} mse={m:.6e} r={base / m:.4f} adj≈{m * 0.408:.6e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decisive: corr(Hadamard err, gain-cov err) at L31 — CV premise gate.

#18085: CV needs ρ≳0.95. If ρ≪ that, kill gain/analytic CV vs Hadamard.
Also report cosine of error vectors and R² of regressing H-err on G-err.
Protocol 3.0.
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


def hada_mu(weights, seed: int) -> np.ndarray:
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def gain_mu(weights) -> np.ndarray:
    """Diagonal gain-cov final post-ReLU mean (cheap analytic)."""
    # Input: N(0,I) → after radius we'd differ; use classic unit-var Gaussian chain
    # Matching train_layer_closure / classical: μ=0, var=1 at input
    mu = np.zeros(W, np.float64)
    var = np.ones(W, np.float64)
    for w in weights:
        Wnp = np.asarray(w, np.float64)
        mu_pre = Wnp.T @ mu
        var_pre = np.maximum((Wnp.T ** 2) @ var, 1e-18)
        sig = np.sqrt(var_pre)
        alpha = mu_pre / sig
        # flopscope pdf/cdf
        af = fnp.asarray(alpha.astype(np.float32), dtype=fnp.float32)
        phi = np.asarray(flops.stats.norm.pdf(af), np.float64)
        Phi = np.asarray(flops.stats.norm.cdf(af), np.float64)
        mu = mu_pre * Phi + sig * phi
        # post-ReLU var (diag): E[relu^2] - mu^2
        # E[z_+^2] = (μ²+σ²)Φ + μσ φ
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sig * phi
        var = np.maximum(ez2 - mu**2, 1e-18)
    return mu


def main() -> None:
    ds = Dataset.load_from_disk(str(MINI))
    rhos = []
    cosines = []
    mse_h, mse_g = [], []
    # pooled regression: eh ≈ β eg
    num = den = 0.0
    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF
        ws = list(mlp.weights)
        mh = hada_mu(ws, seed)
        mg = gain_mu(ws)
        eh, eg = mh - Y, mg - Y
        mse_h.append(float(np.mean(eh**2)))
        mse_g.append(float(np.mean(eg**2)))
        # per-MLP neuron-wise corr
        if eh.std() > 0 and eg.std() > 0:
            rhos.append(float(np.corrcoef(eh, eg)[0, 1]))
            cosines.append(float(np.dot(eh, eg) / (np.linalg.norm(eh) * np.linalg.norm(eg) + 1e-30)))
        num += float(np.dot(eh, eg))
        den += float(np.dot(eg, eg))
        if i % 20 == 0:
            print(i, "rho", rhos[-1] if rhos else None, "mse_h", mse_h[-1], flush=True)

    beta = num / (den + 1e-30)
    # CV: pred = mh - beta*(mg - c); optimal c so E[mg-c]=0 → use mg mean... for mean estimand
    # Standard: μ_cv = μ_h - β (μ_g - E[μ_g]). We don't know E[μ_g]; use 0 target wrong.
    # Better: μ_cv = μ_h - β (μ_g - μ_g_bar) within leave-one-out — skip, just report ρ.
    print(
        f"mean_rho={float(np.mean(rhos)):.4f} median_rho={float(np.median(rhos)):.4f} "
        f"mean_cos={float(np.mean(cosines)):.4f}",
        flush=True,
    )
    print(
        f"mse_h={float(np.mean(mse_h)):.6e} mse_g={float(np.mean(mse_g)):.6e} "
        f"pooled_beta={beta:.4f}",
        flush=True,
    )
    gate = float(np.mean(rhos))
    print(f"CV_GATE {'PASS' if gate >= 0.95 else 'FAIL'} (need mean ρ≥0.95, got {gate:.4f})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

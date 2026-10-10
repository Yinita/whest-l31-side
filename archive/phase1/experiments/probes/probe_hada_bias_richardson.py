#!/usr/bin/env python3
"""Bias decomposition + Richardson / nested half-N debias on Hadamard."""
from __future__ import annotations

import importlib.util
import time
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


def n_frames_budget(hada):
    width = 256
    depth = 32
    n2 = float(width * width)
    flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
    n_forward = int(max(2 * width, min(target / flops_per, hada._MAX_SAMPLES)))
    return max(2, n_forward // (2 * width))


def forward_mean(hada, weights, seed, n_frames, e_r):
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(n_frames, 256, seed=seed, e_r=e_r)
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def forward_nested_halves(hada, weights, seed, n_frames, e_r):
    """One stream: return (μ_full, μ_first_half) with nested frames."""
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(n_frames, 256, seed=seed, e_r=e_r)
        # x has 2*n_frames rows (antipodal pairs). First half of frames = first n_frames rows.
        n_half_rows = (n_frames // 2) * 2
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        x_np = np.asarray(x, np.float64)
        mu_full = x_np.mean(0)
        mu_half = x_np[:n_half_rows].mean(0)
        return mu_full, mu_half


def main():
    hada = load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    NF = n_frames_budget(hada)
    e_r = hada._chi_mean_radius(256)
    print(f"NF={NF}", flush=True)

    # Collect dual independent streams for bias/noise split
    errs_single = []
    bias2 = []
    var_est = []
    # Richardson variants
    errs = {
        "full": [],
        "half": [],
        "rich_1": [],  # 2*full - half (nested)
        "rich_0.5": [],  # full + 0.5*(full-half)
        "rich_0.25": [],
        "rich_neg0.25": [],  # shrink toward half (anti-Richardson)
        "dual_mean": [],  # 0.5*(A+B) same total frames
        "A_only_halfN": [],
    }

    t0 = time.time()
    n = len(ds)
    for i in range(n):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"][-1], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF

        mu_f, mu_h = forward_nested_halves(hada, weights, seed, NF, e_r)
        # independent half-N second stream (for dual / bias split)
        n_a = NF // 2
        n_b = NF - n_a
        mu_a = forward_mean(hada, weights, seed, n_a, e_r)
        mu_b = forward_mean(hada, weights, seed ^ 0xA5A5A5A5, n_b, e_r)
        mu_dual = 0.5 * (mu_a + mu_b)

        # bias/noise from two independent full-N would need 2x budget;
        # use two half-N as proxy for E[μ]≈0.5(A+B), noise from A-B
        # Better: two independent full-N at same NF costs 2x — instead use
        # classical: with μ1=mu_a, μ2=mu_b (same N/2):
        # bias^2 ≈ ||0.5(A+B) - Y||^2 - 0.25*||A-B||^2 / 2? 
        # For equal-var independent: E||μ̄-Y||^2 = bias^2 + var/2
        # E||A-Y||^2 = bias^2 + var
        # So bias^2 = 2*MSE(dual) - MSE(A)  if same bias for A and dual... 
        # Actually A has larger var. Standard:
        # Let μ_bar = (A+B)/2. MSE_bar = b^2 + v/2, MSE_A = b^2 + v
        # => b^2 = 2*MSE_bar - MSE_A, v = 2*(MSE_A - MSE_bar)
        mse_a = float(np.mean((mu_a - Y) ** 2))
        mse_dual = float(np.mean((mu_dual - Y) ** 2))
        b2 = 2.0 * mse_dual - mse_a
        v = 2.0 * (mse_a - mse_dual)
        bias2.append(max(b2, 0.0))
        var_est.append(max(v, 0.0))
        errs_single.append(float(np.mean((mu_f - Y) ** 2)))

        delta = mu_f - mu_h
        errs["full"].append(float(np.mean((mu_f - Y) ** 2)))
        errs["half"].append(float(np.mean((mu_h - Y) ** 2)))
        errs["rich_1"].append(float(np.mean((mu_f + 1.0 * delta - Y) ** 2)))
        errs["rich_0.5"].append(float(np.mean((mu_f + 0.5 * delta - Y) ** 2)))
        errs["rich_0.25"].append(float(np.mean((mu_f + 0.25 * delta - Y) ** 2)))
        errs["rich_neg0.25"].append(float(np.mean((mu_f - 0.25 * delta - Y) ** 2)))
        errs["dual_mean"].append(mse_dual)
        errs["A_only_halfN"].append(mse_a)

        if i % 10 == 0:
            print(
                f"[{i}] full={errs['full'][-1]:.3e} rich0.5={errs['rich_0.5'][-1]:.3e} "
                f"dual={mse_dual:.3e} b2={max(b2,0):.3e}",
                flush=True,
            )

    base = float(np.mean(errs["full"]))
    print(f"\n=== Mini n={n} NF={NF} base_full={base:.6e} ===", flush=True)
    b2m = float(np.mean(bias2))
    vm = float(np.mean(var_est))
    print(
        f"halfN bias/noise proxy: bias2={b2m:.4e} var={vm:.4e} "
        f"bias_frac={b2m / max(b2m + vm, 1e-30):.3f}",
        flush=True,
    )
    ranked = sorted(((float(np.mean(errs[k])), k) for k in errs), key=lambda t: t[0])
    for m, k in ranked:
        print(
            f"{k:14s} mse={m:.6e} ratio={base / m:.4f} proxy_adj={m * 0.40:.6e}",
            flush=True,
        )
    print(f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Two independent full-N Hadamard streams → bias/noise split."""
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


def main():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    hada = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(hada)
    ds = Dataset.load_from_disk(str(MINI))

    n2 = 256 * 256
    flops_per = 32 * 2 * n2 + 32 * 256 * 20
    NF = max(1, int(max(512, min(B * 0.40 * 1.036 / flops_per, 500_000))) // 512)
    e_r = hada._chi_mean_radius(256)
    print("NF", NF, flush=True)

    def fwd(weights, seed):
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(NF, 256, seed=seed, e_r=e_r)
            for w in weights:
                x = fnp.maximum(x @ w, 0.0)
            return np.asarray(fnp.mean(x, 0), np.float64)

    mse_a, mse_bar, bias_vecs = [], [], []
    for i in range(len(ds)):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"][-1], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        a = fwd(weights, seed)
        b = fwd(weights, seed ^ 0xA5A5A5A5)
        bar = 0.5 * (a + b)
        mse_a.append(float(np.mean((a - Y) ** 2)))
        mse_bar.append(float(np.mean((bar - Y) ** 2)))
        bias_vecs.append(bar - Y)
        if i % 20 == 0:
            ma, mb = float(np.mean(mse_a)), float(np.mean(mse_bar))
            b2, v = 2 * mb - ma, 2 * (ma - mb)
            print(
                f"[{i}] MSEa={ma:.4e} MSEbar={mb:.4e} b2={b2:.4e} v={v:.4e} "
                f"frac={b2 / max(b2 + v, 1e-30):.3f}",
                flush=True,
            )

    ma, mb = float(np.mean(mse_a)), float(np.mean(mse_bar))
    b2, v = 2 * mb - ma, 2 * (ma - mb)
    print(
        f"FINAL MSEa={ma:.6e} MSEbar={mb:.6e} bias2={b2:.6e} var={v:.6e} "
        f"bias_frac={b2 / max(b2 + v, 1e-30):.4f}",
        flush=True,
    )
    print(
        f"noise floor (perfect debias) raw≈{max(v, 0):.6e} proxy_adj≈{max(v, 0) * 0.40:.6e}",
        flush=True,
    )
    Bmat = np.stack(bias_vecs)
    print(
        f"mean bias L2={np.sqrt(np.mean(np.sum(Bmat ** 2, 1))):.4e} "
        f"mean|bias|={np.mean(np.abs(Bmat)):.4e}",
        flush=True,
    )
    _, S, _ = np.linalg.svd(Bmat - Bmat.mean(0), full_matrices=False)
    cum = np.cumsum(S**2) / np.sum(S**2)
    print(
        "bias SVD cum energy top1/4/16/64:",
        float(cum[0]),
        float(cum[3]),
        float(cum[15]),
        float(cum[63]),
        flush=True,
    )


if __name__ == "__main__":
    main()

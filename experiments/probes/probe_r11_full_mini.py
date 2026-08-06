#!/usr/bin/env python3
"""Full Mini H vs H@R11 at full frames (BudgetContext)."""
from __future__ import annotations

import math
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
R_PATH = ROOT / "experiments" / "results" / "R11_probe.npy"
B = int(2.72e11)
UTIL, CALIB, MATMUL = 0.40, 1.036, 2.0


def chi_mean(d: int) -> float:
    h = 0.5 * d
    return math.sqrt(2) * math.exp(math.lgamma(h + 0.5) - math.lgamma(h))


def sylvester(d: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h.astype(np.float64)


def pred(weights, base, seed, n_frames, e_r):
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_frames):
        perm = rng.permutation(256)
        signs = rng.choice([-1.0, 1.0], size=256)
        M = base[perm] * signs.reshape(1, -1) * e_r
        blocks += [M, -M]
    x = np.concatenate(blocks, 0).astype(np.float32)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        xf = fnp.asarray(x)
        for w in weights:
            xf = fnp.maximum(xf @ w, 0.0)
        return np.asarray(fnp.mean(xf, 0), np.float64)


def main():
    ds = Dataset.load_from_disk(str(MINI))
    H = sylvester(256) / math.sqrt(256)
    R = np.load(R_PATH)
    U = H @ R
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    e_r = chi_mean(256)

    n2 = 256 * 256
    flops_per = 32 * MATMUL * n2 + 32 * 256 * 20
    n_forward = int(max(512, min(B * UTIL * CALIB / flops_per, 500_000)))
    n_frames = max(1, n_forward // 512)
    print(f"n_frames={n_frames} n_forward≈{n_frames * 512}", flush=True)

    mses_h, mses_u = [], []
    t0 = time.time()
    for i in range(len(ds)):
        ex = ds[i]
        Y = np.asarray(ex["all_layer_means"][-1], np.float64)
        w = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        mh = pred(w, H, seed, n_frames, e_r)
        mu = pred(w, U, seed, n_frames, e_r)
        mses_h.append(float(np.mean((mh - Y) ** 2)))
        mses_u.append(float(np.mean((mu - Y) ** 2)))
        if i % 10 == 0:
            gh = float(np.mean(mses_h))
            gu = float(np.mean(mses_u))
            print(
                f"[{i}] H={gh:.4e} U={gu:.4e} gain={gh / gu:.3f}",
                flush=True,
            )

    raw_h = float(np.mean(mses_h))
    raw_u = float(np.mean(mses_u))
    print(
        f"FULL Mini raw H={raw_h:.6e} U={raw_u:.6e} gain={raw_h / raw_u:.4f}",
        flush=True,
    )
    print(
        f"proxy_adj H={raw_h * 0.40:.6e} U={raw_u * 0.40:.6e}",
        flush=True,
    )
    print(f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

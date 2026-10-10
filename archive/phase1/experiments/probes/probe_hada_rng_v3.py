#!/usr/bin/env python3
"""Compare Hadamard frame RNG schemes (protocol 3.0) vs pure-0.2.1s base.

Variants:
  base     — sequential fnp RNG (canonical)
  spawn    — SeedSequence.spawn(NF) independent perms/signs
  scramble — nested: signs from spawn stream A, perms from B
  anti_sign — each frame also includes sign-flipped twin before antipodal
              (would double frames — instead: flip half the frames' global sign)
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
Hnp = np.asarray(hada._H256 * hada._INV_SQRT_256, np.float32)
scale = np.float32(e_r)
_PM = fnp.asarray([-1.0, 1.0], dtype=fnp.float32)


def forward_blocks(weights, blocks):
    x = fnp.asarray(np.concatenate(blocks, 0), dtype=fnp.float32)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def blocks_base(seed: int):
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
        return x  # already concatenated; special-case forward


def forward_base(weights, seed: int):
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def blocks_spawn(seed: int):
    # Independent child streams per frame
    children = np.random.SeedSequence(int(seed) & 0x7FFFFFFF).spawn(NF)
    blocks = []
    for ch in children:
        rng = np.random.default_rng(ch)
        perm = rng.permutation(W)
        signs = rng.choice([-1.0, 1.0], size=W).astype(np.float32)
        M = Hnp[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks += [pos, -pos]
    return blocks


def blocks_fnp_spawn(seed: int):
    """Approximate spawn using disjoint fnp seeds seed+10007*i."""
    blocks = []
    base = int(seed) & 0x7FFFFFFF
    for i in range(NF):
        rng = fnp.random.default_rng((base + 10007 * i) & 0x7FFFFFFF)
        perm = np.asarray(rng.permutation(W))
        signs = np.asarray(rng.choice(_PM, size=W))
        M = Hnp[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks += [pos, -pos]
    return blocks


def blocks_pair_xor(seed: int):
    """NF/2 frames from seed, NF/2 from seed^mask — like dual but interleaved one pass."""
    blocks = []
    for s in (int(seed) & 0x7FFFFFFF, (int(seed) ^ 0xA5A5A5A5) & 0x7FFFFFFF):
        rng = fnp.random.default_rng(s)
        for _ in range(NF // 2):
            perm = np.asarray(rng.permutation(W))
            signs = np.asarray(rng.choice(_PM, size=W))
            M = Hnp[perm] * signs.reshape(1, -1)
            pos = M * scale
            blocks += [pos, -pos]
    return blocks


def main() -> None:
    ds = Dataset.load_from_disk(str(MINI))
    methods = {
        "base": None,
        "np_spawn": blocks_spawn,
        "fnp_stride": blocks_fnp_spawn,
        "pair_xor": blocks_pair_xor,
    }
    errs = {k: [] for k in methods}
    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        ws = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF
        errs["base"].append(float(np.mean((forward_base(ws, seed) - Y) ** 2)))
        for k, fn in methods.items():
            if k == "base":
                continue
            pred = forward_blocks(ws, fn(seed))
            errs[k].append(float(np.mean((pred - Y) ** 2)))
        if i % 20 == 0:
            print(i, {k: float(np.mean(errs[k])) for k in methods}, flush=True)

    base = float(np.mean(errs["base"]))
    print("base", base, "ref", 7.3833e-07, flush=True)
    for k in methods:
        m = float(np.mean(errs[k]))
        print(f"{k:12s} mse={m:.6e} r={base / m:.4f} adj≈{m * 0.408:.6e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Focused Hadamard column-sign designs (full-ONB frames → perms moot)."""
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
W = 256


def load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def n_frames(hada):
    n2 = float(W * W)
    flops_per = 32 * hada._MATMUL_PER_LAYER * n2 + 32 * W * 20.0
    target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
    n_forward = int(max(2 * W, min(target / flops_per, hada._MAX_SAMPLES)))
    return max(1, n_forward // (2 * W))


def sylvester_pm(d: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h.astype(np.float64)


def make_signs(kind: str, nf: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "random":
        return rng.choice([-1.0, 1.0], size=(nf, W))
    if kind == "balanced":
        S = np.ones((nf, W), np.float64)
        n_neg = nf // 2
        for j in range(W):
            idx = rng.permutation(nf)[:n_neg]
            S[idx, j] = -1.0
        return S
    if kind == "hada_colperm":
        H = sylvester_pm(W)
        rows = rng.permutation(W)[:nf]
        return H[rows][:, rng.permutation(W)]
    if kind == "balanced_hada":
        # start from hada rows, then flip columns to balance ± counts
        H = sylvester_pm(W)
        rows = rng.permutation(W)[:nf]
        S = H[rows].copy()
        for j in range(W):
            n_pos = int(np.sum(S[:, j] > 0))
            target = nf // 2
            if n_pos > target:
                flip = np.where(S[:, j] > 0)[0]
                rng.shuffle(flip)
                S[flip[: n_pos - target], j] *= -1
            elif n_pos < target:
                flip = np.where(S[:, j] < 0)[0]
                rng.shuffle(flip)
                S[flip[: target - n_pos], j] *= -1
        return S
    raise ValueError(kind)


def forward_signs(hada, weights, signs, e_r):
    H = np.asarray(hada._H256 * hada._INV_SQRT_256, np.float64)
    scale = float(e_r)
    blocks = []
    for i in range(signs.shape[0]):
        M = H * signs[i].reshape(1, -1)  # no perm — full ONB set unchanged
        pos = (M * scale).astype(np.float32)
        blocks += [pos, -pos]
    x = fnp.asarray(np.concatenate(blocks, 0), dtype=fnp.float32)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def main():
    hada = load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    NF = n_frames(hada)
    e_r = hada._chi_mean_radius(W)
    print(f"NF={NF}", flush=True)

    kinds = ["baseline", "random_np", "balanced", "hada_colperm", "balanced_hada"]
    errs = {k: [] for k in kinds}

    for i in range(len(ds)):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"][-1], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF

        # production baseline
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
            for w in weights:
                x = fnp.maximum(x @ w, 0.0)
            errs["baseline"].append(
                float(np.mean((np.asarray(fnp.mean(x, 0), np.float64) - Y) ** 2))
            )

        for kind in kinds[1:]:
            sk = "random" if kind == "random_np" else kind
            S = make_signs(sk, NF, seed ^ (hash(kind) & 0x7FFFFFFF))
            pred = forward_signs(hada, weights, S, e_r)
            errs[kind].append(float(np.mean((pred - Y) ** 2)))

        if i % 10 == 0:
            b = float(np.mean(errs["baseline"]))
            best = min((float(np.mean(errs[k])), k) for k in kinds)
            print(
                f"[{i}] base={b:.4e} best={best[1]}={best[0]:.4e} r={b / best[0]:.4f}",
                flush=True,
            )

    base = float(np.mean(errs["baseline"]))
    print(f"\n=== Mini n={len(ds)} base={base:.6e} ===", flush=True)
    for m, k in sorted((float(np.mean(errs[k])), k) for k in kinds):
        print(
            f"{k:14s} mse={m:.6e} ratio={base / m:.4f} proxy_adj={m * 0.408:.6e}",
            flush=True,
        )


if __name__ == "__main__":
    main()

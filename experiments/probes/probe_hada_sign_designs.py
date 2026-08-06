#!/usr/bin/env python3
"""Variance designs on Hadamard column-sign patterns (perms are moot for full ONB frames)."""
from __future__ import annotations

import importlib.util
import math
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
    """Return (nf, W) of ±1 sign patterns."""
    rng = np.random.default_rng(seed)
    if kind == "random":
        return rng.choice([-1.0, 1.0], size=(nf, W))
    if kind == "noperm_random":
        # same as random — used with identity perm
        return rng.choice([-1.0, 1.0], size=(nf, W))
    if kind == "balanced":
        # each column: exactly floor(nf/2) minus, rest plus; shuffle per col
        S = np.ones((nf, W), np.float64)
        n_neg = nf // 2
        for j in range(W):
            idx = rng.permutation(nf)[:n_neg]
            S[idx, j] = -1.0
        return S
    if kind == "hada_rows":
        # use nf distinct Sylvester rows as sign patterns; seed permutes which
        H = sylvester_pm(W)
        rows = rng.permutation(W)[:nf]
        return H[rows]
    if kind == "hada_rows_colperm":
        H = sylvester_pm(W)
        rows = rng.permutation(W)[:nf]
        S = H[rows]
        col = rng.permutation(W)
        return S[:, col]
    if kind == "orthog_approx":
        # QR of random Gaussian → take first nf rows of Q signs via sign(Q)
        A = rng.standard_normal((W, W))
        Q, _ = np.linalg.qr(A)
        S = np.sign(Q[:nf])
        S[S == 0] = 1.0
        return S
    if kind == "maxmin_greedy":
        # greedy max-min Hamming among random candidates
        pool = rng.choice([-1.0, 1.0], size=(nf * 8, W))
        chosen = [pool[0]]
        for _ in range(1, nf):
            best_i, best_d = 0, -1
            for i in range(len(pool)):
                d = min(int(np.sum(pool[i] != c)) for c in chosen)
                if d > best_d:
                    best_d, best_i = d, i
            chosen.append(pool[best_i])
        return np.stack(chosen)
    raise ValueError(kind)


def forward_with_signs(hada, weights, signs: np.ndarray, e_r: float, use_perm: bool, seed: int):
    H = np.asarray(hada._H256 * hada._INV_SQRT_256, np.float64)
    scale = float(e_r)
    rng = np.random.default_rng(seed ^ 0x1111)
    blocks = []
    nf = signs.shape[0]
    for i in range(nf):
        if use_perm:
            perm = rng.permutation(W)
            M = H[perm] * signs[i].reshape(1, -1)
        else:
            M = H * signs[i].reshape(1, -1)
        pos = (M * scale).astype(np.float32)
        blocks.append(pos)
        blocks.append(-pos)
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

    kinds = [
        ("baseline", True, "random"),  # current: random perm + random signs
        ("noperm_rand", False, "noperm_random"),  # drop perm
        ("balanced", True, "balanced"),
        ("balanced_np", False, "balanced"),
        ("hada_rows", False, "hada_rows"),
        ("hada_colperm", False, "hada_rows_colperm"),
        ("orthog_sg", False, "orthog_approx"),
        ("maxmin", False, "maxmin_greedy"),
    ]
    errs = {k[0]: [] for k in kinds}

    for i in range(len(ds)):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"][-1], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        for name, use_perm, kind in kinds:
            S = make_signs(kind, NF, seed ^ (hash(name) & 0xFFFFFFFF))
            # baseline uses same seed path as production for fairness
            if name == "baseline":
                with flops.BudgetContext(flop_budget=B * 2, quiet=True):
                    x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
                    for w in weights:
                        x = fnp.maximum(x @ w, 0.0)
                    pred = np.asarray(fnp.mean(x, axis=0), np.float64)
            else:
                pred = forward_with_signs(
                    hada, weights, S, e_r, use_perm=use_perm, seed=seed
                )
            errs[name].append(float(np.mean((pred - Y) ** 2)))
        if i % 10 == 0:
            b = float(np.mean(errs["baseline"]))
            best = min((float(np.mean(errs[k])), k) for k in errs)
            print(
                f"[{i}] base={b:.4e} best={best[1]}={best[0]:.4e} r={b / best[0]:.4f}",
                flush=True,
            )

    base = float(np.mean(errs["baseline"]))
    print(f"\n=== Mini n={len(ds)} base={base:.6e} ===", flush=True)
    ranked = sorted(((float(np.mean(errs[k])), k) for k in errs), key=lambda t: t[0])
    for m, k in ranked:
        print(
            f"{k:14s} mse={m:.6e} ratio={base / m:.4f} proxy_adj={m * 0.408:.6e}",
            flush=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hadamard pathwise CVs that survive antipodal (even / ReLU controls).

Odd controls (L0, pre1) have empirical mean ≈0 under ± pairs → zero correction.
"""
from __future__ import annotations

import importlib.util
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
B = int(2.72e11)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def multi_cv(h: np.ndarray, c: np.ndarray, e_c: np.ndarray, ridge_rel: float) -> np.ndarray:
    """CV with known E[c]=e_c. Do NOT empirically recenter c (would cancel E)."""
    n = float(h.shape[0])
    mu = h.mean(0)
    cc = c - e_c.reshape(1, -1)
    gram = (cc.T @ cc) / n
    scale = float(np.mean(np.diag(gram))) + 1e-18
    gram = gram + ridge_rel * scale * np.eye(gram.shape[0])
    cov = ((h - mu).T @ cc) / n
    beta = np.linalg.solve(gram, cov.T).T
    return mu - beta @ cc.mean(0)


def proj(c: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Project controls and expectations with same ±1 matrix."""
    rng = np.random.default_rng(seed)
    d = c.shape[1]
    P = rng.choice([-1.0, 1.0], size=(d, k)) / math.sqrt(d)
    # For expectations: only valid if we project e the same way when e is vector
    return c @ P, P  # caller projects e


def main():
    hada = load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    e_r = hada._chi_mean_radius(256)
    ex2 = (e_r * e_r) / 256.0  # E[x_i^2]

    # variants built per-MLP from w0
    # We'll compute a fixed list of names dynamically
    names_order = []
    errs = {}

    def add(name, pred, Y):
        if name not in errs:
            errs[name] = []
            names_order.append(name)
        errs[name].append(float(np.mean((pred - Y) ** 2)))

    t0 = time.time()
    n = len(ds)
    for i in range(n):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"][-1], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        w0 = np.asarray(weights[0], np.float64)
        col_n2 = np.sum(w0 * w0, axis=0)  # ||col||^2
        col_n = np.sqrt(np.maximum(col_n2, 1e-30))

        width, depth = 256, 32
        n2 = float(width * width)
        flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * width * 20.0
        target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
        n_forward = int(max(2 * width, min(target / flops_per, hada._MAX_SAMPLES)))
        n_frames = max(1, n_forward // (2 * width))

        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x0 = hada._hadamard_directions(n_frames, width, seed=seed, e_r=e_r)
            x0_np = np.asarray(x0, np.float64)
            pre1 = x0_np @ w0
            a1 = np.maximum(pre1, 0.0)
            h = fnp.asarray(a1, dtype=fnp.float32)
            # also keep mid layer 15
            a_mid = None
            for li, w in enumerate(weights[1:], start=1):
                h = fnp.maximum(h @ w, 0.0)
                if li == 15:
                    a_mid = np.asarray(h, np.float64)
            last = np.asarray(h, np.float64)

        add("none", last.mean(0), Y)

        # --- even / ReLU CVs ---
        # 1) pre^2 with exact E under 2nd-moment match
        e_pre2 = ex2 * col_n2
        for ridge in (1e-2, 1e-3, 1e-4):
            add(f"pre2_r{ridge:.0e}", multi_cv(last, pre1 * pre1, e_pre2, ridge), Y)

        # 2) scalar sum pre^2
        e_spre2 = float(np.sum(e_pre2))
        add(
            "spre2",
            multi_cv(last, (pre1 * pre1).sum(1, keepdims=True), np.array([e_spre2]), 1e-3),
            Y,
        )

        # 3) L1 relu with Gaussian E (known kill risk) + sphere note
        e_a1 = col_n * INV_SQRT_2PI
        for ridge in (1e-2, 1e-3, 1e-4):
            add(f"a1_r{ridge:.0e}", multi_cv(last, a1, e_a1, ridge), Y)

        # 4) a1^2 : for Gaussian half-normal E[relu^2]=0.5 σ^2
        e_a1sq = 0.5 * col_n2
        for ridge in (1e-2, 1e-3):
            add(f"a1sq_r{ridge:.0e}", multi_cv(last, a1 * a1, e_a1sq, ridge), Y)

        # 5) scalar sum a1, sum a1^2
        add(
            "sa1",
            multi_cv(last, a1.sum(1, keepdims=True), np.array([float(e_a1.sum())]), 1e-3),
            Y,
        )
        add(
            "sa1sq",
            multi_cv(
                last,
                (a1 * a1).sum(1, keepdims=True),
                np.array([float(e_a1sq.sum())]),
                1e-3,
            ),
            Y,
        )

        # 6) projected a1 (k=16,32) — fewer params
        for k in (16, 32):
            rng = np.random.default_rng(seed ^ (0xA1 + k))
            P = rng.choice([-1.0, 1.0], size=(256, k)) / math.sqrt(256.0)
            c = a1 @ P
            e = e_a1 @ P
            add(f"a1p{k}", multi_cv(last, c, e, 1e-3), Y)

        # 7) projected pre2
        for k in (16, 32):
            rng = np.random.default_rng(seed ^ (0xB2 + k))
            P = rng.choice([-1.0, 1.0], size=(256, k)) / math.sqrt(256.0)
            c = (pre1 * pre1) @ P
            e = e_pre2 @ P
            add(f"pre2p{k}", multi_cv(last, c, e, 1e-3), Y)

        # 8) a1 + a1sq combined projected
        rng = np.random.default_rng(seed ^ 0xC3)
        P = rng.choice([-1.0, 1.0], size=(256, 16)) / math.sqrt(256.0)
        c = np.concatenate([a1 @ P, (a1 * a1) @ P], 1)
        e = np.concatenate([e_a1 @ P, e_a1sq @ P])
        add("a1+a1sq_p16", multi_cv(last, c, e, 1e-3), Y)

        # 9) mid-layer a15 as CV with E=sample? skip — no known E
        # use a15^2 with gain-based E? skip for now

        if i % 10 == 0:
            b = float(np.mean(errs["none"]))
            best = min((float(np.mean(errs[k])), k) for k in names_order)
            print(
                f"[{i}] none={b:.4e} best={best[1]}={best[0]:.4e} r={b / best[0]:.4f}",
                flush=True,
            )

    base = float(np.mean(errs["none"]))
    print(f"\n=== Full Mini n={n} base={base:.6e} ===", flush=True)
    ranked = sorted(((float(np.mean(errs[k])), k) for k in names_order), key=lambda t: t[0])
    for m, k in ranked:
        print(
            f"{k:18s} mse={m:.6e} ratio={base / m:.4f} proxy_adj={m * 0.40:.6e}",
            flush=True,
        )
    print(f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

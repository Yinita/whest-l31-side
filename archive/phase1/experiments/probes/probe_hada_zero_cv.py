#!/usr/bin/env python3
"""Hadamard pathwise CVs with exact E[control]=0 (L0 / pre-ReLU L1).

L1-ReLU CV previously killed (~9e-07) — Gaussian E[relu] wrong / high-dim.
Here controls have exact mean 0 under antipodal Hadamard.
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


def load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def multi_cv(h: np.ndarray, c: np.ndarray, ridge_rel: float) -> np.ndarray:
    """h (n,d), c (n,k) with E[c]=0 known. Return CV-adjusted mean."""
    n = float(h.shape[0])
    mu = h.mean(0)
    cc = c  # already mean-zero in expectation; still center empirically for ridge stab
    cc = cc - cc.mean(0, keepdims=True)
    gram = (cc.T @ cc) / n
    scale = float(np.mean(np.diag(gram))) + 1e-18
    gram = gram + ridge_rel * scale * np.eye(gram.shape[0])
    cov = ((h - mu).T @ cc) / n  # (d,k)
    beta = np.linalg.solve(gram, cov.T).T  # (d,k)
    # E[c]=0 → subtract beta @ mean(c_emp); using centered cc mean is 0,
    # so use uncentered mean for the correction term
    return mu - beta @ c.mean(0)


def proj_controls(c: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Random ±1/√d projections of controls → (n,k)."""
    rng = np.random.default_rng(seed)
    d = c.shape[1]
    P = rng.choice([-1.0, 1.0], size=(d, k)) / math.sqrt(d)
    return c @ P


def run_mlp(hada, weights, seed, variants):
    width, depth = 256, 32
    n2 = float(width * width)
    flops_per = depth * hada._MATMUL_PER_LAYER * n2 + depth * width * 20.0
    target = max(float(B) * hada._UTIL * hada._UTIL_CALIB, flops_per)
    n_forward = int(max(2 * width, min(target / flops_per, hada._MAX_SAMPLES)))
    n_frames = max(1, n_forward // (2 * width))
    e_r = hada._chi_mean_radius(width)

    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x0 = hada._hadamard_directions(n_frames, width, seed=seed, e_r=e_r)
        x0_np = np.asarray(x0, np.float64)
        w0 = weights[0]
        pre1 = np.asarray(x0 @ w0, np.float64)  # E[pre1]=0 exact
        h = fnp.maximum(x0 @ w0, 0.0)
        for w in weights[1:]:
            h = fnp.maximum(h @ w, 0.0)
        last = np.asarray(h, np.float64)

    out = {}
    out["none"] = last.mean(0)
    for name, ctrl, ridge in variants:
        if ctrl == "L0":
            c = x0_np
        elif ctrl == "pre1":
            c = pre1
        elif ctrl.startswith("L0p"):
            k = int(ctrl[3:])
            c = proj_controls(x0_np, k, seed=seed ^ 0xC0FFEE)
        elif ctrl.startswith("pre1p"):
            k = int(ctrl[5:])
            c = proj_controls(pre1, k, seed=seed ^ 0xBEEF)
        elif ctrl == "L0sum":
            c = x0_np.sum(1, keepdims=True)
        elif ctrl == "pre1sum":
            c = pre1.sum(1, keepdims=True)
        elif ctrl == "L0+pre1p16":
            c = np.concatenate(
                [
                    proj_controls(x0_np, 16, seed ^ 1),
                    proj_controls(pre1, 16, seed ^ 2),
                ],
                1,
            )
        else:
            raise ValueError(ctrl)
        out[name] = multi_cv(last, c, ridge)
    return out


def main():
    hada = load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    # name, control_spec, ridge
    ridges = [1e-2, 1e-3, 1e-4]
    variants = []
    for r in ridges:
        variants.append((f"L0_r{r:.0e}", "L0", r))
        variants.append((f"pre1_r{r:.0e}", "pre1", r))
    for k in [8, 16, 32, 64]:
        variants.append((f"L0p{k}_r1e-3", f"L0p{k}", 1e-3))
        variants.append((f"pre1p{k}_r1e-3", f"pre1p{k}", 1e-3))
    variants.append(("L0sum_r1e-3", "L0sum", 1e-3))
    variants.append(("pre1sum_r1e-3", "pre1sum", 1e-3))
    variants.append(("L0+pre1p16_r1e-3", "L0+pre1p16", 1e-3))

    names = ["none"] + [v[0] for v in variants]
    errs = {n: [] for n in names}
    t0 = time.time()
    n = len(ds)
    for i in range(n):
        ex = ds[i]
        weights = [fnp.asarray(ex["weights"][l], dtype=fnp.float32) for l in range(32)]
        Y = np.asarray(ex["all_layer_means"][-1], np.float64)
        seed = int(ex["mlp_seed"]) & 0x7FFFFFFF
        preds = run_mlp(hada, weights, seed, variants)
        for nme, p in preds.items():
            errs[nme].append(float(np.mean((p - Y) ** 2)))
        if i % 10 == 0:
            b = float(np.mean(errs["none"]))
            best = min(
                ((float(np.mean(errs[k])), k) for k in names if k != "none" and errs[k]),
                default=(b, "none"),
            )
            print(
                f"[{i}] none={b:.4e} best={best[1]}={best[0]:.4e} "
                f"r={b / best[0]:.3f}",
                flush=True,
            )

    base = float(np.mean(errs["none"]))
    print(f"\n=== Full Mini n={n} base={base:.6e} proxy_adj={base * 0.40:.6e} ===")
    ranked = sorted(
        ((float(np.mean(errs[k])), k) for k in names),
        key=lambda t: t[0],
    )
    for m, k in ranked:
        print(
            f"{k:22s} mse={m:.6e} ratio={base / m:.4f} proxy_adj={m * 0.40:.6e}",
            flush=True,
        )
    print(f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

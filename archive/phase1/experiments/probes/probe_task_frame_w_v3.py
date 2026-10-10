#!/usr/bin/env python3
"""P2b task-driven: Full-supervised Hadamard frame weights → Mini gate.

Streams Full shards (no full-weight RAM). Protocol 3.0 seeds.
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
CACHE = ROOT / "experiments/results/full_cache"
OUT = ROOT / "experiments/probes/results"
B = int(2.72e11)
W = 256
D = 32
NF = 50
SPS = 2 * W
e_r = hada._chi_mean_radius(W)


def estimator_seed(raw: int) -> int:
    ss = fnp.random.SeedSequence(int(raw)).spawn(3)
    return int(ss[2].generate_state(1)[0]) & 0x7FFFFFFF


def frame_means_from_wts(Wts: np.ndarray, seed: int) -> np.ndarray:
    """Wts: (D, W, W) float32 numpy."""
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
        for li in range(D):
            x = fnp.maximum(x @ fnp.asarray(Wts[li], dtype=fnp.float32), 0.0)
        last = np.asarray(x, np.float64).reshape(NF, SPS, W)
    return last.mean(1)


def fit_shared_w(means: np.ndarray, Y: np.ndarray, ridge: float, tr: np.ndarray) -> np.ndarray:
    """means (n, NF, W), Y (n, W). Ridge toward uniform, sum w = 1."""
    u = np.full(NF, 1.0 / NF)
    A = ridge * np.eye(NF)
    b = ridge * u
    for i in tr:
        M = means[i]  # (NF, W)
        A = A + M @ M.T
        b = b + M @ Y[i]
    K = np.zeros((NF + 1, NF + 1))
    K[:NF, :NF] = A
    K[:NF, NF] = 1.0
    K[NF, :NF] = 1.0
    rhs = np.zeros(NF + 1)
    rhs[:NF] = b
    rhs[NF] = 1.0
    return np.linalg.solve(K, rhs)[:NF]


def build_or_load_cache() -> tuple[np.ndarray, np.ndarray]:
    cache_path = OUT / "task_frame_means_full250.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        print(f"loaded {cache_path} n={z['means'].shape[0]}", flush=True)
        return z["means"], z["Y"]

    OUT.mkdir(parents=True, exist_ok=True)
    means_list, Y_list = [], []
    total = 0
    for p in sorted(CACHE.glob("full_shard_*.npz")):
        z = np.load(p, mmap_mode="r")
        n = int(z["weights"].shape[0])
        for i in range(n):
            Wts = np.asarray(z["weights"][i], np.float32)  # one MLP only
            Y = np.asarray(z["all_layer_means"][i, -1], np.float64)
            seed = estimator_seed(int(z["mlp_seed"][i]))
            M = frame_means_from_wts(Wts, seed)
            means_list.append(M)
            Y_list.append(Y)
            if total % 25 == 0:
                mse = float(np.mean((M.mean(0) - Y) ** 2))
                print(total, "uni_mse", f"{mse:.6e}", flush=True)
            total += 1
        del z
    means = np.stack(means_list)
    Y = np.stack(Y_list)
    np.savez_compressed(cache_path, means=means, Y=Y)
    print(f"wrote {cache_path} n={total}", flush=True)
    return means, Y


def main() -> None:
    means, Y = build_or_load_cache()
    n = means.shape[0]
    idx = np.arange(n)
    folds = np.array_split(idx, 5)
    ridges = [1e-2, 1e0, 1e2, 1e4, 1e6]
    print("=== Full 5-fold ===", flush=True)
    best_r, best_ridge, best_w = 0.0, None, None
    for ridge in ridges:
        oos_u, oos_w = [], []
        for hold in folds:
            tr = np.setdiff1d(idx, hold)
            w = fit_shared_w(means, Y, ridge, tr)
            for i in hold:
                pu = means[i].mean(0)
                pw = w @ means[i]
                oos_u.append(float(np.mean((pu - Y[i]) ** 2)))
                oos_w.append(float(np.mean((pw - Y[i]) ** 2)))
        mu, mw = float(np.mean(oos_u)), float(np.mean(oos_w))
        r = mu / mw
        print(f"ridge={ridge:.0e} uni={mu:.6e} w={mw:.6e} r={r:.4f}", flush=True)
        if r > best_r:
            best_r, best_ridge = r, ridge
            best_w = fit_shared_w(means, Y, ridge, idx)

    print(f"best Full OOS r={best_r:.4f} ridge={best_ridge}", flush=True)
    print(
        "w min/max/std",
        float(best_w.min()),
        float(best_w.max()),
        float(best_w.std()),
        flush=True,
    )
    np.savez(OUT / "task_frame_w_v3.npz", w=best_w, ridge=best_ridge, full_oos_r=best_r)

    if best_r < 1.02:
        print("GATE FAIL Full OOS r<1.02 — skip Mini", flush=True)
        print("DONE", flush=True)
        return

    ds = Dataset.load_from_disk(str(MINI))
    errs_u, errs_w = [], []
    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        Ym = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF
        Wts = np.stack([np.asarray(w, np.float32) for w in mlp.weights], axis=0)
        M = frame_means_from_wts(Wts, seed)
        pu, pw = M.mean(0), best_w @ M
        errs_u.append(float(np.mean((pu - Ym) ** 2)))
        errs_w.append(float(np.mean((pw - Ym) ** 2)))
        if i % 20 == 0:
            print(i, float(np.mean(errs_u)), float(np.mean(errs_w)), flush=True)
    mu, mw = float(np.mean(errs_u)), float(np.mean(errs_w))
    print(
        f"Mini uni={mu:.6e} w={mw:.6e} r={mu/mw:.4f} "
        f"adj_u≈{mu*0.408:.6e} adj_w≈{mw*0.408:.6e}",
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

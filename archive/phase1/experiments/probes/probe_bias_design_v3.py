#!/usr/bin/env python3
"""protocol-3.0: global bias/scale; search K global Hadamard designs on Full→Mini."""
from __future__ import annotations

import importlib.util
import time
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
B = int(2.72e11)
W = 256
NF = 50
e_r = hada._chi_mean_radius(W)
H = np.asarray(hada._H256 * hada._INV_SQRT_256, np.float32)
scale = np.float32(e_r)


def make_frames(seed: int, nf: int = NF):
    rng = np.random.default_rng(int(seed))
    return [
        (rng.permutation(W), rng.choice([-1.0, 1.0], size=W).astype(np.float32))
        for _ in range(nf)
    ]


def forward_frames(weights, frames):
    blocks = []
    for perm, signs in frames:
        M = H[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks += [pos, -pos]
    x = fnp.asarray(np.concatenate(blocks, 0), dtype=fnp.float32)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def forward_base(weights, seed):
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(NF, W, seed=int(seed) & 0x7FFFFFFF, e_r=e_r)
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def estimator_seed(raw: int) -> int:
    ss = fnp.random.SeedSequence(int(raw)).spawn(3)
    return int(ss[2].generate_state(1)[0])


def main() -> None:
    ds = Dataset.load_from_disk(str(MINI))
    c = np.load(CACHE / "l30_feat_cache_v3.npz")
    g_bias = np.mean(c["Y31"] - c["B31"], 0)
    sc = float(np.sum(c["B31"] * c["Y31"]) / np.sum(c["B31"] ** 2))
    print(f"bias_rms={np.sqrt(np.mean(g_bias**2)):.4e} scale={sc:.6f}", flush=True)

    errs = {k: [] for k in ["base", "bias", "scale", "b+s"]}
    t0 = time.time()
    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        ws = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        pb = forward_base(ws, mlp.seed)
        for k, p in [
            ("base", pb),
            ("bias", pb + g_bias),
            ("scale", pb * sc),
            ("b+s", pb * sc + g_bias),
        ]:
            errs[k].append(float(np.mean((p - Y) ** 2)))
        if i % 25 == 0:
            print(
                "bias",
                i,
                float(np.mean(errs["base"])),
                float(np.mean(errs["bias"])),
                flush=True,
            )
    base = float(np.mean(errs["base"]))
    print("--- bias/scale ---", flush=True)
    print("base", base, "ref", 7.3833e-07, flush=True)
    for k in errs:
        m = float(np.mean(errs[k]))
        print(f"{k:8s} mse={m:.6e} r={base / m:.4f} adj≈{m * 0.408:.6e}", flush=True)

    shards = sorted(CACHE.glob("full_shard_*.npz"))
    Wa = np.concatenate([np.load(p)["weights"] for p in shards], 0)
    Ya = np.concatenate([np.load(p)["all_layer_means"][:, -1] for p in shards], 0)
    Sa = np.concatenate([np.load(p)["mlp_seed"] for p in shards], 0)
    n_fit = 40
    idx = np.random.default_rng(0).choice(len(Wa), n_fit, replace=False)
    print(f"--- design search Full n={n_fit} ---", flush=True)

    base_full = []
    for i in idx:
        ws = [fnp.asarray(Wa[i, l], dtype=fnp.float32) for l in range(32)]
        seed = estimator_seed(int(Sa[i]))
        base_full.append(float(np.mean((forward_base(ws, seed) - Ya[i]) ** 2)))
    base_full_m = float(np.mean(base_full))
    print(f"Full subset per-MLP base mse={base_full_m:.6e}", flush=True)

    K = 12
    best = None
    for k in range(K):
        fr = make_frames(10_000 + k * 997)
        mses = []
        for i in idx:
            ws = [fnp.asarray(Wa[i, l], dtype=fnp.float32) for l in range(32)]
            pred = forward_frames(ws, fr)
            mses.append(float(np.mean((pred - Ya[i]) ** 2)))
        m = float(np.mean(mses))
        print(f"  design {k} mse={m:.6e} r_vs_base={base_full_m / m:.4f}", flush=True)
        if best is None or m < best[0]:
            best = (m, k, fr)
    print(f"BEST design {best[1]} Full mse={best[0]:.6e}", flush=True)

    errs_g = []
    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        ws = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        pg = forward_frames(ws, best[2])
        errs_g.append(float(np.mean((pg - Y) ** 2)))
        if i % 25 == 0:
            print("mini-g", i, float(np.mean(errs_g)), flush=True)
    mg = float(np.mean(errs_g))
    print(f"Mini global_best mse={mg:.6e} r_vs_base={base / mg:.4f} adj≈{mg * 0.408:.6e}")
    print(f"Mini base         mse={base:.6e} adj≈{base * 0.408:.6e}")
    print(f"elapsed={time.time() - t0:.0f}s")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

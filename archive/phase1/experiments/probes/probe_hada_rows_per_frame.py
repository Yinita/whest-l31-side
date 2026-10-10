#!/usr/bin/env python3
"""Rows-per-frame sweep: same total samples, trade ONB completeness vs #frames.

Full ONB (R=256, nf=50) is pure-0.2.1s. Smaller R → more frames → more
independent column-sign draws at fixed N = nf * 2 * R.
Uses MLP.from_row(..., seed_protocol 3.0).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset
from whestbench.domain import MLP

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "probes" / "results"
OUT.mkdir(parents=True, exist_ok=True)
MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
B = int(2.72e11)
W = 256
# Target total forward samples ≈ baseline nf*2*W = 50*512 = 25600
_N_TARGET = 50 * 2 * W
_ROWS = [256, 128, 64, 32, 16]


def load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def directions_partial(hada, n_rows: int, n_frames: int, seed: int, e_r: float):
    H = np.asarray(hada._H256 * hada._INV_SQRT_256, np.float64)
    rng = np.random.default_rng(seed)
    scale = float(e_r)
    blocks = []
    for _ in range(n_frames):
        rows = rng.permutation(W)[:n_rows]
        signs = rng.choice([-1.0, 1.0], size=W)
        M = H[rows] * signs.reshape(1, -1)
        pos = (M * scale).astype(np.float32)
        blocks += [pos, -pos]
    return fnp.asarray(np.concatenate(blocks, 0), dtype=fnp.float32)


def main():
    hada = load_hada()
    ds = Dataset.load_from_disk(str(MINI))
    e_r = hada._chi_mean_radius(W)
    configs = []
    for r in _ROWS:
        nf = max(1, _N_TARGET // (2 * r))
        configs.append((f"R{r}_nf{nf}", r, nf))
    print("configs", configs, flush=True)

    errs = {name: [] for name, _, _ in configs}
    # Also exact baseline via production path
    errs["baseline021"] = []

    for i in range(len(ds)):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        weights = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF

        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(50, W, seed=seed, e_r=e_r)
            for w in weights:
                x = fnp.maximum(x @ w, 0.0)
            errs["baseline021"].append(
                float(np.mean((np.asarray(fnp.mean(x, 0), np.float64) - Y) ** 2))
            )

        for name, r, nf in configs:
            with flops.BudgetContext(flop_budget=B * 2, quiet=True):
                x = directions_partial(hada, r, nf, seed ^ (0x1000 * r), e_r)
                # Fairer: same seed for R=256 vs baseline; others offset by r
                if r == 256:
                    x = directions_partial(hada, r, nf, seed, e_r)
                for w in weights:
                    x = fnp.maximum(x @ w, 0.0)
                errs[name].append(
                    float(np.mean((np.asarray(fnp.mean(x, 0), np.float64) - Y) ** 2))
                )

        if i % 10 == 0:
            b = float(np.mean(errs["baseline021"]))
            means = {k: float(np.mean(v)) for k, v in errs.items() if v}
            best_k = min(means, key=means.get)
            print(
                f"[{i}] base={b:.4e} best={best_k}={means[best_k]:.4e} "
                f"r={b / means[best_k]:.4f}",
                flush=True,
            )
            (OUT / "hada_rows_partial.json").write_text(
                json.dumps({**means, "n": i + 1})
            )

    base = float(np.mean(errs["baseline021"]))
    ranking = sorted(
        ((float(np.mean(errs[k])), k) for k in errs),
        key=lambda t: t[0],
    )
    print(f"\n=== Mini n={len(ds)} base={base:.6e} ===", flush=True)
    out_rank = []
    for m, k in ranking:
        print(
            f"{k:16s} mse={m:.6e} ratio={base / m:.4f} proxy_adj={m * 0.408:.6e}",
            flush=True,
        )
        out_rank.append(
            {"kind": k, "mse": m, "ratio": base / m, "proxy_adj": m * 0.408}
        )
    (OUT / "hada_rows_final.json").write_text(
        json.dumps({"base": base, "n": len(ds), "ranking": out_rank}, indent=2)
    )


if __name__ == "__main__":
    main()

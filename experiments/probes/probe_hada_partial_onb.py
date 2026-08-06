#!/usr/bin/env python3
"""Partial-ONB frame size sweep: fewer rows/frame → more frames @ fixed sample count.

Full ONB (r=256) uses nf≈50 frames. With r rows/frame and antipodal, same N:
  n_frames = N / (2*r), more sign diversity, weaker per-frame orthogonality.

Uses MLP.from_row(..., seed_protocol_version=\"3.0\").
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
# Target total positive+antipodal samples matching pure-0.2.1s (nf=50, r=256).
N_TOTAL = 50 * 2 * W  # 25600


def load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def directions_partial(hada, n_rows: int, n_frames: int, seed: int, e_r: float):
    """n_frames of n_rows random Hadamard rows × col signs × antipodal."""
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
    # row sizes: full ONB + halves that divide N_TOTAL/(2r) cleanly
    row_sizes = [256, 128, 64, 32, 16, 8]
    kinds = [f"r{r}" for r in row_sizes]
    errs = {k: [] for k in kinds}
    utils_proxy = {}
    for r in row_sizes:
        nf = N_TOTAL // (2 * r)
        utils_proxy[f"r{r}"] = {"n_rows": r, "n_frames": nf, "n_samples": nf * 2 * r}

    print(f"N_TOTAL={N_TOTAL} designs={utils_proxy}", flush=True)

    for i in range(len(ds)):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        weights = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF

        for r in row_sizes:
            nf = N_TOTAL // (2 * r)
            with flops.BudgetContext(flop_budget=B * 2, quiet=True):
                if r == 256:
                    x = hada._hadamard_directions(nf, W, seed=seed, e_r=e_r)
                else:
                    x = directions_partial(hada, r, nf, seed=seed, e_r=e_r)
                for w in weights:
                    x = fnp.maximum(x @ w, 0.0)
                pred = np.asarray(fnp.mean(x, axis=0), np.float64)
            errs[f"r{r}"].append(float(np.mean((pred - Y) ** 2)))

        if i % 10 == 0:
            means = {k: float(np.mean(errs[k])) for k in kinds}
            best = min(means.items(), key=lambda kv: kv[1])
            print(
                f"[{i}] base=r256={means['r256']:.4e} best={best[0]}={best[1]:.4e} "
                f"r={means['r256'] / best[1]:.4f}",
                flush=True,
            )
            (OUT / "hada_partial_onb_partial.json").write_text(
                json.dumps({"n": i + 1, **means, "meta": utils_proxy})
            )

    base = float(np.mean(errs["r256"]))
    ranking = []
    print(f"\n=== Mini n={len(ds)} base(r256)={base:.6e} ===", flush=True)
    for m, k in sorted((float(np.mean(errs[k])), k) for k in kinds):
        line = (
            f"{k:8s} mse={m:.6e} ratio={base / m:.4f} proxy_adj={m * 0.408:.6e} "
            f"meta={utils_proxy[k]}"
        )
        print(line, flush=True)
        ranking.append(
            {
                "kind": k,
                "mse": m,
                "ratio": base / m,
                "proxy_adj": m * 0.408,
                **utils_proxy[k],
            }
        )
    (OUT / "hada_partial_onb_final.json").write_text(
        json.dumps({"base": base, "n": len(ds), "ranking": ranking}, indent=2)
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Within-frame sign balance / OA-lite vs iid signs (full ONB, protocol 3.0).

All designs: nf=50 full Hadamard frames × antipodal × E[χ], same N.
Only column-sign generation changes.
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
NF = 50


def load_hada():
    spec = importlib.util.spec_from_file_location(
        "hada", ROOT / "experiments" / "pure021s_hada_u40.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def signs_iid(rng: np.random.Generator) -> np.ndarray:
    return rng.choice([-1.0, 1.0], size=W)


def signs_balanced(rng: np.random.Generator) -> np.ndarray:
    s = np.ones(W, np.float64)
    s[rng.permutation(W)[: W // 2]] = -1.0
    return s


def signs_block4(rng: np.random.Generator) -> np.ndarray:
    """Partition coords into 4 blocks of 64; each block exactly balanced."""
    s = np.ones(W, np.float64)
    perm = rng.permutation(W)
    for b in range(4):
        idx = perm[b * 64 : (b + 1) * 64]
        s[idx[:32]] = -1.0
    return s


def signs_pair_anti(rng: np.random.Generator, frame_i: int, prev: np.ndarray | None):
    """Even frames: balanced; odd frames: flip half of previous (fresh half)."""
    if frame_i % 2 == 0 or prev is None:
        return signs_balanced(rng)
    s = prev.copy()
    flip = rng.permutation(W)[: W // 2]
    s[flip] *= -1.0
    return s


def forward_design(hada, weights, e_r, seed: int, kind: str) -> np.ndarray:
    H = np.asarray(hada._H256 * hada._INV_SQRT_256, np.float64)
    rng = np.random.default_rng(seed)
    scale = float(e_r)
    blocks = []
    prev = None
    for i in range(NF):
        # consume perm like production (matches baseline RNG shape when kind=baseline)
        perm = rng.permutation(W)
        if kind == "baseline":
            # production path: use fnp below instead
            pass
        if kind == "iid_np":
            signs = signs_iid(rng)
        elif kind == "bal_frame":
            signs = signs_balanced(rng)
        elif kind == "block4":
            signs = signs_block4(rng)
        elif kind == "pair_anti":
            signs = signs_pair_anti(rng, i, prev)
            prev = signs
        else:
            raise ValueError(kind)
        M = H[perm] * signs.reshape(1, -1)
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
    e_r = hada._chi_mean_radius(W)
    kinds = ["baseline", "iid_np", "bal_frame", "block4", "pair_anti"]
    errs = {k: [] for k in kinds}
    print(f"NF={NF} kinds={kinds}", flush=True)

    for i in range(len(ds)):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        weights = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF

        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
            for w in weights:
                x = fnp.maximum(x @ w, 0.0)
            errs["baseline"].append(
                float(np.mean((np.asarray(fnp.mean(x, 0), np.float64) - Y) ** 2))
            )

        for kind in kinds[1:]:
            pred = forward_design(hada, weights, e_r, seed, kind)
            errs[kind].append(float(np.mean((pred - Y) ** 2)))

        if i % 10 == 0:
            means = {k: float(np.mean(errs[k])) for k in kinds}
            best = min(means.items(), key=lambda kv: kv[1])
            print(
                f"[{i}] base={means['baseline']:.4e} best={best[0]}={best[1]:.4e} "
                f"r={means['baseline'] / best[1]:.4f}",
                flush=True,
            )
            (OUT / "hada_sign_balance_partial.json").write_text(
                json.dumps({"n": i + 1, **means})
            )

    base = float(np.mean(errs["baseline"]))
    ranking = []
    print(f"\n=== Mini n={len(ds)} base={base:.6e} ===", flush=True)
    for m, k in sorted((float(np.mean(errs[k])), k) for k in kinds):
        print(
            f"{k:12s} mse={m:.6e} ratio={base / m:.4f} proxy_adj={m * 0.408:.6e}",
            flush=True,
        )
        ranking.append({"kind": k, "mse": m, "ratio": base / m, "proxy_adj": m * 0.408})
    (OUT / "hada_sign_balance_final.json").write_text(
        json.dumps({"base": base, "n": len(ds), "ranking": ranking}, indent=2)
    )


if __name__ == "__main__":
    main()

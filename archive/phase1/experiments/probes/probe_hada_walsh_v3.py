"""Walsh-ordered / bit-reversed Hadamard frames vs Sylvester base (protocol 3.0).

Same util/NF/antipodal/E[χ]; only row ordering of the Hadamard matrix changes.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset
from whestbench.domain import MLP

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hada", ROOT / "experiments" / "pure021s_hada_u40.py")
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


def bitrev_indices(n: int) -> np.ndarray:
    bits = int(math.log2(n))
    idx = np.arange(n)
    rev = np.zeros(n, np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def make_H(order: str) -> np.ndarray:
    H = np.asarray(hada._H256 * hada._INV_SQRT_256, np.float32)
    if order == "sylvester":
        return H
    if order == "walsh":
        return H[bitrev_indices(W)]
    if order == "dyadic":
        # gray-code row order
        g = np.arange(W) ^ (np.arange(W) >> 1)
        return H[np.argsort(g)]
    raise ValueError(order)


def directions(H: np.ndarray, seed: int) -> fnp.ndarray:
    rng = fnp.random.default_rng(int(seed) & 0x7FFFFFFF)
    scale = fnp.float32(e_r)
    blocks = []
    Hf = fnp.asarray(H, dtype=fnp.float32)
    pm = fnp.asarray([-1.0, 1.0], dtype=fnp.float32)
    for _ in range(NF):
        perm = rng.permutation(W)
        signs = rng.choice(pm, size=W)
        M = Hf[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks.append(pos)
        blocks.append(-pos)
    return fnp.concatenate(blocks, axis=0)


def forward(weights, H, seed: int):
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = directions(H, seed)
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def main() -> None:
    ds = Dataset.load_from_disk(str(MINI))
    orders = ["sylvester", "walsh", "dyadic"]
    Hs = {o: make_H(o) for o in orders}
    # also: alternate sylvester/walsh per frame via mixing predictions 50/50
    # (same total FLOPs: half frames each) — handled as separate method below
    errs = {o: [] for o in orders}
    errs["mix_sw"] = []
    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        ws = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF
        preds = {}
        for o in orders:
            preds[o] = forward(ws, Hs[o], seed)
            errs[o].append(float(np.mean((preds[o] - Y) ** 2)))
        # mix: 25 sylvester frames + 25 walsh (independent RNG streams)
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            # reuse half-budget by manually building
            rng = fnp.random.default_rng(seed)
            scale = fnp.float32(e_r)
            pm = fnp.asarray([-1.0, 1.0], dtype=fnp.float32)
            blocks = []
            for oi, H in enumerate([Hs["sylvester"], Hs["walsh"]]):
                Hf = fnp.asarray(H, dtype=fnp.float32)
                for _ in range(NF // 2):
                    perm = rng.permutation(W)
                    signs = rng.choice(pm, size=W)
                    M = Hf[perm] * signs.reshape(1, -1)
                    pos = M * scale
                    blocks.append(pos)
                    blocks.append(-pos)
            x = fnp.concatenate(blocks, axis=0)
            for w in ws:
                x = fnp.maximum(x @ w, 0.0)
            pmix = np.asarray(fnp.mean(x, axis=0), np.float64)
        errs["mix_sw"].append(float(np.mean((pmix - Y) ** 2)))
        if i % 20 == 0:
            print(i, {k: float(np.mean(errs[k])) for k in errs}, flush=True)

    base = float(np.mean(errs["sylvester"]))
    print("base(sylvester)", base, "ref", 7.3833e-07)
    for k in errs:
        m = float(np.mean(errs[k]))
        print(f"{k:12s} mse={m:.6e} r={base / m:.4f} proxy_adj={m * 0.408:.6e}")
    print("DONE")


if __name__ == "__main__":
    main()

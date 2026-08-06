"""LD signs vs official fnp Hadamard base (seed_protocol 3.0)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import Dataset
from whestbench.domain import MLP

spec = importlib.util.spec_from_file_location("hada", "experiments/pure021s_hada_u40.py")
hada = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hada)

MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/"
    "snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
ds = Dataset.load_from_disk(str(MINI))
B = int(2.72e11)
W = 256
NF = 50
e_r = hada._chi_mean_radius(W)
Hnp = np.asarray(hada._H256 * hada._INV_SQRT_256, np.float32)
scale = np.float32(e_r)
_PM = fnp.asarray([-1.0, 1.0], dtype=fnp.float32)


def primes(n: int) -> list[int]:
    out: list[int] = []
    x = 2
    while len(out) < n:
        if all(x % p for p in out):
            out.append(x)
        x += 1
    return out


BASES = primes(W)


def vdc(n: int, base: int) -> float:
    v = 0.0
    f = 1.0 / base
    k = n
    while k:
        v += (k % base) * f
        k //= base
        f /= base
    return v


def forward_np_blocks(weights, blocks):
    x = fnp.asarray(np.concatenate(blocks, 0), dtype=fnp.float32)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        for w in weights:
            x = fnp.maximum(x @ w, 0.0)
        return np.asarray(fnp.mean(x, axis=0), np.float64)


def design_weyl(seed: int, sync_consume: bool):
    rng = fnp.random.default_rng(int(seed) & 0x7FFFFFFF)
    a = np.sqrt(np.arange(1, W + 1, dtype=np.float64) * 0.5 + np.pi)
    a = a - np.floor(a)
    u0 = (int(seed) & 0x7FFFFFFF) / 2**31
    blocks = []
    for i in range(NF):
        perm = np.asarray(rng.permutation(W))
        if sync_consume:
            _ = rng.choice(_PM, size=W)
        signs = np.where(((i * a + u0) % 1.0) >= 0.5, 1.0, -1.0).astype(np.float32)
        M = Hnp[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks += [pos, -pos]
    return blocks


def design_halton_sync(seed: int):
    rng = fnp.random.default_rng(int(seed) & 0x7FFFFFFF)
    u0 = (int(seed) & 0x7FFFFFFF) / 2**31
    blocks = []
    for i in range(NF):
        perm = np.asarray(rng.permutation(W))
        _ = rng.choice(_PM, size=W)
        h = np.array([vdc(i + 1 + (int(seed) % 997), BASES[j]) for j in range(W)])
        h = (h + u0) % 1.0
        signs = np.where(h >= 0.5, 1.0, -1.0).astype(np.float32)
        M = Hnp[perm] * signs.reshape(1, -1)
        pos = M * scale
        blocks += [pos, -pos]
    return blocks


def main() -> None:
    errs = {k: [] for k in ["base", "weyl", "weyl_sync", "halton_sync"]}
    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        ws = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF
        with flops.BudgetContext(flop_budget=B * 2, quiet=True):
            x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
            for w in ws:
                x = fnp.maximum(x @ w, 0.0)
            pb = np.asarray(fnp.mean(x, axis=0), np.float64)
        errs["base"].append(float(np.mean((pb - Y) ** 2)))
        for k, blocks in [
            ("weyl", design_weyl(mlp.seed, False)),
            ("weyl_sync", design_weyl(mlp.seed, True)),
            ("halton_sync", design_halton_sync(mlp.seed)),
        ]:
            pred = forward_np_blocks(ws, blocks)
            errs[k].append(float(np.mean((pred - Y) ** 2)))
        if i % 20 == 0:
            print(i, {k: float(np.mean(errs[k])) for k in errs}, flush=True)

    base = float(np.mean(errs["base"]))
    print("base", base, "ref", 7.3833e-07, "relerr", abs(base - 7.3833e-07) / 7.3833e-07)
    for k in errs:
        m = float(np.mean(errs[k]))
        print(f"{k:12s} mse={m:.6e} r={base / m:.4f} proxy_adj={m * 0.408:.6e}")
    print("DONE")


if __name__ == "__main__":
    main()

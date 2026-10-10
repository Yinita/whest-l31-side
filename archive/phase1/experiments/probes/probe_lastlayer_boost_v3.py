#!/usr/bin/env python3
"""Budget shift via frame truncation + last-layer extras (one full forward).

One NF=50 Hadamard path; evaluate:
  base: mean of all frames
  fK + boot/gauss: mean of first K frames' L31, plus N_extra last-layer
    samples from L30 cloud of those K frames (FLOP proxy-matched).
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
B = int(2.72e11)
W = 256
D = 32
NF = 50
SPS = 2 * W
e_r = hada._chi_mean_radius(W)


def run_path(weights, seed: int):
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        x = hada._hadamard_directions(NF, W, seed=seed, e_r=e_r)
        a30 = None
        for li, w in enumerate(weights):
            x = fnp.maximum(x @ w, 0.0)
            if li == D - 2:  # after weight[30] → L30
                a30 = np.asarray(x, np.float64)
        last = np.asarray(x, np.float64)
    return last.reshape(NF, SPS, W), a30.reshape(NF, SPS, W)


def last_only_mean(a30_flat, w31, n_extra, mode, rng):
    if n_extra <= 0:
        return np.zeros(W)
    if mode == "bootstrap":
        idx = rng.integers(0, a30_flat.shape[0], size=n_extra)
        z = a30_flat[idx]
    else:
        mu = a30_flat.mean(0)
        sig = np.sqrt(np.maximum(a30_flat.var(0, ddof=1), 1e-18))
        z = np.maximum(rng.normal(mu, sig, size=(n_extra, W)), 0.0)
    with flops.BudgetContext(flop_budget=B * 2, quiet=True):
        y = fnp.maximum(fnp.asarray(z, dtype=fnp.float32) @ w31, 0.0)
        return np.asarray(fnp.mean(y, axis=0), np.float64)


def main() -> None:
    ds = Dataset.load_from_disk(str(MINI))
    # name -> (n_full_frames, mode or None, blend)
    # blend None => count-weighted; float => α*full + (1-α)*extra
    specs = [
        ("base", 50, None, None),
        ("f45_boot", 45, "bootstrap", None),
        ("f40_boot", 40, "bootstrap", None),
        ("f40_gauss", 40, "diag", None),
        ("f35_boot", 35, "bootstrap", None),
        ("f40_b08", 40, "bootstrap", 0.85),
    ]
    errs = {s[0]: [] for s in specs}

    for i in range(100):
        ex = ds[i]
        mlp = MLP.from_row(ex, seed_protocol_version="3.0")
        ws = list(mlp.weights)
        Y = np.asarray(ex["final_means"], np.float64)
        seed = int(mlp.seed) & 0x7FFFFFFF
        last_f, a30_f = run_path(ws, seed)
        rng = np.random.default_rng(seed ^ 0x1234)

        for name, nf, mode, blend in specs:
            mu_full = last_f[:nf].reshape(-1, W).mean(0)
            if mode is None:
                pred = mu_full
            else:
                n_extra = int((NF - nf) * D * SPS)
                a30_use = a30_f[:nf].reshape(-1, W)
                mu_ex = last_only_mean(a30_use, ws[-1], n_extra, mode, rng)
                if blend is None:
                    n_f = nf * SPS
                    pred = (n_f * mu_full + n_extra * mu_ex) / (n_f + n_extra)
                else:
                    pred = blend * mu_full + (1.0 - blend) * mu_ex
            errs[name].append(float(np.mean((pred - Y) ** 2)))

        if i % 20 == 0:
            print(i, {k: float(np.mean(errs[k])) for k in errs}, flush=True)

    base = float(np.mean(errs["base"]))
    print("base", base, "ref", 7.3833e-07, flush=True)
    for name, *_ in specs:
        m = float(np.mean(errs[name]))
        print(f"{name:12s} mse={m:.6e} r={base / m:.4f} adj≈{m * 0.408:.6e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

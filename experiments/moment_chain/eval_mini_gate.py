#!/usr/bin/env python3
"""Mini gate: moment-chain (baked / DAgger) vs Hadamard pure-0.2.1s raw MSE.

Compares final-layer MSE only (no FLOPs adjustment). Chain is expected ~5e-6;
Hadamard with large N is ~3e-7. Documents the gap; does not ship unless ratio>1.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments"))

from portable import load_coefs_b64, probe_cumulants, run_chain  # noqa: E402

MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)


def _sylvester_hadamard(dim: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < dim:
        h = np.block([[h, h], [h, -h]])
    return h.astype(np.float64)


def hadamard_final_mse(
    weights: list[np.ndarray],
    target: np.ndarray,
    n_samples: int,
    seed: int,
) -> float:
    """Antipodal Hadamard + E[χ] radius — numpy offline (no flopscope)."""
    width = weights[0].shape[0]
    depth = len(weights)
    H = _sylvester_hadamard(width) / math.sqrt(width)
    half = 0.5 * float(width)
    e_r = math.sqrt(2.0) * math.exp(math.lgamma(half + 0.5) - math.lgamma(half))
    n_frames = max(1, n_samples // (2 * width))
    n_frames = min(n_frames, 200)
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_frames):
        perm = rng.permutation(width)
        signs = rng.choice([-1.0, 1.0], size=width)
        M = H[perm] * signs.reshape(1, -1) * e_r
        blocks.append(M)
        blocks.append(-M)
    x = np.concatenate(blocks, axis=0)
    for W in weights:
        x = np.maximum(x @ W, 0.0)
    pred = x.mean(0)
    return float(np.mean((pred - target) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--hada-n", type=int, default=25600, help="~50 frames × 512")
    ap.add_argument(
        "--coefs",
        type=str,
        default=str(Path(__file__).resolve().parent / "port_coefs.b64.txt"),
    )
    ap.add_argument("--mean-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = Dataset.load_from_disk(str(MINI))
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    coefs = load_coefs_b64(Path(args.coefs)) if args.coefs else None

    chain_mses, hada_mses, unc_mses = [], [], []
    t0 = time.time()
    for k, i in enumerate(idxs):
        ex = ds[int(i)]
        W = [np.asarray(ex["weights"][l], dtype=np.float64) for l in range(32)]
        y = np.asarray(ex["final_means"], dtype=np.float64)
        pr = probe_cumulants(W, args.n_probe, seed=10_000 + int(i))
        unc = run_chain(W, pr, coefs=None)
        ch = run_chain(W, pr, coefs, mean_only=args.mean_only)
        hd = hadamard_final_mse(W, y, args.hada_n, seed=int(ex["mlp_seed"]))
        unc_mses.append(float(np.mean((unc[-1] - y) ** 2)))
        chain_mses.append(float(np.mean((ch[-1] - y) ** 2)))
        hada_mses.append(hd)
        if (k + 1) % 5 == 0:
            print(
                f"[{k+1}/{len(idxs)}] chain={np.mean(chain_mses):.4e} "
                f"hada={np.mean(hada_mses):.4e} unc={np.mean(unc_mses):.4e} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )

    out = {
        "n": len(idxs),
        "coefs": args.coefs,
        "mean_only": args.mean_only,
        "hada_n": args.hada_n,
        "n_probe": args.n_probe,
        "uncorrected_mse": float(np.mean(unc_mses)),
        "chain_mse": float(np.mean(chain_mses)),
        "hada_mse": float(np.mean(hada_mses)),
        "chain_vs_hada": float(np.mean(hada_mses) / np.mean(chain_mses)),
        "chain_vs_unc": float(np.mean(unc_mses) / np.mean(chain_mses)),
    }
    print(json.dumps(out, indent=2))
    art = Path(__file__).resolve().parent / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "eval_mini_gate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

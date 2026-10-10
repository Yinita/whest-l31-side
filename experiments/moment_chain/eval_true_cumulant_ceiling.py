#!/usr/bin/env python3
"""Probe-noise ceiling: atlas connected pre-cumulants vs MC probe N=4096.

Atlas stores raw moments; convert to central/connected forms matching the
MC probe (k3, k4=central4-3var², kiij=E[zc_i² zc_j]).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable import DEPTH, load_coefs_b64, probe_cumulants, run_chain  # noqa: E402
from train_dagger_atlas import load_shard_weights  # noqa: E402

ATLAS_REPO = "keenanpepper/arc-whestbench-higher-moments-2026"
OUT_DIR = Path(__file__).resolve().parent / "artifacts"


def connected_pre_probe(z) -> list[tuple]:
    probes = []
    for l in range(DEPTH):
        mu = z["pre_mean"][l].astype(np.float64)
        M11 = z["pre_M11"][l].astype(np.float64)
        M21 = z["pre_M21"][l].astype(np.float64)
        m2 = z["pre_m2"][l].astype(np.float64)
        m3 = z["pre_m3"][l].astype(np.float64)
        m4 = z["pre_m4"][l].astype(np.float64)
        k3 = m3 - 3.0 * mu * m2 + 2.0 * mu**3
        c4 = m4 - 4.0 * mu * m3 + 6.0 * mu**2 * m2 - 3.0 * mu**4
        var = np.maximum(m2 - mu**2, 0.0)
        k4 = c4 - 3.0 * var**2
        kiij = (
            M21
            - 2.0 * (mu[:, None] * M11)
            - np.outer(m2, mu)
            + 2.0 * np.outer(mu**2, mu)
        )
        probes.append((k3, k4, kiij))
    return probes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=24)
    ap.add_argument("--val-n", type=int, default=6)
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument(
        "--coefs",
        type=str,
        default=str(Path(__file__).parent / "port_coefs.b64.txt"),
    )
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n = args.train_n + args.val_n
    weights_list, ids = load_shard_weights(n)
    coefs = load_coefs_b64(Path(args.coefs))
    val_idxs = list(range(args.train_n, n))
    t0 = time.time()

    mc_mses, true_mses = [], []
    for i in val_idxs:
        mid = ids[i]
        W = [weights_list[i][l] for l in range(DEPTH)]
        path = hf_hub_download(
            ATLAS_REPO,
            f"full/mlp_{mid:05d}.npz",
            repo_type="dataset",
            local_files_only=True,
        )
        z = np.load(path)
        y = z["mean"][-1].astype(np.float64)
        mc_pr = probe_cumulants(W, args.n_probe, seed=1000 + mid)
        tr_pr = connected_pre_probe(z)
        mc = run_chain(W, mc_pr, coefs)
        tr = run_chain(W, tr_pr, coefs)
        mc_mses.append(float(np.mean((mc[-1] - y) ** 2)))
        true_mses.append(float(np.mean((tr[-1] - y) ** 2)))
        print(f"val {mid}: mc={mc_mses[-1]:.4e} true={true_mses[-1]:.4e}", flush=True)

    out = {
        "n_val": len(val_idxs),
        "coefs": args.coefs,
        "mc_probe_mse": float(np.mean(mc_mses)),
        "true_cumulant_mse": float(np.mean(true_mses)),
        "ratio_mc_vs_true": float(np.mean(mc_mses) / np.mean(true_mses)),
        "elapsed_s": time.time() - t0,
    }
    print(json.dumps(out, indent=2))
    (OUT_DIR / "true_cumulant_ceiling.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

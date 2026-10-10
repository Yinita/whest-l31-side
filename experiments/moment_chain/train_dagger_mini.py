#!/usr/bin/env python3
"""DAgger-style stabilizer fit on Mini all_layer_means.

Fits per-layer mean (+ optional MC-teacher var/offdiag) correctors on the
chain's own rolled-forward states. Mini has no M11 atlas, so default is
mean-only; pass --mc-cov-n N to estimate post-ReLU cov via MC teacher.

Usage:
  .venv/bin/python experiments/moment_chain/train_dagger_mini.py
  .venv/bin/python experiments/moment_chain/train_dagger_mini.py --mc-cov-n 8192
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from portable import (  # noqa: E402
    DEPTH,
    WIDTH,
    load_coefs_b64,
    mean_features,
    probe_cumulants,
    run_chain,
    save_coefs_b64,
    step_np,
    tcross_feats,
    var_features,
)

MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
OUT_DIR = Path(__file__).resolve().parent / "artifacts"


def _mc_layer_moments(
    weights: list[np.ndarray], n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher: post-ReLU mean and second-moment M11 per layer via MC."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, WIDTH))
    means = np.zeros((DEPTH, WIDTH))
    M11s = np.zeros((DEPTH, WIDTH, WIDTH))
    for l, W in enumerate(weights):
        z = x @ W
        h = np.maximum(z, 0.0)
        means[l] = h.mean(0)
        M11s[l] = (h.T @ h) / n
        x = h
    return means, M11s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=80)
    ap.add_argument("--val-n", type=int, default=20)
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--mc-cov-n", type=int, default=0, help="0=mean-only DAgger")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = Dataset.load_from_disk(str(MINI))
    n_total = min(len(ds), args.train_n + args.val_n)
    train_n = min(args.train_n, n_total - 1)
    val_n = n_total - train_n
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ds))[:n_total]
    train_idx = order[:train_n]
    val_idx = order[train_n:]

    print(f"load Mini: train={train_n} val={val_n} probe={args.n_probe} mc_cov={args.mc_cov_n}")
    t0 = time.time()

    weights_all: list[list[np.ndarray]] = []
    alm_all: list[np.ndarray] = []
    probes: list = []
    teachers: list[tuple[np.ndarray, np.ndarray] | None] = []

    for i, mi in enumerate(order):
        ex = ds[int(mi)]
        W = [np.asarray(ex["weights"][l], dtype=np.float64) for l in range(DEPTH)]
        alm = np.asarray(ex["all_layer_means"], dtype=np.float64)
        weights_all.append(W)
        alm_all.append(alm)
        probes.append(probe_cumulants(W, args.n_probe, seed=10_000 + int(mi)))
        if args.mc_cov_n > 0:
            teachers.append(_mc_layer_moments(W, args.mc_cov_n, seed=20_000 + int(mi)))
        else:
            teachers.append(None)
        if (i + 1) % 20 == 0:
            print(f"  loaded {i+1}/{n_total} ({time.time()-t0:.0f}s)", flush=True)

    # Uncorrected baseline on val
    base_mses = []
    for j, vi in enumerate(val_idx):
        # map vi (dataset index) -> position in order
        pos = int(np.where(order == vi)[0][0])
        est = run_chain(weights_all[pos], probes[pos], coefs=None)
        base_mses.append(float(np.mean((est[-1] - alm_all[pos][-1]) ** 2)))
    print(f"uncorrected val final mse={np.mean(base_mses):.4e}", flush=True)

    # Baked #314695 coefs on val (distribution shift check)
    baked_path = Path(__file__).resolve().parent / "port_coefs.b64.txt"
    if baked_path.exists():
        baked = load_coefs_b64(baked_path)
        baked_mses = []
        for vi in val_idx:
            pos = int(np.where(order == vi)[0][0])
            est = run_chain(weights_all[pos], probes[pos], baked, mean_only=False)
            baked_mses.append(float(np.mean((est[-1] - alm_all[pos][-1]) ** 2)))
        print(f"baked#314695 val final mse={np.mean(baked_mses):.4e}", flush=True)

    # Sequential DAgger fit on train positions 0..train_n-1
    train_pos = list(range(train_n))
    states = [(np.zeros(WIDTH), np.eye(WIDTH)) for _ in train_pos]
    coefs: list[tuple] = []
    mean_only = args.mc_cov_n <= 0

    for l in range(DEPTH):
        Xm, ym = [], []
        Xv, yv = [], []
        Xo, yo = [], []
        cache = []
        iu = np.triu_indices(WIDTH, k=1)
        for t in train_pos:
            W = weights_all[t][l]
            m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(*states[t], W)
            k3, k4, kiij = probes[t][l]
            Fm = mean_features(a, sig_pre, k3, k4)
            cache.append((m_hat, S_hat, Fm, a, sig_pre, Spre, mu_pre, kiij, k3, k4))
            tm = alm_all[t][l]
            Xm.append(Fm)
            ym.append(tm - m_hat)
            if not mean_only and teachers[t] is not None:
                t_mean, t_M11 = teachers[t]
                # Prefer atlas-quality means from Mini; cov from MC teacher
                tS = t_M11[l] - np.outer(t_mean[l], t_mean[l])
                # Align cov teacher mean to official alm for residual consistency
                tS = t_M11[l] - np.outer(tm, tm)  # use official mean in centering
                Fv = var_features(a, sig_pre, k3, k4)
                Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
                Xv.append(Fv)
                yv.append(np.diag(tS) - np.diag(S_hat))
                Xo.append(np.stack([f[iu] for f in Fo], axis=1))
                yo.append((tS - S_hat)[iu])

        Am, *_ = np.linalg.lstsq(np.concatenate(Xm), np.concatenate(ym), rcond=None)
        if mean_only:
            Av = np.zeros(6)
            Ao = np.zeros(2)
        else:
            Av, *_ = np.linalg.lstsq(np.concatenate(Xv), np.concatenate(yv), rcond=None)
            Ao, *_ = np.linalg.lstsq(np.concatenate(Xo), np.concatenate(yo), rcond=None)
        coefs.append((Am, Av, Ao))

        for ti, t in enumerate(train_pos):
            m_hat, S_hat, Fm, a, sig_pre, Spre, mu_pre, kiij, k3, k4 = cache[ti]
            m = m_hat + Fm @ Am
            if mean_only:
                S = S_hat
            else:
                Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
                S = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
                S[np.diag_indices(WIDTH)] = np.diag(S_hat) + var_features(
                    a, sig_pre, k3, k4
                ) @ Av
            states[t] = (m, S)

        if l in (0, 7, 15, 23, 31):
            r = np.mean(
                [
                    np.mean((states[t][0] - alm_all[t][l]) ** 2)
                    for t in train_pos
                ]
            )
            print(f"layer {l+1:02d}: train mean-mse {r:.4e} ({time.time()-t0:.0f}s)", flush=True)

    # Val with refit coefs
    val_mses = []
    for vi in val_idx:
        pos = int(np.where(order == vi)[0][0])
        est = run_chain(weights_all[pos], probes[pos], coefs, mean_only=mean_only)
        val_mses.append(float(np.mean((est[-1] - alm_all[pos][-1]) ** 2)))
    val_mse = float(np.mean(val_mses))
    print(f"\n=== DAgger {'mean-only' if mean_only else 'full'} val final mse={val_mse:.4e} ===")
    print(f"uncorrected baseline={np.mean(base_mses):.4e}  ratio={np.mean(base_mses)/val_mse:.3f}")

    tag = "meanonly" if mean_only else f"mccov{args.mc_cov_n}"
    coef_path = OUT_DIR / f"dagger_mini_{tag}_coefs.b64.txt"
    save_coefs_b64(coefs, coef_path)
    meta = {
        "val_mse": val_mse,
        "uncorrected_val_mse": float(np.mean(base_mses)),
        "train_n": train_n,
        "val_n": val_n,
        "n_probe": args.n_probe,
        "mc_cov_n": args.mc_cov_n,
        "mean_only": mean_only,
        "train_idx": [int(x) for x in train_idx],
        "val_idx": [int(x) for x in val_idx],
        "coef_path": str(coef_path),
    }
    meta_path = OUT_DIR / f"dagger_mini_{tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {coef_path}\nwrote {meta_path}\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

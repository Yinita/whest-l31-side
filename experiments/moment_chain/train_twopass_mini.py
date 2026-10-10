#!/usr/bin/env python3
"""Two-pass residual refit on Mini with MC cov teacher (no atlas).

Quick method check while atlas downloads. Pass-1 sequential DAgger;
pass-2 additive residual on frozen rollout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from datasets import Dataset

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
from train_dagger_mini import _mc_layer_moments  # noqa: E402

MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
OUT_DIR = Path(__file__).resolve().parent / "artifacts"


def apply_corr(m_hat, S_hat, Fm, Fv, Fo, Am, Av, Ao):
    w = m_hat.shape[0]
    m = m_hat + Fm @ Am
    S = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
    S = np.array(S, copy=True)
    S[np.diag_indices(w)] = np.diag(S_hat) + Fv @ Av
    return m, S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=40)
    ap.add_argument("--val-n", type=int, default=10)
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--mc-cov-n", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = Dataset.load_from_disk(str(MINI))
    n = args.train_n + args.val_n
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ds))[:n]
    t0 = time.time()
    print(f"load Mini train={args.train_n} val={args.val_n} mc={args.mc_cov_n}", flush=True)

    weights, alms, probes, teachers = [], [], [], []
    for i, mi in enumerate(order):
        ex = ds[int(mi)]
        W = [np.asarray(ex["weights"][l], dtype=np.float64) for l in range(DEPTH)]
        weights.append(W)
        alms.append(np.asarray(ex["all_layer_means"], dtype=np.float64))
        probes.append(probe_cumulants(W, args.n_probe, seed=30_000 + int(mi)))
        teachers.append(_mc_layer_moments(W, args.mc_cov_n, seed=40_000 + int(mi)))
        if (i + 1) % 10 == 0:
            print(f"  loaded {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    train_n = args.train_n
    val_pos = list(range(train_n, n))
    iu = np.triu_indices(WIDTH, k=1)

    def eval_c(coefs):
        ms = []
        for p in val_pos:
            est = run_chain(weights[p], probes[p], coefs, mean_only=False)
            ms.append(float(np.mean((est[-1] - alms[p][-1]) ** 2)))
        return float(np.mean(ms))

    print(f"uncorrected={eval_c(None):.4e}", flush=True)
    baked = load_coefs_b64(Path(__file__).parent / "port_coefs.b64.txt")
    print(f"baked={eval_c(baked):.4e}", flush=True)

    # Pass-1
    states = [(np.zeros(WIDTH), np.eye(WIDTH)) for _ in range(train_n)]
    coefs1 = []
    for l in range(DEPTH):
        Xm, ym, Xv, yv, Xo, yo, cache = [], [], [], [], [], [], []
        for t in range(train_n):
            m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(*states[t], weights[t][l])
            k3, k4, kiij = probes[t][l]
            Fm = mean_features(a, sig_pre, k3, k4)
            Fv = var_features(a, sig_pre, k3, k4)
            Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
            cache.append((m_hat, S_hat, Fm, Fv, Fo))
            tm = alms[t][l]
            t_mean, t_M11 = teachers[t]
            tS = t_M11[l] - np.outer(tm, tm)
            Xm.append(Fm); ym.append(tm - m_hat)
            Xv.append(Fv); yv.append(np.diag(tS) - np.diag(S_hat))
            Xo.append(np.stack([f[iu] for f in Fo], 1)); yo.append((tS - S_hat)[iu])
        Am, *_ = np.linalg.lstsq(np.concatenate(Xm), np.concatenate(ym), rcond=None)
        Av, *_ = np.linalg.lstsq(np.concatenate(Xv), np.concatenate(yv), rcond=None)
        Ao, *_ = np.linalg.lstsq(np.concatenate(Xo), np.concatenate(yo), rcond=None)
        coefs1.append((Am, Av, Ao))
        for t in range(train_n):
            states[t] = apply_corr(*cache[t], Am, Av, Ao)
        if l in (0, 15, 31):
            r = np.mean([np.mean((states[t][0] - alms[t][l]) ** 2) for t in range(train_n)])
            print(f"pass1 L{l+1:02d} train {r:.4e}", flush=True)

    p1 = eval_c(coefs1)
    print(f"pass-1 val={p1:.4e}", flush=True)

    # Pass-2 residual
    Xm = [[] for _ in range(DEPTH)]
    ym = [[] for _ in range(DEPTH)]
    Xv = [[] for _ in range(DEPTH)]
    yv = [[] for _ in range(DEPTH)]
    Xo = [[] for _ in range(DEPTH)]
    yo = [[] for _ in range(DEPTH)]
    for t in range(train_n):
        m, S = np.zeros(WIDTH), np.eye(WIDTH)
        for l in range(DEPTH):
            m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(m, S, weights[t][l])
            k3, k4, kiij = probes[t][l]
            Am, Av, Ao = coefs1[l]
            Fm = mean_features(a, sig_pre, k3, k4)
            Fv = var_features(a, sig_pre, k3, k4)
            Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
            m1, S1 = apply_corr(m_hat, S_hat, Fm, Fv, Fo, Am, Av, Ao)
            tm = alms[t][l]
            tS = teachers[t][1][l] - np.outer(tm, tm)
            Xm[l].append(Fm); ym[l].append(tm - m1)
            Xv[l].append(Fv); yv[l].append(np.diag(tS) - np.diag(S1))
            Xo[l].append(np.stack([f[iu] for f in Fo], 1)); yo[l].append((tS - S1)[iu])
            m, S = m1, S1
    coefs2 = []
    for l in range(DEPTH):
        dAm, *_ = np.linalg.lstsq(np.concatenate(Xm[l]), np.concatenate(ym[l]), rcond=None)
        dAv, *_ = np.linalg.lstsq(np.concatenate(Xv[l]), np.concatenate(yv[l]), rcond=None)
        dAo, *_ = np.linalg.lstsq(np.concatenate(Xo[l]), np.concatenate(yo[l]), rcond=None)
        Am, Av, Ao = coefs1[l]
        coefs2.append((Am + dAm, Av + dAv, Ao + dAo))
    p2 = eval_c(coefs2)
    print(f"pass-2 val={p2:.4e}", flush=True)

    save_coefs_b64(coefs1, OUT_DIR / "dagger_mini_twopass_p1.b64.txt")
    save_coefs_b64(coefs2, OUT_DIR / "dagger_mini_twopass_p2.b64.txt")
    meta = {
        "pass1_val_mse": p1,
        "pass2_val_mse": p2,
        "ratio_p2_vs_p1": p1 / p2 if p2 > 0 else None,
        "train_n": train_n,
        "val_n": args.val_n,
        "mc_cov_n": args.mc_cov_n,
        "elapsed_s": time.time() - t0,
    }
    (OUT_DIR / "dagger_mini_twopass_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

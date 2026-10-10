#!/usr/bin/env python3
"""Atlas DAgger + two-pass residual refit (true M11 from keenanpepper atlas).

Uses local full_cache shards for weights and HF higher-moments atlas for
(mean, M11) teachers. Pass-1 = sequential DAgger (chain_port_refit). Pass-2 =
freeze pass-1 coefs, re-roll, fit additive residual heads on that trajectory.

Usage:
  .venv/bin/python experiments/moment_chain/train_dagger_atlas.py
  .venv/bin/python experiments/moment_chain/train_dagger_atlas.py --train-n 40 --val-n 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

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

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "experiments" / "results" / "full_cache"
OUT_DIR = Path(__file__).resolve().parent / "artifacts"
ATLAS_REPO = "keenanpepper/arc-whestbench-higher-moments-2026"


def load_shard_weights(n: int) -> tuple[list[np.ndarray], list[int]]:
    """Load first n nets from full_shard_*.npz (mlp_id order)."""
    weights: list[np.ndarray] = []
    ids: list[int] = []
    for si in range(5):
        p = CACHE / f"full_shard_{si:03d}.npz"
        z = np.load(p)
        for i in range(z["weights"].shape[0]):
            weights.append(np.asarray(z["weights"][i], dtype=np.float64))
            ids.append(int(z["mlp_id"][i]))
            if len(weights) >= n:
                return weights, ids
    return weights, ids


def load_atlas(mlp_id: int) -> dict:
    path = hf_hub_download(
        ATLAS_REPO, f"full/mlp_{mlp_id:05d}.npz", repo_type="dataset"
    )
    z = np.load(path)
    mean = z["mean"].astype(np.float64)
    M11 = z["M11"].astype(np.float64)
    return {"mean": mean, "M11": M11, "official_alm": z["official_alm"].astype(np.float64)}


def apply_correction(m_hat, S_hat, Fm, Fv, Fo, Am, Av, Ao):
    w = m_hat.shape[0]
    m = m_hat + Fm @ Am
    S = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
    S = S.copy()
    S[np.diag_indices(w)] = np.diag(S_hat) + Fv @ Av
    return m, S


def sequential_dagger(
    weights_list, atlases, probes, train_n: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pass-1: sequential DAgger fit on train nets 0..train_n-1."""
    states = [(np.zeros(WIDTH), np.eye(WIDTH)) for _ in range(train_n)]
    coefs = []
    iu = np.triu_indices(WIDTH, k=1)
    t0 = time.time()
    for l in range(DEPTH):
        Xm, ym, Xv, yv, Xo, yo, cache = [], [], [], [], [], [], []
        for t in range(train_n):
            W = weights_list[t][l]
            m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(*states[t], W)
            k3, k4, kiij = probes[t][l]
            Fm = mean_features(a, sig_pre, k3, k4)
            Fv = var_features(a, sig_pre, k3, k4)
            Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
            cache.append((m_hat, S_hat, Fm, Fv, Fo))
            tm = atlases[t]["mean"][l]
            tS = atlases[t]["M11"][l] - np.outer(tm, tm)
            Xm.append(Fm)
            ym.append(tm - m_hat)
            Xv.append(Fv)
            yv.append(np.diag(tS) - np.diag(S_hat))
            Xo.append(np.stack([f[iu] for f in Fo], axis=1))
            yo.append((tS - S_hat)[iu])
        Am, *_ = np.linalg.lstsq(np.concatenate(Xm), np.concatenate(ym), rcond=None)
        Av, *_ = np.linalg.lstsq(np.concatenate(Xv), np.concatenate(yv), rcond=None)
        Ao, *_ = np.linalg.lstsq(np.concatenate(Xo), np.concatenate(yo), rcond=None)
        coefs.append((Am, Av, Ao))
        for t in range(train_n):
            m_hat, S_hat, Fm, Fv, Fo = cache[t]
            states[t] = apply_correction(m_hat, S_hat, Fm, Fv, Fo, Am, Av, Ao)
        if l in (0, 15, 31):
            r = np.mean(
                [np.mean((states[t][0] - atlases[t]["mean"][l]) ** 2) for t in range(train_n)]
            )
            print(f"  pass1 L{l+1:02d} train mean-mse {r:.4e} ({time.time()-t0:.0f}s)", flush=True)
    return coefs


def pass2_residual(
    weights_list, atlases, probes, coefs1, train_n: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pass-2: roll with frozen coefs1, fit additive residual heads."""
    iu = np.triu_indices(WIDTH, k=1)
    # Collect per-layer design on frozen rollout
    Xm = [[] for _ in range(DEPTH)]
    ym = [[] for _ in range(DEPTH)]
    Xv = [[] for _ in range(DEPTH)]
    yv = [[] for _ in range(DEPTH)]
    Xo = [[] for _ in range(DEPTH)]
    yo = [[] for _ in range(DEPTH)]

    for t in range(train_n):
        m = np.zeros(WIDTH)
        S = np.eye(WIDTH)
        for l in range(DEPTH):
            W = weights_list[t][l]
            m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(m, S, W)
            k3, k4, kiij = probes[t][l]
            Am, Av, Ao = coefs1[l]
            Fm = mean_features(a, sig_pre, k3, k4)
            Fv = var_features(a, sig_pre, k3, k4)
            Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
            m1, S1 = apply_correction(m_hat, S_hat, Fm, Fv, Fo, Am, Av, Ao)
            tm = atlases[t]["mean"][l]
            tS = atlases[t]["M11"][l] - np.outer(tm, tm)
            Xm[l].append(Fm)
            ym[l].append(tm - m1)
            Xv[l].append(Fv)
            yv[l].append(np.diag(tS) - np.diag(S1))
            Xo[l].append(np.stack([f[iu] for f in Fo], axis=1))
            yo[l].append((tS - S1)[iu])
            m, S = m1, S1

    coefs2 = []
    for l in range(DEPTH):
        dAm, *_ = np.linalg.lstsq(np.concatenate(Xm[l]), np.concatenate(ym[l]), rcond=None)
        dAv, *_ = np.linalg.lstsq(np.concatenate(Xv[l]), np.concatenate(yv[l]), rcond=None)
        dAo, *_ = np.linalg.lstsq(np.concatenate(Xo[l]), np.concatenate(yo[l]), rcond=None)
        Am, Av, Ao = coefs1[l]
        coefs2.append((Am + dAm, Av + dAv, Ao + dAo))
    return coefs2


def eval_coefs(weights_list, atlases, probes, coefs, idxs) -> float:
    mses = []
    for i in idxs:
        est = run_chain(
            [weights_list[i][l] for l in range(DEPTH)],
            probes[i],
            coefs,
            mean_only=False,
        )
        mses.append(float(np.mean((est[-1] - atlases[i]["mean"][-1]) ** 2)))
    return float(np.mean(mses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=40)
    ap.add_argument("--val-n", type=int, default=10)
    ap.add_argument("--n-probe", type=int, default=4096)
    ap.add_argument("--seed-probe-base", type=int, default=1000)
    ap.add_argument("--no-prefetch", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = args.train_n + args.val_n
    print(f"loading {n} nets from full_cache + atlas...", flush=True)
    t0 = time.time()
    weights_list, ids = load_shard_weights(n)
    assert len(weights_list) == n

    if not args.no_prefetch:
        patterns = [f"full/mlp_{mid:05d}.npz" for mid in ids]
        print(f"prefetching {len(patterns)} atlas files...", flush=True)
        snapshot_download(
            ATLAS_REPO, repo_type="dataset", allow_patterns=patterns, max_workers=8
        )

    atlases = []
    probes = []
    for i, mid in enumerate(ids):
        atl = load_atlas(mid)
        # sanity vs shard alm if needed
        atlases.append(atl)
        probes.append(
            probe_cumulants(
                [weights_list[i][l] for l in range(DEPTH)],
                args.n_probe,
                seed=args.seed_probe_base + mid,
            )
        )
        if (i + 1) % 10 == 0:
            print(f"  loaded {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    train_n = args.train_n
    val_idxs = list(range(train_n, n))

    unc = eval_coefs(weights_list, atlases, probes, None, val_idxs)
    print(f"uncorrected val mse={unc:.4e}", flush=True)

    baked_path = Path(__file__).resolve().parent / "port_coefs.b64.txt"
    if baked_path.exists():
        baked = load_coefs_b64(baked_path)
        b = eval_coefs(weights_list, atlases, probes, baked, val_idxs)
        print(f"baked#314695 val mse={b:.4e}", flush=True)

    print("pass-1 sequential DAgger...", flush=True)
    coefs1 = sequential_dagger(weights_list, atlases, probes, train_n)
    p1 = eval_coefs(weights_list, atlases, probes, coefs1, val_idxs)
    print(f"pass-1 val mse={p1:.4e}", flush=True)

    print("pass-2 residual refit on frozen rollout...", flush=True)
    coefs2 = pass2_residual(weights_list, atlases, probes, coefs1, train_n)
    p2 = eval_coefs(weights_list, atlases, probes, coefs2, val_idxs)
    print(f"pass-2 val mse={p2:.4e}", flush=True)

    # Also: pass-2 as full re-fit (replace, not additive) on frozen rollout
    # for diagnostics — sequential_dagger states already saw corrections;
    # here we refit absolute targets from frozen pass-1 states.
    print("pass-2b absolute refit on frozen rollout...", flush=True)
    iu = np.triu_indices(WIDTH, k=1)
    coefs2b = []
    # collect
    cache_l = [[] for _ in range(DEPTH)]
    for t in range(train_n):
        m = np.zeros(WIDTH)
        S = np.eye(WIDTH)
        for l in range(DEPTH):
            W = weights_list[t][l]
            m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(m, S, W)
            k3, k4, kiij = probes[t][l]
            Am, Av, Ao = coefs1[l]
            Fm = mean_features(a, sig_pre, k3, k4)
            Fv = var_features(a, sig_pre, k3, k4)
            Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
            m1, S1 = apply_correction(m_hat, S_hat, Fm, Fv, Fo, Am, Av, Ao)
            tm = atlases[t]["mean"][l]
            tS = atlases[t]["M11"][l] - np.outer(tm, tm)
            cache_l[l].append((Fm, tm - m_hat, Fv, np.diag(tS) - np.diag(S_hat), Fo, (tS - S_hat)[iu]))
            m, S = m1, S1
    for l in range(DEPTH):
        Xm = np.concatenate([c[0] for c in cache_l[l]])
        ym = np.concatenate([c[1] for c in cache_l[l]])
        Xv = np.concatenate([c[2] for c in cache_l[l]])
        yv = np.concatenate([c[3] for c in cache_l[l]])
        Xo = np.concatenate([np.stack([f[iu] for f in c[4]], axis=1) for c in cache_l[l]])
        yo = np.concatenate([c[5] for c in cache_l[l]])
        Am, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        Av, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        Ao, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
        coefs2b.append((Am, Av, Ao))
    p2b = eval_coefs(weights_list, atlases, probes, coefs2b, val_idxs)
    print(f"pass-2b abs-refit val mse={p2b:.4e}", flush=True)

    best_name, best_coefs, best_mse = "pass1", coefs1, p1
    for name, c, m in [("pass2", coefs2, p2), ("pass2b", coefs2b, p2b)]:
        if m < best_mse:
            best_name, best_coefs, best_mse = name, c, m

    coef_path = OUT_DIR / f"dagger_atlas_{best_name}_coefs.b64.txt"
    save_coefs_b64(best_coefs, coef_path)
    # also save pass1 always
    save_coefs_b64(coefs1, OUT_DIR / "dagger_atlas_pass1_coefs.b64.txt")
    save_coefs_b64(coefs2, OUT_DIR / "dagger_atlas_pass2_coefs.b64.txt")

    meta = {
        "train_n": train_n,
        "val_n": args.val_n,
        "n_probe": args.n_probe,
        "uncorrected_val_mse": unc,
        "pass1_val_mse": p1,
        "pass2_residual_val_mse": p2,
        "pass2b_abs_val_mse": p2b,
        "best": best_name,
        "best_val_mse": best_mse,
        "ids": ids,
        "coef_path": str(coef_path),
        "elapsed_s": time.time() - t0,
    }
    if baked_path.exists():
        meta["baked_val_mse"] = b
    meta_path = OUT_DIR / "dagger_atlas_twopass_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {coef_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

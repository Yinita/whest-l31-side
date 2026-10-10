#!/usr/bin/env python3
"""P2-1b: b111 proxies + oracle-feature DAgger gate.

1) Correlate cheap O(W²) proxies with true atlas b111.
2) Refit mean-head DAgger with extra oracle b111 feature (capacity probe).
   If oracle doesn't help OOS, stabilizer injection is closed; keep
   analytic-κ3-transport as a separate later axis.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

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
from probe_n3_blocks import connected, third_order_blocks  # noqa: E402
from train_dagger_atlas import apply_correction, load_shard_weights  # noqa: E402

ATLAS = "keenanpepper/arc-whestbench-higher-moments-2026"
OUT = Path(__file__).resolve().parent / "artifacts"


def load_atlas(mid: int) -> dict:
    path = hf_hub_download(
        ATLAS, f"full/mlp_{mid:05d}.npz", repo_type="dataset", local_files_only=True
    )
    return dict(np.load(path))


def proxies_for_layer(W: np.ndarray, c21: np.ndarray, mu3: np.ndarray, cov: np.ndarray):
    """Return dict of (W_out,) proxy vectors for b111 at this transition."""
    # Exact storable parts
    b3 = (W**3 * mu3[:, None]).sum(0)
    CW = c21 @ W
    unrestricted = ((W**2) * CW).sum(0)
    diag_overlap = (W**3 * np.diagonal(c21)[:, None]).sum(0)
    b21 = 3.0 * (unrestricted - diag_overlap)

    # Proxy A: drop b111 → 0
    p0 = np.zeros(W.shape[1])

    # Proxy B: rank-1 factor on signed |mu3|^{1/3}
    s = np.sign(mu3) * np.abs(mu3) ** (1.0 / 3.0)
    # sum_{i,j,k all} W_ia W_ja W_ka s_i s_j s_k = (W.T @ s)^3
    # subtract diagonal-ish overlaps approximately via b3-scale
    Ws = W.T @ s
    p_r1 = Ws**3 - b3  # crude: full factor cube minus (3)

    # Proxy C: use top-r eigenspace of cov — CP-ish
    # κ ≈ sum_{pqr} G_pqr u_p[i] u_q[j] u_r[k]; take G diagonal-only G_ppp = γ_p
    r = 8
    evals, evecs = np.linalg.eigh(0.5 * (cov + cov.T))
    U = evecs[:, -r:]  # (W, r)
    # project mu3 onto factors: γ_p ≈ (U_p · mu3-related); use U.T @ mu3
    gamma = U.T @ mu3  # (r,)
    # b111-ish: (W.T @ U @ gamma_scaled)^3 style
    # sum_{p} gamma_p (W.T @ U[:,p])^3
    p_eig = np.zeros(W.shape[1])
    for p in range(r):
        p_eig = p_eig + gamma[p] * (W.T @ U[:, p]) ** 3
    p_eig = p_eig - b3  # remove pure (3)-like piece

    # Proxy D: residual of "unrestricted pairwise" vs clipped
    # 3 sum_i≠k W_i^2 W_k c21 already in b21; try 6*sum distinct from cov skew
    # Use: ((c21 * cov) @ W * W).sum — ad hoc
    p_mix = ((c21 * cov) @ W * W).sum(0)

    return {
        "zero": p0,
        "rank1": p_r1,
        "eigh8": p_eig,
        "mix": p_mix,
        "b3": b3,
        "b21": b21,
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if np.std(a) < 1e-30 or np.std(b) < 1e-30:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    train_n, val_n = 24, 6
    n = train_n + val_n
    n_probe = 4096
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    weights_list, ids = load_shard_weights(n)
    atlases = [load_atlas(mid) for mid in ids]
    print(f"loaded {n} atlas nets ({time.time()-t0:.0f}s)", flush=True)

    # --- Part 1: proxy correlations ---
    corrs = {k: [] for k in ("rank1", "eigh8", "mix", "b3", "b21")}
    for i in range(n):
        z = atlases[i]
        W = weights_list[i]
        blocks = third_order_blocks(z, W)
        post = connected(z, "")
        for ell in range(DEPTH - 1):
            true = blocks["b111"][ell]
            px = proxies_for_layer(
                W[ell + 1],
                post["c21"][ell],
                # post mu3
                (
                    z["m3"][ell].astype(np.float64)
                    - 3 * z["mean"][ell].astype(np.float64) * z["m2"][ell].astype(np.float64)
                    + 2 * z["mean"][ell].astype(np.float64) ** 3
                ),
                post["cov"][ell],
            )
            for k in corrs:
                corrs[k].append(corr(px[k], true))
    corr_summary = {k: float(np.nanmean(v)) for k, v in corrs.items()}
    print("mean corr(proxy, b111):", json.dumps(corr_summary, indent=2), flush=True)

    # --- Part 2: DAgger with oracle b111 feature ---
    # Precompute b111[layer] for each net; for layer 0 no previous → zeros
    b111_all = []
    for i in range(n):
        blocks = third_order_blocks(atlases[i], weights_list[i])
        # blocks['b111'] indexes transitions ell=0..30 producing pre at 1..31
        # align to post-layer index l: for correcting layer l outputs, use b111 of
        # transition into this layer's pre (l-1), else 0 for l=0
        arr = np.zeros((DEPTH, WIDTH))
        arr[1:] = blocks["b111"]  # shape (31, W)
        b111_all.append(arr)

    probes = [
        probe_cumulants(
            [weights_list[i][l] for l in range(DEPTH)], n_probe, seed=1000 + ids[i]
        )
        for i in range(n)
    ]
    print(f"probes done ({time.time()-t0:.0f}s)", flush=True)

    baked = load_coefs_b64(Path(__file__).parent / "port_coefs.b64.txt")

    def eval_coefs(coefs, mean_only=False):
        ms = []
        for i in range(train_n, n):
            est = run_chain(
                [weights_list[i][l] for l in range(DEPTH)],
                probes[i],
                coefs,
                mean_only=mean_only,
            )
            y = atlases[i]["mean"][-1].astype(np.float64)
            ms.append(float(np.mean((est[-1] - y) ** 2)))
        return float(np.mean(ms))

    print(f"baked val={eval_coefs(baked):.4e}", flush=True)

    # Sequential DAgger: features = [standard 8 mean feats | b111_oracle]
    # Extend Am to 9 coefs; Av/Ao unchanged from standard fit
    states = [(np.zeros(WIDTH), np.eye(WIDTH)) for _ in range(train_n)]
    coefs_ext = []
    iu = np.triu_indices(WIDTH, k=1)
    for l in range(DEPTH):
        Xm, ym, Xv, yv, Xo, yo, cache = [], [], [], [], [], [], []
        for t in range(train_n):
            W = weights_list[t][l]
            m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(*states[t], W)
            k3, k4, kiij = probes[t][l]
            Fm = mean_features(a, sig_pre, k3, k4)  # (W, 8)
            # append oracle b111 column
            f9 = np.concatenate([Fm, b111_all[t][l][:, None]], axis=1)  # (W, 9)
            Fv = var_features(a, sig_pre, k3, k4)
            Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
            cache.append((m_hat, S_hat, f9, Fv, Fo, Fm))
            tm = atlases[t]["mean"][l].astype(np.float64)
            tS = atlases[t]["M11"][l].astype(np.float64) - np.outer(tm, tm)
            Xm.append(f9)
            ym.append(tm - m_hat)
            Xv.append(Fv)
            yv.append(np.diag(tS) - np.diag(S_hat))
            Xo.append(np.stack([f[iu] for f in Fo], axis=1))
            yo.append((tS - S_hat)[iu])
        Am, *_ = np.linalg.lstsq(np.concatenate(Xm), np.concatenate(ym), rcond=None)
        Av, *_ = np.linalg.lstsq(np.concatenate(Xv), np.concatenate(yv), rcond=None)
        Ao, *_ = np.linalg.lstsq(np.concatenate(Xo), np.concatenate(yo), rcond=None)
        coefs_ext.append((Am, Av, Ao))  # Am length 9
        for t in range(train_n):
            m_hat, S_hat, f9, Fv, Fo, Fm = cache[t]
            # apply with 9-dim Am on f9
            m = m_hat + f9 @ Am
            S = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
            S = np.array(S, copy=True)
            S[np.diag_indices(WIDTH)] = np.diag(S_hat) + Fv @ Av
            states[t] = (m, S)

    # Eval extended coefs manually (run_chain expects 8-dim Am)
    def eval_ext():
        ms = []
        for i in range(train_n, n):
            m = np.zeros(WIDTH)
            S = np.eye(WIDTH)
            for l in range(DEPTH):
                W = weights_list[i][l]
                m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(m, S, W)
                k3, k4, kiij = probes[i][l]
                Am, Av, Ao = coefs_ext[l]
                Fm = mean_features(a, sig_pre, k3, k4)
                f9 = np.concatenate([Fm, b111_all[i][l][:, None]], axis=1)
                Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
                Fv = var_features(a, sig_pre, k3, k4)
                m = m_hat + f9 @ Am
                S = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
                S = np.array(S, copy=True)
                S[np.diag_indices(WIDTH)] = np.diag(S_hat) + Fv @ Av
            y = atlases[i]["mean"][-1].astype(np.float64)
            ms.append(float(np.mean((np.maximum(m, 0.0) - y) ** 2)))
        return float(np.mean(ms))

    # Baseline sequential DAgger without b111 (8-feat) for fair compare on same split
    states = [(np.zeros(WIDTH), np.eye(WIDTH)) for _ in range(train_n)]
    coefs8 = []
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
            tm = atlases[t]["mean"][l].astype(np.float64)
            tS = atlases[t]["M11"][l].astype(np.float64) - np.outer(tm, tm)
            Xm.append(Fm)
            ym.append(tm - m_hat)
            Xv.append(Fv)
            yv.append(np.diag(tS) - np.diag(S_hat))
            Xo.append(np.stack([f[iu] for f in Fo], axis=1))
            yo.append((tS - S_hat)[iu])
        Am, *_ = np.linalg.lstsq(np.concatenate(Xm), np.concatenate(ym), rcond=None)
        Av, *_ = np.linalg.lstsq(np.concatenate(Xv), np.concatenate(yv), rcond=None)
        Ao, *_ = np.linalg.lstsq(np.concatenate(Xo), np.concatenate(yo), rcond=None)
        coefs8.append((Am, Av, Ao))
        for t in range(train_n):
            states[t] = apply_correction(*cache[t], Am, Av, Ao)

    p8 = eval_coefs(coefs8)
    p9 = eval_ext()
    print(f"DAgger-8 val={p8:.4e}", flush=True)
    print(f"DAgger-8+oracle_b111 val={p9:.4e}  ratio={p8/p9 if p9 else None:.4f}", flush=True)

    # Also: use best proxy (by corr) as feature instead of oracle
    best_proxy = max(corr_summary, key=lambda k: abs(corr_summary[k]))
    print(f"best proxy by |corr|: {best_proxy} ({corr_summary[best_proxy]:.3f})", flush=True)

    # Build proxy feature arrays aligned like b111_all
    proxy_all = []
    for i in range(n):
        z = atlases[i]
        W = weights_list[i]
        post = connected(z, "")
        arr = np.zeros((DEPTH, WIDTH))
        mu3 = (
            z["m3"].astype(np.float64)
            - 3 * z["mean"].astype(np.float64) * z["m2"].astype(np.float64)
            + 2 * z["mean"].astype(np.float64) ** 3
        )
        for ell in range(DEPTH - 1):
            px = proxies_for_layer(W[ell + 1], post["c21"][ell], mu3[ell], post["cov"][ell])
            arr[ell + 1] = px[best_proxy]
        proxy_all.append(arr)

    states = [(np.zeros(WIDTH), np.eye(WIDTH)) for _ in range(train_n)]
    coefs_px = []
    for l in range(DEPTH):
        Xm, ym, Xv, yv, Xo, yo, cache = [], [], [], [], [], [], []
        for t in range(train_n):
            W = weights_list[t][l]
            m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(*states[t], W)
            k3, k4, kiij = probes[t][l]
            Fm = mean_features(a, sig_pre, k3, k4)
            f9 = np.concatenate([Fm, proxy_all[t][l][:, None]], axis=1)
            Fv = var_features(a, sig_pre, k3, k4)
            Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
            cache.append((m_hat, S_hat, f9, Fv, Fo))
            tm = atlases[t]["mean"][l].astype(np.float64)
            tS = atlases[t]["M11"][l].astype(np.float64) - np.outer(tm, tm)
            Xm.append(f9)
            ym.append(tm - m_hat)
            Xv.append(Fv)
            yv.append(np.diag(tS) - np.diag(S_hat))
            Xo.append(np.stack([f[iu] for f in Fo], axis=1))
            yo.append((tS - S_hat)[iu])
        Am, *_ = np.linalg.lstsq(np.concatenate(Xm), np.concatenate(ym), rcond=None)
        Av, *_ = np.linalg.lstsq(np.concatenate(Xv), np.concatenate(yv), rcond=None)
        Ao, *_ = np.linalg.lstsq(np.concatenate(Xo), np.concatenate(yo), rcond=None)
        coefs_px.append((Am, Av, Ao))
        for t in range(train_n):
            m_hat, S_hat, f9, Fv, Fo = cache[t]
            m = m_hat + f9 @ Am
            S = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
            S = np.array(S, copy=True)
            S[np.diag_indices(WIDTH)] = np.diag(S_hat) + Fv @ Av
            states[t] = (m, S)

    def eval_px():
        ms = []
        for i in range(train_n, n):
            m = np.zeros(WIDTH)
            S = np.eye(WIDTH)
            for l in range(DEPTH):
                W = weights_list[i][l]
                m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(m, S, W)
                k3, k4, kiij = probes[i][l]
                Am, Av, Ao = coefs_px[l]
                Fm = mean_features(a, sig_pre, k3, k4)
                f9 = np.concatenate([Fm, proxy_all[i][l][:, None]], axis=1)
                Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
                Fv = var_features(a, sig_pre, k3, k4)
                m = m_hat + f9 @ Am
                S = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
                S = np.array(S, copy=True)
                S[np.diag_indices(WIDTH)] = np.diag(S_hat) + Fv @ Av
            y = atlases[i]["mean"][-1].astype(np.float64)
            ms.append(float(np.mean((np.maximum(m, 0.0) - y) ** 2)))
        return float(np.mean(ms))

    ppx = eval_px()
    print(f"DAgger-8+proxy_{best_proxy} val={ppx:.4e}", flush=True)

    out = {
        "corr_summary": corr_summary,
        "baked_val_mse": eval_coefs(baked),
        "dagger8_val_mse": p8,
        "dagger8_oracle_b111_val_mse": p9,
        "oracle_ratio_vs_8": (p8 / p9) if p9 else None,
        "best_proxy": best_proxy,
        "dagger8_proxy_val_mse": ppx,
        "train_n": train_n,
        "val_n": val_n,
        "elapsed_s": time.time() - t0,
        "verdict": (
            "KEEP_INJECT"
            if p9 < 0.9 * p8
            else "KILL_STABILIZER_INJECT_try_analytic_transport"
        ),
    }
    path = OUT / "p21b_b111_gate.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

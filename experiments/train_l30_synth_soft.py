#!/usr/bin/env python3
"""Train L30 soft residual recenter on synthetic He-MLPs (GPU GT + Hadamard).

Target: δ ≈ Y30 − μ_s with quad features; soft scale sc; transfer-check on Mini.
Also probe weight-ish μ_hat (diag-gain residual) for independent soft blend.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
MINI = Path(
    "/home/sti/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
    "/snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini"
)
CACHE = Path("/dev/shm/l30_act_cache")
OUT = ROOT / "experiments" / "l30_synth_soft.npz"
B = int(2.72e11)
UTIL, CALIB, MATMUL = 0.40, 1.036, 2.0
W, D = 256, 32
RECENTER = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GT = 262144
N_SYNTH = 400


def chi_mean(dim: int) -> float:
    half = 0.5 * float(dim)
    return math.sqrt(2.0) * math.exp(math.lgamma(half + 0.5) - math.lgamma(half))


def sylvester_hadamard(dim: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat(
            [torch.cat([h, h], 1), torch.cat([h, -h], 1)],
            0,
        )
    return h


def n_frames_budget() -> int:
    n2 = float(W * W)
    flops_per = D * MATMUL * n2 + D * W * 20.0
    target = max(B * UTIL * CALIB, flops_per)
    n_forward = int(max(2 * W, min(target / flops_per, 500_000)))
    return max(1, n_forward // (2 * W))


NF = n_frames_budget()
E_R = chi_mean(W)
H256 = (sylvester_hadamard(W) / math.sqrt(W)).to(DEVICE)
print(f"device={DEVICE} NF={NF} E_R={E_R:.4f} N_GT={N_GT} N_SYNTH={N_SYNTH}", flush=True)


def he_mlp(seed: int) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    scale = (2.0 / W) ** 0.5
    return [
        (torch.randn(W, W, generator=g) * scale).to(DEVICE) for _ in range(D)
    ]


@torch.no_grad()
def gt_means(weights: list[torch.Tensor], n: int = N_GT) -> torch.Tensor:
    x = torch.randn(n, W, device=DEVICE)
    out = []
    for w in weights:
        x = torch.relu(x @ w)
        out.append(x.mean(0))
    return torch.stack(out, 0)  # (D, W)


@torch.no_grad()
def hadamard_to_l30(weights: list[torch.Tensor], seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    blocks = []
    for _ in range(NF):
        perm = torch.randperm(W, generator=g)
        signs = torch.randint(0, 2, (W,), generator=g).float() * 2 - 1
        M = H256[perm] * signs.reshape(1, -1).to(DEVICE) * E_R
        blocks.append(M)
        blocks.append(-M)
    x = torch.cat(blocks, 0)  # (NF*2*W, W) = (NF*512, W) wait width rows per frame
    # each M is (W, W); NF frames * 2 antipodal → N = NF * 2 * W
    for li in range(RECENTER + 1):
        x = torch.relu(x @ weights[li])
    mu_s = x.mean(0)
    x_pre = x
    for li in range(RECENTER + 1, D):
        x = torch.relu(x @ weights[li])
    mu_b = x.mean(0)
    return mu_s, mu_b, x_pre


@torch.no_grad()
def diag_gain(weights: list[torch.Tensor], stop_li: int):
    mu = torch.zeros(W, device=DEVICE)
    var = torch.ones(W, device=DEVICE)
    inv_sqrt_2pi = 0.3989422804014327
    for li in range(stop_li + 1):
        w = weights[li]
        mu_pre = w.T @ mu
        var_pre = torch.clamp((w * w).T @ var, min=1e-12)
        sigma = torch.sqrt(var_pre)
        alpha = mu_pre / sigma
        phi = torch.exp(-0.5 * alpha * alpha) * inv_sqrt_2pi
        Phi = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sigma * phi
        var = torch.clamp(ez2 - mu * mu, min=0.0)
    return mu, alpha, phi, sigma


def base_feats(mu_s, mu_g, alpha, phi, sigma) -> np.ndarray:
    s = np.maximum(sigma, 1e-8)
    gap = mu_s - mu_g
    gl = np.full_like(mu_s, gap.mean())
    a = alpha
    F = np.stack(
        [
            np.ones_like(mu_s),
            a,
            a * a,
            a * a * a,
            phi,
            gap / s,
            gl / s,
            mu_s / s,
            mu_g / s,
            np.log(s),
            phi * a,
            sigma,
            (mu_s - mu_s.mean()) / s,
            gap,
            s / (s.mean() + 1e-12),
        ],
        0,
    )
    cols = [F[j] for j in range(15)]
    for ai in range(1, 15):
        for bi in range(ai, 15):
            cols.append(F[ai] * F[bi])
    return np.stack(cols, 0)  # (120, W)


def rollout_np(x_pre: np.ndarray, w31: np.ndarray, delta: np.ndarray) -> np.ndarray:
    shift = delta.astype(np.float32) @ w31
    pre = x_pre @ w31  # if x_pre is activations at L30
    return np.maximum(pre + shift, 0.0).mean(0)


def fit_ridge(X, y, lam):
    # X (N, F), y (N,)
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    mu[0] = 0.0
    sd[0] = 1.0
    Xs = (X - mu) / sd
    Fdim = Xs.shape[1]
    beta = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(Fdim), Xs.T @ y)
    return beta, mu, sd


def main():
    t_all = time.time()
    # ---- synth collect ----
    FM = np.zeros((N_SYNTH, 120, W), np.float64)
    MuS = np.zeros((N_SYNTH, W))
    MuB = np.zeros((N_SYNTH, W))
    MuG = np.zeros((N_SYNTH, W))
    Y30 = np.zeros((N_SYNTH, W))
    Y31 = np.zeros((N_SYNTH, W))
    Sig = np.zeros((N_SYNTH, W))
    # store L30 acts? too big for 400 * 25600 * 256 * 4 ≈ 10GB — store only for Mini eval
    print("generating synth...", flush=True)
    for i in range(N_SYNTH):
        seed = 10_000 + i
        weights = he_mlp(seed)
        Y = gt_means(weights).detach().cpu().numpy()
        mu_s, mu_b, _ = hadamard_to_l30(weights, seed=seed)
        mu_s = mu_s.detach().cpu().numpy()
        mu_b = mu_b.detach().cpu().numpy()
        mu_g, alpha, phi, sigma = diag_gain(weights, RECENTER)
        mu_g = mu_g.detach().cpu().numpy()
        alpha = alpha.detach().cpu().numpy()
        phi = phi.detach().cpu().numpy()
        sigma = sigma.detach().cpu().numpy()
        FM[i] = base_feats(mu_s, mu_g, alpha, phi, sigma)
        MuS[i], MuB[i], MuG[i] = mu_s, mu_b, mu_g
        Y30[i], Y31[i] = Y[RECENTER], Y[-1]
        Sig[i] = np.maximum(sigma, 1e-8)
        if i % 20 == 0:
            print(
                f"[{i}] L30s={np.mean((mu_s-Y30[i])**2):.3e} "
                f"L31={np.mean((mu_b-Y31[i])**2):.3e}",
                flush=True,
            )

    base_l30 = float(np.mean((MuS - Y30) ** 2))
    base_l31 = float(np.mean((MuB - Y31) ** 2))
    print(f"SYNTH base L30={base_l30:.4e} L31={base_l31:.4e}", flush=True)

    # 5-fold soft residual on synth (proxy L31 via α-capture curve on L30 quality)
    folds = np.array_split(np.random.default_rng(0).permutation(N_SYNTH), 5)
    best = None
    print("--- synth 5-fold residual → L30 quality ---", flush=True)
    for lam in [1e-3, 3e-3, 1e-2, 3e-2]:
        for sc in [0.05, 0.1, 0.2, 0.35, 0.5]:
            oos_l30 = np.zeros(N_SYNTH)
            for te in folds:
                tr = np.array([j for j in range(N_SYNTH) if j not in set(te.tolist())])
                X = np.concatenate([FM[i].T for i in tr], 0)
                y = np.concatenate([(Y30[i] - MuS[i]) / Sig[i] for i in tr], 0)
                beta, mu, sd = fit_ridge(X, y, lam)
                for i in te:
                    Xi = (FM[i].T - mu) / sd
                    delta = sc * Sig[i] * (Xi @ beta)
                    oos_l30[i] = np.mean((MuS[i] + delta - Y30[i]) ** 2)
            m = float(oos_l30.mean())
            r = base_l30 / m
            print(f"lam={lam:.0e} sc={sc:.2f} L30oos={m:.4e} r={r:.3f}", flush=True)
            if best is None or m < best[0]:
                best = (m, lam, sc, r)

    print("BEST synth L30", best, flush=True)
    # refit full
    X = np.concatenate([FM[i].T for i in range(N_SYNTH)], 0)
    y = np.concatenate([(Y30[i] - MuS[i]) / Sig[i] for i in range(N_SYNTH)], 0)
    beta, feat_mu, feat_sd = fit_ridge(X, y, best[1])
    sc = best[2]

    # ---- Mini transfer with act cache ----
    print("Mini transfer...", flush=True)
    ds = Dataset.load_from_disk(str(MINI))
    n = len(ds)
    mses_base, mses_corr, l30_base, l30_corr = [], [], [], []
    for i in range(n):
        ex = ds[i]
        Y = np.asarray(ex["all_layer_means"], np.float64)
        y30, y31 = Y[RECENTER], Y[-1]
        x_pre = np.load(CACHE / f"x{i}.npy").astype(np.float32)
        mu_s = x_pre.mean(0).astype(np.float64)
        w31 = np.asarray(ex["weights"][31], np.float32)
        # baseline finish
        base_pred = np.maximum(x_pre @ w31, 0.0).mean(0)
        mses_base.append(np.mean((base_pred - y31) ** 2))
        l30_base.append(np.mean((mu_s - y30) ** 2))
        # features via torch diag-gain on Mini weights
        weights = [
            torch.as_tensor(np.asarray(ex["weights"][l], np.float32), device=DEVICE)
            for l in range(D)
        ]
        mu_g, alpha, phi, sigma = diag_gain(weights, RECENTER)
        feats = base_feats(
            mu_s,
            mu_g.detach().cpu().numpy(),
            alpha.detach().cpu().numpy(),
            phi.detach().cpu().numpy(),
            sigma.detach().cpu().numpy(),
        )
        sig = np.maximum(sigma.detach().cpu().numpy(), 1e-8)
        Xi = (feats.T - feat_mu) / feat_sd
        delta = sc * sig * (Xi @ beta)
        pred = rollout_np(x_pre, w31, delta)
        mses_corr.append(np.mean((pred - y31) ** 2))
        l30_corr.append(np.mean((mu_s + delta - y30) ** 2))
        if i % 20 == 0:
            print(i, "base", mses_base[-1], "corr", mses_corr[-1], flush=True)

    mb, mc = float(np.mean(mses_base)), float(np.mean(mses_corr))
    print(
        f"MINI L31 base={mb:.4e} corr={mc:.4e} r={mb/mc:.3f} "
        f"L30r={float(np.mean(l30_base))/float(np.mean(l30_corr)):.3f}",
        flush=True,
    )

    np.savez(
        OUT,
        beta=beta.astype(np.float64),
        feat_mu=feat_mu,
        feat_sd=feat_sd,
        lam=best[1],
        scale=sc,
        synth_l30_r=best[3],
        mini_l31_base=mb,
        mini_l31_corr=mc,
        mini_r=mb / mc,
    )
    print(f"wrote {OUT} elapsed={time.time()-t_all:.0f}s", flush=True)
    print("SHIP?" , mb / mc > 1.03 and float(np.mean(l30_corr)) < float(np.mean(l30_base)), flush=True)


if __name__ == "__main__":
    main()

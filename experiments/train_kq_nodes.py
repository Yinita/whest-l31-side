#!/usr/bin/env python3
"""Offline: optimize a spherical point set under depth-L arc-cosine energy.

Minimizes Σ_{i<j} κ_L(u_i·u_j) (+ repulsion) for unit vectors in R^d.
Writes experiments/kq_nodes_d{d}_n{n}_L{L}.npz for the estimator to load/inline.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def j1(theta: np.ndarray) -> np.ndarray:
    return np.sin(theta) + (np.pi - theta) * np.cos(theta)


def kappa_depth(cosines: np.ndarray, depth: int) -> np.ndarray:
    c = np.clip(cosines, -1.0, 1.0)
    for _ in range(depth):
        th = np.arccos(c)
        c = j1(th) / math.pi
    return c


def energy_and_grad(U: np.ndarray, depth: int) -> tuple[float, np.ndarray]:
    """U: (n, d) unit rows. Energy = mean_{i≠j} κ(u_i·u_j)."""
    n, d = U.shape
    G = U @ U.T
    K = kappa_depth(G, depth)
    np.fill_diagonal(K, 0.0)
    denom = float(n * (n - 1))
    energy = float(K.sum()) / denom

    # dκ/dc via finite depth — use autograd-free: numerical on cosines is pricey.
    # Analytic: for one layer κ=J1(θ)/π, θ=arccos(c), dκ/dc = (π-θ)/π.
    # Multi-layer: chain rule through iterations.
    dK_dc = np.ones_like(G)
    c = np.clip(G, -1.0, 1.0)
    # forward store
    stack = [c.copy()]
    for _ in range(depth):
        th = np.arccos(c)
        c = j1(th) / math.pi
        stack.append(c.copy())
    # backward
    grad_c = np.ones_like(G)  # dE/dκ accumulated then to c
    # energy = sum K / denom with K=final c off-diag; dE/dK_ij = 1/denom
    dE_dK = np.full_like(G, 1.0 / denom)
    np.fill_diagonal(dE_dK, 0.0)
    gc = dE_dK
    for ell in range(depth - 1, -1, -1):
        c_prev = stack[ell]
        th = np.arccos(np.clip(c_prev, -1.0, 1.0))
        # κ = (sinθ + (π-θ)cosθ)/π ; dκ/dc = (π-θ)/π
        dkappa_dc = (np.pi - th) / math.pi
        gc = gc * dkappa_dc
    # dE/dG_ij = gc_ij; G=UU^T ⇒ dE/dU = (gc+gc^T) U
    sym = gc + gc.T
    dU = sym @ U
    # Riemannian project: remove radial component
    dU -= (np.sum(dU * U, axis=1, keepdims=True)) * U
    return energy, dU


def project_sphere(U: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(U, axis=1, keepdims=True)
    return U / np.maximum(nrm, 1e-12)


def optimize(n: int, d: int, depth: int, steps: int, lr: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = project_sphere(rng.standard_normal((n, d)))
    # warm start: mix with a Hadamard-like subset if n<=d
    if n <= d and (d & (d - 1)) == 0:
        # take n random signed-permuted Hadamard rows
        H = np.array([[1.0]])
        while H.shape[0] < d:
            H = np.block([[H, H], [H, -H]])
        H = H / math.sqrt(d)
        perm = rng.permutation(d)[:n]
        signs = rng.choice([-1.0, 1.0], size=d)
        U = H[perm] * signs.reshape(1, -1)
        U = project_sphere(U)
    mom = np.zeros_like(U)
    for t in range(steps):
        e, g = energy_and_grad(U, depth)
        mom = 0.9 * mom + g
        U = project_sphere(U - lr * mom)
        if t % max(1, steps // 10) == 0 or t == steps - 1:
            print(f"step {t:5d}  energy={e:.6f}")
    return U.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = ap.parse_args()
    out = args.out or Path(
        f"experiments/kq_nodes_d{args.d}_n{args.n}_L{args.depth}.npz"
    )
    U = optimize(args.n, args.d, args.depth, args.steps, args.lr, args.seed)
    # also store antipodes separately? estimator will add ±
    np.savez(out, nodes=U.astype(np.float32), depth=args.depth, seed=args.seed)
    print(f"wrote {out} shape={U.shape}")


if __name__ == "__main__":
    main()

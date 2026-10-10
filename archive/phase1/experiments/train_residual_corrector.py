"""Train a radiant-style offline residual corrector on synth MLPs (not Mini).

Fits shared ridge θ so that residual ≈ σ · (θ · f), where f are cheap
per-neuron features from the v0.2.13s base estimate + last-layer stats.
Evaluates transfer on Mini GT without fitting on Mini.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, load_dataset, sample_mlp

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "base", ROOT / "experiments" / "v0212s_floor094.py"
)
base = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(base)

B = int(2.72e11)
WIDTH, DEPTH = 256, 32
N_GT = 80_000
N_TRAIN = 120
N_VAL = 30
RIDGE = 1e-2
OUT = ROOT / "experiments" / "corrector_weights.npz"


def _gt_final(mlp: MLP, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, mlp.width)).astype(np.float32)
    for w in mlp.weights:
        x = np.maximum(x @ np.asarray(w, np.float32), 0.0)
    return x.mean(axis=0).astype(np.float64)


def _features_and_pred(mlp: MLP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu, sigma, feats[W,F]) using budgeted base predict + cheap extras."""
    E = base.Estimator()
    with flops.BudgetContext(flop_budget=B, quiet=True):
        pred = E.predict(mlp, B)
    mu = np.asarray(pred[-1], np.float64)

    w_last = np.asarray(mlp.weights[-1], np.float32)
    col_n = np.sqrt(np.maximum(np.sum(w_last * w_last, axis=0), 1e-30)).astype(np.float64)
    row_n = np.sqrt(np.maximum(np.sum(w_last * w_last, axis=1), 1e-30)).astype(np.float64)
    sigma = np.maximum(col_n * np.sqrt(0.5), 1e-8)  # rough post-ReLU scale proxy

    # gain-only reference (cheap relative to RQMC)
    with flops.BudgetContext(flop_budget=B, quiet=True):
        mu_g = np.asarray(base._GAIN_SCALE * base._gain_cov_final(list(mlp.weights)), np.float64)

    gap = mu - mu_g
    z = mu / sigma
    zg = gap / sigma
    # last-layer fan-in / fan-out asymmetry
    asymmetry = (col_n - row_n.mean()) / (col_n.mean() + 1e-8)

    feats = np.stack(
        [
            np.ones_like(mu),
            z,
            z * z,
            zg,
            zg * zg,
            asymmetry,
            np.log1p(col_n),
            mu_g / sigma,
        ],
        axis=1,
    ).astype(np.float64)
    return mu, sigma, feats


def _fit_ridge(F: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    # F: [N, F], y: [N]
    gram = F.T @ F
    gram = gram + ridge * (np.trace(gram) / gram.shape[0] + 1e-18) * np.eye(gram.shape[0])
    return np.linalg.solve(gram, F.T @ y)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def main() -> None:
    t0 = time.time()
    print(f"sampling {N_TRAIN + N_VAL} synth MLPs + GT N={N_GT}", flush=True)
    rows = []
    for i in range(N_TRAIN + N_VAL):
        rng = fnp.random.default_rng(10_000 + i)
        mlp = sample_mlp(width=WIDTH, depth=DEPTH, rng=rng, seed=20_000 + i)
        y = _gt_final(mlp, N_GT, seed=30_000 + i)
        mu, sigma, feats = _features_and_pred(mlp)
        resid = (y - mu) / sigma
        rows.append((mu, sigma, feats, y, resid, "train" if i < N_TRAIN else "val"))
        if (i + 1) % 10 == 0:
            print(f"  synth {i + 1}/{N_TRAIN + N_VAL} t={time.time() - t0:.0f}s", flush=True)

    F_tr = np.concatenate([r[2] for r in rows if r[5] == "train"], axis=0)
    y_tr = np.concatenate([r[4] for r in rows if r[5] == "train"], axis=0)
    theta = _fit_ridge(F_tr, y_tr, RIDGE)
    print("theta", theta, flush=True)

    def eval_split(tag: str) -> float:
        base_mses, corr_mses = [], []
        for mu, sigma, feats, y, _, t in rows:
            if t != tag:
                continue
            delta = sigma * (feats @ theta)
            base_mses.append(_mse(mu, y))
            corr_mses.append(_mse(mu + delta, y))
        b, c = float(np.mean(base_mses)), float(np.mean(corr_mses))
        print(f"{tag}: base={b:.4e} corr={c:.4e} ratio={b / c:.4f}", flush=True)
        return b / c

    eval_split("train")
    eval_split("val")

    # Transfer to Mini (no refit)
    print("loading Mini for transfer eval", flush=True)
    ds = load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")
    base_mses, corr_mses = [], []
    for i in range(100):
        r = ds[i]
        W = [fnp.asarray(w, dtype=fnp.float32) for w in r["weights"]]
        mlp = MLP(width=256, depth=len(W), weights=W, seed=int(r["mlp_seed"]))
        y = np.asarray(r["final_means"], np.float64)
        mu, sigma, feats = _features_and_pred(mlp)
        delta = sigma * (feats @ theta)
        base_mses.append(_mse(mu, y))
        corr_mses.append(_mse(mu + delta, y))
        if (i + 1) % 20 == 0:
            print(f"  mini {i + 1}/100", flush=True)
    b, c = float(np.mean(base_mses)), float(np.mean(corr_mses))
    print(f"MINI transfer: base={b:.4e} corr={c:.4e} ratio={b / c:.4f}", flush=True)

    np.savez(
        OUT,
        theta=theta.astype(np.float64),
        ridge=np.asarray(RIDGE),
        feature_names=np.asarray(
            ["1", "z", "z2", "zg", "zg2", "asym", "log_coln", "zgain"], dtype=object
        ),
    )
    print(f"wrote {OUT} t={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

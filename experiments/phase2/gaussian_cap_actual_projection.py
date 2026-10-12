"""Actualstate capped211 vs liveCoff metric proxy, targetfree diagnostic."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path

import numpy as np


def main():
    output = Path("results/phase2/ablation_v30_047_r02_actual_projection.json")
    assert not output.exists()
    d = dict(np.load("benchmark/data/v29_real_source_v30_059_r05/network_0.npz"))
    c = d["covariance"].astype("float64")
    np.fill_diagonal(c, 0)
    alpha = d["alpha"].astype("float64")
    sigma = d["sigma"].astype("float64")
    p = d["w1"].astype("float64")
    q = d["w2"].astype("float64")
    phi = np.exp(-alpha * alpha / 2) / np.sqrt(2 * np.pi)
    mean = sigma * phi + alpha * sigma * p
    a = 2 * mean * (1 - p)
    b = 2 * (p * (1 - p) - mean * phi / sigma)
    t = c @ a
    cap = p[:, None] * p[None, :] * ((c * b[None, :]) @ c)
    cap += c * (
        q[:, None] * p[None, :] * (t[:, None] - c * a[None, :])
        + q[None, :] * p[:, None] * (t[None, :] - c * a[:, None])
    )
    np.fill_diagonal(cap, 0)
    lam = float(np.sum(c * cap) / np.sum(c * c))
    residual = cap - lam * c
    output.write_text(
        json.dumps(
            dict(
                layer=14,
                network=0,
                cap_rms=float(np.sqrt(np.mean(cap * cap))),
                projection_lambda=lam,
                residual_rms=float(np.sqrt(np.mean(residual * residual))),
                explained_fraction=float(1 - np.sum(residual * residual) / np.sum(cap * cap)),
                scope="Gaussian3path capped211 of actualV29prestate projected ontoCoff, targetfree; this leastsquares overlap alone does not identify existinglambda core cumulant conversion or harmonic scaling; no score",
            ),
            indent=2,
        )
        + "\n"
    )
    print(output.read_text())


if __name__ == "__main__":
    main()

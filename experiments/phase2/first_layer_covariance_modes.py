"""O13/O14 actual pre-registered first-layer covariance rank1 feasibility."""

# ruff: noqa: E402 -- force BLAS threads before numerical imports
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import json
from pathlib import Path

import numpy as np


def main():
    rows = []
    for i in range(8):
        w = np.load(f"benchmark/data/independent_v30_115/network_{i}.npz")["weights"][0].astype(
            np.float64
        )
        cov = w.T @ w
        # Deterministic power iteration, avoids full eigen decomposition; not exact dominant proof.
        q = np.ones(1024)
        q /= np.linalg.norm(q)
        for _ in range(64):
            q = cov @ q
            q /= np.linalg.norm(q)
        value = float(q @ cov @ q)
        proxy = value * np.outer(q, q)
        off = cov.copy()
        np.fill_diagonal(off, 0)
        offproxy = proxy.copy()
        np.fill_diagonal(offproxy, 0)
        rows.append(
            dict(
                network_id=i,
                rayleigh_value=value,
                eigen_residual=float(np.linalg.norm(cov @ q - value * q)),
                full_energy_fraction=float(value**2 / np.sum(cov * cov)),
                offdiag_relative_frobenius_error=float(
                    np.linalg.norm(off - offproxy) / np.linalg.norm(off)
                ),
                offdiag_max_error=float(np.max(abs(off - offproxy))),
            )
        )
    p = Path("results/phase2/ablation_v30_075_076_actual_first_layer_modes.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Actual preregistered first-layer covariance rank1 power-iteration proxy energy and errors measured without labels. Not full deep-state covariance/source tensor bound; no candidate tuning or independent score.",
                scope="Actual network first-layer feasibility diagnostic",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

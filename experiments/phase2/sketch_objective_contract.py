"""H22/H24 fixed legal downstream objective, PSD and weighted lowrank identity."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(524)
    rows = []
    for n, rank in [(7, 2), (11, 3), (19, 4)]:
        residual = rng.normal(size=(n, n))
        next_weights = rng.normal(size=(n, n)) / np.sqrt(n)
        gate = rng.uniform(0.1, 1, size=n)
        downstream = next_weights * gate[None, :]
        objective = downstream @ downstream.T
        values = np.linalg.eigvalsh(objective)
        assert values.min() > -1e-12
        q, _ = np.linalg.qr(residual @ downstream[:, :rank])
        approx = q @ (q.T @ residual)
        error = residual - approx
        direct = float(np.sum((error @ downstream) ** 2))
        trace = float(np.trace(error @ objective @ error.T))
        assert abs(direct - trace) < 1e-10
        rows.append(
            dict(
                n=n,
                rank=rank,
                min_eigenvalue=float(values.min()),
                objective_error=abs(direct - trace),
                weighted_residual=direct,
            )
        )
    path = Path("results/phase2/ablation_h22_h24_sketch_objective.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                rule="One future-layer W and frozen legal gate only; sketch=Rres*(Wnext*gate)[:,:rank], rank unchanged. H22 applies to D21; H24 needs explicit full basis weighting mapping.",
                conclusion="PSD weighted residual objective exact; no reference used. Fixed next gate must be estimated from current lawful state and kept independent of target. Full state/cost and candidate integration pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

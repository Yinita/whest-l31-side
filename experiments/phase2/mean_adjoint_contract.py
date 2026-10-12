"""L25 legal mean-only Jacobian adjoint contract; not full source sensitivity."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(120)
    rows = []
    for n, depth in [(4, 3), (7, 5), (11, 2)]:
        weights = rng.normal(size=(depth, n, n)) / np.sqrt(n)
        gates = rng.uniform(size=(depth, n))
        delta = rng.normal(size=n)
        test = rng.normal(size=n)
        tangent = delta.copy()
        for w, gate in zip(weights, gates):
            tangent = gate * (w.T @ tangent)
        adjoint = test.copy()
        for w, gate in zip(weights[::-1], gates[::-1]):
            adjoint = w @ (gate * adjoint)
        error = float(abs(test @ tangent - adjoint @ delta))
        assert error < 1e-12
        rows.append(dict(n=n, depth=depth, dot_error=error))
    path = Path("results/phase2/ablation_v30_120_mean_adjoint_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="J_l=diag(Phi_l)W_l.T; adjoint J_l.T v=W_l(Phi_l*v)",
                conclusion="Exact dot-product duality for frozen legal mean gates. It omits covariance/cumulant state responses; a source-specific D3/D21 removal contraction and rank-aware cost proof are still required. No ranking or deletion outcome measured.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

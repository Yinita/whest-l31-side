"""L08 freeze default-zero correction trajectory and affine ridge invariance."""

import ast
import json
from pathlib import Path

import numpy as np


def main():
    source = Path("estimators/k3_v29/estimator.py")
    tree = ast.parse(source.read_text())
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "NO_CORR" for t in n.targets)
    )
    assert '"1"' in ast.unparse(assignment.value) or "'1'" in ast.unparse(assignment.value)
    rng = np.random.default_rng(108)
    x = rng.normal(size=(100, 13))
    y = rng.normal(size=100)
    zero = np.zeros((100, 2))
    base = np.linalg.solve(x.T @ x + 0.001 * np.eye(13), x.T @ y)
    added = np.concatenate([x, zero], axis=1)
    beta = np.linalg.solve(added.T @ added + 0.001 * np.eye(15), added.T @ y)
    error = float(np.max(abs(x @ base - added @ beta)))
    assert error < 1e-10
    p = Path("results/phase2/ablation_v30_103_zero_trajectory_contract.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                max_prediction_difference=error,
                zero_coefficients=beta[-2:].tolist(),
                conclusion="Frozen V29 default NO_CORR=1 makes previous original correction and propagated/Phi-gated correction identically zero. Adding these two zero columns cannot add statistical information; ridge predictions invariant. Nonzero alternative parent would require explicit new version, not silently changing baseline.",
                scope="Exact default-path feature degeneracy proof; no official score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

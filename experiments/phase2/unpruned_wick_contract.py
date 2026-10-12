"""Independent Hermite Wick check for all newly reconstructed vectors."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermeval


def main():
    spec = importlib.util.spec_from_file_location(
        "unpruned", "estimators/ablation_v30_036_r03/estimator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for (k, p), (const, e, poly, cdfpoly) in zip(module.WICK_PAIRS, module.WICK_UNIFIED):
        for alpha in (-2.0, -0.3, 0.0, 1.7):
            sigma = 1.3
            phi = math.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi)
            actual = (
                const
                * sigma**e
                * (
                    np.polynomial.polynomial.polyval(alpha, poly or [0]) * phi
                    + np.polynomial.polynomial.polyval(alpha, cdfpoly or [0])
                    * ((1 + math.erf(alpha / math.sqrt(2))) / 2)
                )
            )
            if k >= p:
                d = k - p
                expected = (
                    math.factorial(p) * ((1 + math.erf(alpha / math.sqrt(2))) / 2)
                    if d == 0
                    else math.factorial(p)
                    * (-1) ** (d - 1)
                    * hermeval(alpha, [0.0] * (d - 1) + [1.0])
                    * phi
                    / sigma**d
                )
                assert abs(actual - expected) < 1e-10
                rows.append(dict(k=k, p=p, alpha=alpha, error=float(abs(actual - expected))))
    out = Path("results/phase2/ablation_v30_036_r03_wick_contract.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="Independent NumPy Hermite basis for k>=p; inheritedlowerderivative vectors unchanged, fullproductionnumeric gate pending",
            ),
            indent=2,
        )
        + "\n"
    )
    print(len(rows))


if __name__ == "__main__":
    main()

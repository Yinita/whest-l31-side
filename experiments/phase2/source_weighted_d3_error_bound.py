"""O14 output-weighted diagonal contraction bound for covariance edge error."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(764)
    rows = []
    for n in (4, 7, 11):
        error = rng.normal(size=(n, n, n))
        w = rng.normal(size=(n, n)) / np.sqrt(n)
        coeff = rng.normal(size=n)
        delta = np.einsum("ijk,ai,aj,ak->a", error, w, w, w)
        # Row tensor test has Frobenius norm ||w_a||²^(3/2).
        bound = np.linalg.norm(error) * (np.linalg.norm(w, axis=1) ** 3)
        assert np.all(abs(delta) <= bound + 1e-10)
        scalar = float(coeff @ delta)
        combined = np.einsum("a,ai,aj,ak->ijk", coeff, w, w, w)
        tighter = float(np.linalg.norm(error) * np.linalg.norm(combined))
        assert abs(scalar) <= tighter + 1e-10
        rows.append(
            dict(
                n=n,
                output_error_norm=float(np.linalg.norm(delta)),
                output_bound_norm=float(np.linalg.norm(bound)),
                scalar_error=scalar,
                weighted_tensor_bound=tighter,
            )
        )
    p = Path("results/phase2/ablation_v30_076_weighted_output_error_bound.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="|<E,w_a³>|<=||E||F||w_a||2³; |sum_a c_a<E,w_a³>|<=||E||F||sum_a c_a w_a³||F",
                conclusion="Output-oriented contraction error bounds checked. Bound may be loose; actual source E unavailable and no reference labels used. Does not grant rank1 approximation acceptance.",
                scope="Synthetic deterministic bound contract",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

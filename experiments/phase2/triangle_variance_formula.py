"""H31 exact Rademacher variance of nonsymmetric trace(A^3) estimator."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(613)
    rows = []
    for n in (3, 4, 7):
        c = rng.normal(size=(n, n))
        c = (c + c.T) / 2
        np.fill_diagonal(c, 0)
        a = rng.normal(size=n)
        matrix = a[:, None] * c
        b = matrix @ matrix @ matrix
        probes = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)
        samples = np.einsum("si,ij,sj->s", probes, b, probes)
        symmetric = (b + b.T) / 2
        np.fill_diagonal(symmetric, 0)
        formula = 2 * np.sum(symmetric * symmetric)
        assert abs(formula - samples.var()) < 1e-10
        signal = float(np.trace(b))
        needed = int(np.ceil(formula / max((0.1 * abs(signal)) ** 2, 1e-30)))
        rows.append(
            dict(
                n=n,
                variance_exact=float(formula),
                variance_enumerated=float(samples.var()),
                trace_signal=signal,
                probes_for_10pct_relative_standard_error=needed,
            )
        )
    path = Path("results/phase2/ablation_v30_061_triangle_variance_formula.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="Var(z^T B z)=2||offdiag((B+B.T)/2)||_F², B=(diag(a)Coff)^3; independent sample average divides by s",
                conclusion="Exact variance law verified by exhaustive probes; real network signal/variance and remaining-budget sample count not measured. Near-zero trace makes relative accuracy expensive; cannot claim unbiasedness alone solves score.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

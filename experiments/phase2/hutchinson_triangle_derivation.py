"""Exact expectation of triangle trace estimator with distinct exclusions."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(61)
    rows = []
    for n in [3, 4, 7]:
        c = rng.normal(size=(n, n))
        c = (c + c.T) / 2
        np.fill_diagonal(c, 0)
        a = rng.normal(size=n)
        matrix = a[:, None] * c
        exact = sum(
            a[i] * a[j] * a[k] * c[i, j] * c[j, k] * c[k, i]
            for i, j, k in itertools.product(range(n), repeat=3)
            if len({i, j, k}) == 3
        )
        probes = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)
        values = np.sum(probes * ((probes @ matrix.T) @ matrix.T @ matrix.T), axis=1)
        err = abs(float(values.mean()) - exact)
        assert err < 1e-11
        rows.append(
            dict(
                n=n,
                exact=exact,
                expectation=float(values.mean()),
                error=err,
                variance=float(values.var()),
                probes=len(probes),
            )
        )
    p = Path("results/phase2/ablation_v30_061_triangle_unbiased.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                scope="ExhaustiveRademachersmall expectation of tr((diag(a)Coff)^3); fullactualcovariance not lowrank",
                rows=rows,
                cost="For each output, eachprobe3densematvec O(n²), alloutputs O(s*n³); variance and FLOP gate must be measured",
                conclusion="Unbiased under fixedmatrix; reference-free seed needed, no candidate score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

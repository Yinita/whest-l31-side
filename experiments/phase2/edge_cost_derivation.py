"""Exact diagonal contraction order for K111 times covariance-edge production."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(74)
    rows = []
    for n, r in [(4, 2), (7, 3), (11, 4)]:
        a, b, c = rng.normal(size=(3, n, r))
        u, v = rng.normal(size=(2, n))
        w = rng.normal(size=(n, n))
        covariance = rng.normal(size=(n, n))
        covariance = (covariance + covariance.T) / 2
        tensor = np.einsum("ir,jr,kr->ijk", a, b, c)
        edge = tensor * (
            u[:, None, None] * v[None, :, None] * v[None, None, :] * covariance[None, :, :]
            + v[:, None, None] * u[None, :, None] * v[None, None, :] * covariance[:, None, :]
            + v[:, None, None] * v[None, :, None] * u[None, None, :] * covariance[:, :, None]
        )
        mask = np.fromfunction(lambda i, j, k: (i != j) & (i != k) & (j != k), (n, n, n))
        edge *= mask
        reference = np.einsum("ijk,oi,oj,ok->o", edge, w, w, w)
        # No cubic tensor in this candidate-order computation; per-row pair matrix.
        result = []
        for output in range(n):
            total = 0.0
            for rank in range(r):
                for orientation in range(3):
                    legs = [a[:, rank] * w[output], b[:, rank] * w[output], c[:, rank] * w[output]]
                    singled = orientation
                    others = [axis for axis in range(3) if axis != singled]
                    x = legs[singled] * u
                    y = legs[others[0]] * v
                    z = legs[others[1]] * v
                    pair = covariance * y[:, None] * z[None, :]
                    np.fill_diagonal(pair, 0)
                    sums = x.sum() - x[:, None] - x[None, :]
                    total += np.sum(pair * sums)
            result.append(total)
        rows.append(dict(n=n, rank=r, max_error=float(np.max(abs(np.asarray(result) - reference)))))
    assert max(row["max_error"] for row in rows) < 1e-9
    output = Path("results/phase2/ablation_v30_074_edge_cost.json")
    assert not output.exists()
    output.write_text(
        json.dumps(
            dict(
                scope="Exact diagonal final W contraction of represented CP K111 times covariance edge, before additional K111 repeated-content subtraction",
                formula="For each output o,rank t,orientation: sum_j!=k Cjk*yj*zk*(sum_i xi-xj-xk)",
                largest_intermediate="n by n, per output/rank/orientation",
                cost="3*R*n outputs * O(n^2): O(R*n^3), plus O(R*n^2) leg scaling; covariance rank not required; CP symmetrization multiplies constant",
                gate="At n1024,R16, leading elementwise cost comparable to full network FLOPs; no end-to-end score or admissible120s/0.4s resource claim",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

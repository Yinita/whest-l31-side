"""Exact distinct symmetric CP transport via repeated-content thin correction."""

import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    output = Path("results/phase2/ablation_v30_059_060_r04_factor_repeats.json")
    assert not output.exists()
    rng = np.random.default_rng(6002)
    rows = []
    for n, r in ((4, 2), (8, 3), (16, 4)):
        legs = rng.normal(size=(3, n, r)).astype("float32")
        w = (rng.normal(size=(n, n)) / np.sqrt(n)).astype("float32")
        dense = (
            sum(np.einsum("ir,jr,kr->ijk", *parts) for parts in itertools.permutations(legs)) / 6
        )
        mask = np.fromfunction(lambda i, j, k: (i != j) & (i != k) & (j != k), (n, n, n))
        expected = np.einsum("ai,bj,ck,ijk->abc", w, w, w, dense * mask, optimize=True)
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            a, b, c = [fnp.array(x) for x in legs]
            matrix = fnp.array(w)
            u, v, t = [matrix @ x for x in (a, b, c)]
            d3 = fnp.sum(a * b * c, axis=1)
            full3 = fnp.sum(u * v * t, axis=1)
            full21 = ((u * v) @ t.T + (u * t) @ v.T + (v * t) @ u.T) / 3
            w2 = matrix * matrix
            transported = (u @ (b * c).T + v @ (a * c).T + t @ (a * b).T) / 3
            repeated3 = 3 * fnp.sum(w2 * transported, axis=1) - 2 * ((w2 * matrix) @ d3)
            repeated21 = (
                ((w2 @ (a * b)) @ t.T + (w2 @ (a * c)) @ v.T + (w2 @ (b * c)) @ u.T) / 3
                + 2 * (matrix * transported) @ matrix.T
                - 2 * ((w2 * d3[None, :]) @ matrix.T)
            )
            actual3 = full3 - repeated3
            actual21 = full21 - repeated21
        expected3 = np.einsum("iii->i", expected)
        expected21 = np.stack([expected[i, i, :] for i in range(n)])
        err3 = float(np.linalg.norm(np.asarray(actual3) - expected3) / np.linalg.norm(expected3))
        err21 = float(
            np.linalg.norm(np.asarray(actual21) - expected21) / np.linalg.norm(expected21)
        )
        assert max(err3, err21) < 1e-5
        rows.append(
            dict(
                n=n,
                rank=r,
                d3_relative_error=err3,
                d21_relative_error=err21,
                flops=ctx.summary_dict()["flops_used"],
                max_charged_elements=n * n,
            )
        )
    output.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="R_abc=delta_ab D21_ac+delta_ac D21_ab+delta_bc D21_ba-2delta_abdelta_ac D3_a; transport contracted without n3 arrays",
                scope="Exact symmetric CP distinct transport component; dense oracle only outside charged computation",
                limitations=[
                    "Repeated correction retains dense n3-cost GEMM (matrix*transported)@matrix.T; first term and transported reuse CP factors",
                    "Rank1 edge source factors and birth coverage integration pending",
                    "No real network score",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

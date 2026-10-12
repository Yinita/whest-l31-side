"""Exact edge-times-CP output D3 with distinct exclusions and legal operations."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    path = Path("results/phase2/ablation_v30_074_r02_charged_edge_contraction.json")
    assert not path.exists()
    rng = np.random.default_rng(7402)
    rows = []
    for n, r in ((4, 2), (8, 3), (16, 4)):
        legs = rng.normal(size=(3, n, r)).astype("float32")
        u, v = rng.normal(size=(2, n)).astype("float32")
        w = rng.normal(size=(n, n)).astype("float32")
        cov = rng.normal(size=(n, n)).astype("float32")
        cov = (cov + cov.T) / 2
        tensor = np.einsum("ir,jr,kr->ijk", *legs)
        edge = tensor * (
            u[:, None, None] * v[None, :, None] * v[None, None, :] * cov[None, :, :]
            + v[:, None, None] * u[None, :, None] * v[None, None, :] * cov[:, None, :]
            + v[:, None, None] * v[None, :, None] * u[None, None, :] * cov[:, :, None]
        )
        mask = np.fromfunction(lambda i, j, k: (i != j) & (i != k) & (j != k), (n, n, n))
        expected = np.einsum("ijk,oi,oj,ok->o", edge * mask, w, w, w)
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            a, b, c = [fnp.array(x) for x in legs]
            uu, vv = fnp.array(u), fnp.array(v)
            ww = fnp.array(w)
            cc = fnp.array(cov)
            outputs = []
            for output in range(n):
                total = fnp.zeros((), dtype=fnp.float32)
                for rank in range(r):
                    for orientation in range(3):
                        factors = [
                            a[:, rank] * ww[output],
                            b[:, rank] * ww[output],
                            c[:, rank] * ww[output],
                        ]
                        others = [axis for axis in range(3) if axis != orientation]
                        x = factors[orientation] * uu
                        y = factors[others[0]] * vv
                        z = factors[others[1]] * vv
                        pair = cc * y[:, None] * z[None, :]
                        fnp.fill_diagonal(pair, 0)
                        total = total + fnp.sum(pair * (fnp.sum(x) - x[:, None] - x[None, :]))
                outputs.append(total)
            actual = fnp.stack(outputs)
        relative = float(np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected))
        assert relative < 1e-5
        rows.append(
            dict(
                n=n,
                rank=r,
                relative_error=relative,
                flops=ctx.summary_dict()["flops_used"],
                max_intermediate_elements=n * n,
            )
        )
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="Full represented CP edge D3 output contraction with all source indices distinct; nonsymmetric CP summands suffice by linearity, no cubic candidate tensor",
                cost="3Rn matrix elementwise contractions; O(Rn3) plus O(Rn2), many API calls; full network timing not established",
                score=None,
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

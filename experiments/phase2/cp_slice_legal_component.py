"""H13/H14/H30 legal CP D3/D21 after one transport, with charged operations."""

import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(433)
    rows = []
    for n, r in [(8, 2), (16, 3), (32, 4)]:
        factors = rng.normal(size=(3, n, r)).astype("float32")
        w = rng.normal(size=(n, n)).astype("float32") / np.sqrt(n)
        t = sum(np.einsum("ir,jr,kr->ijk", *p) for p in itertools.permutations(factors)) / 6
        exact = np.einsum("ai,bj,ck,ijk->abc", w, w, w, t, optimize=True)
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            weight = fnp.array(w)
            a, b, c = [weight @ fnp.array(f) for f in factors]
            d3 = fnp.sum(a * b * c, axis=1)
            d21 = ((a * b) @ c.T + (a * c) @ b.T + (b * c) @ a.T) / 3
        e3 = float(np.max(abs(np.asarray(d3) - np.einsum("iii->i", exact))))
        e21 = float(
            np.max(
                abs(
                    np.asarray(d21)
                    - np.array([[exact[i, i, j] for j in range(n)] for i in range(n)])
                )
            )
        )
        assert max(e3, e21) < 2e-5
        rows.append(
            dict(n=n, rank=r, d3_error=e3, d21_error=e21, flops=ctx.summary_dict()["flops_used"])
        )
    p = Path("results/phase2/ablation_h13_h14_h30_cp_slice_component.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Actual legal CP transport and slice operations verified. Birth-source coefficients/own repeated subtraction and finite-lifetime state integration remain separate gates.",
                scope="Charged represented-source component",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

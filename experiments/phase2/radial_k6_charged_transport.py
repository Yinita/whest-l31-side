"""Legal radial K6 metric transport without sixth-order candidate tensors."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.radial_k6_transport import lift


def main():
    output = Path("results/phase2/ablation_v30_084_r02_charged_metric_transport.json")
    assert not output.exists()
    rng = np.random.default_rng(8402)
    rows = []
    for n in (2, 3, 4):
        w = (rng.normal(size=(n, n)) / np.sqrt(n)).astype("float32")
        a = rng.normal(size=(n, n)).astype("float32")
        metric = a @ a.T
        q = 0.13
        dense = q * lift(metric.astype("float64"))
        exact = np.einsum(
            "ai,bj,ck,dl,em,fp,ijklmp->abcdef", w, w, w, w, w, w, dense, optimize=True
        )
        expected = np.array([exact[(i,) * 6] for i in range(n)])
        with flops.BudgetContext(flop_budget=2**30) as ctx:
            matrix = fnp.array(w)
            original = fnp.array(metric)
            transported = matrix @ original @ matrix.T
            diagonal = fnp.diag(transported)
            marginal = q * diagonal * diagonal * diagonal
        err = float(np.linalg.norm(np.asarray(marginal) - expected) / np.linalg.norm(expected))
        assert err < 2e-6
        rows.append(
            dict(
                n=n,
                marginal_relative_error=err,
                flops=ctx.summary_dict()["flops_used"],
                largest_array_elements=n * n,
            )
        )
    output.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="Legal scalar radial state transport and marginal readout only; explicit sixth tensor independent oracle outside charged path",
                limitations=[
                    "Dense metric transport cubic cost, amortization not assumed",
                    "Full nonlinear radial scalar production and omitted harmonics unresolved",
                    "No benchmark candidate or adjusted score",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

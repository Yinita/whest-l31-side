"""H15/O03 legal weighted CP capped contraction including repeated exclusions."""

import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(456)
    rows = []
    for n, r in [(8, 2), (16, 3), (32, 4)]:
        a, b, c = rng.normal(size=(3, n, r)).astype("float32")
        u = rng.normal(size=n).astype("float32")
        tensor = (
            sum(
                np.einsum("ir,jr,kr->ijk", x, y, z) for x, y, z in itertools.permutations((a, b, c))
            )
            / 6
        )
        expected = np.einsum("i,ijk->jk", u, tensor)
        d21 = np.array([[tensor[i, i, j] for j in range(n)] for i in range(n)])
        expected -= u[:, None] * d21 + u[None, :] * d21.T
        np.fill_diagonal(expected, 0)
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            aa, bb, cc = [fnp.array(x) for x in (a, b, c)]
            weight = fnp.array(u)
            forward = (bb * (aa.T @ weight)[None, :]) @ cc.T
            forward = forward + (cc * (bb.T @ weight)[None, :]) @ aa.T
            forward = forward + (aa * (cc.T @ weight)[None, :]) @ bb.T
            full = (forward + forward.T) / 6
            repeats = ((aa * bb) @ cc.T + (aa * cc) @ bb.T + (bb * cc) @ aa.T) / 3
            cap = full - weight[:, None] * repeats - weight[None, :] * repeats.T
            fnp.fill_diagonal(cap, 0)
        error = float(np.max(abs(np.asarray(cap) - expected)))
        assert error < 2e-5
        rows.append(dict(n=n, rank=r, error=error, flops=ctx.summary_dict()["flops_used"]))
    path = Path("results/phase2/ablation_v30_045_weighted_cp_cap_component.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Legal CP weighted capped contraction, all a=b/c and b=c exclusions verified. Full hubsource state extraction and official coefficients/production map still pending; no score.",
                scope="Charged component correctness",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

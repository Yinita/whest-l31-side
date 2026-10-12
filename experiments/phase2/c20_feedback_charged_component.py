"""C20 measured factor regroup charge with unavoidable dense cross term retained."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(203)
    rows = []
    for n, r in [(32, 4), (64, 8), (128, 16)]:
        values = [
            rng.normal(size=s).astype("float32")
            for s in [(n, n), (n, n), (n, r), (n, r), (n, r), (n, r), (n,)]
        ]
        outputs = []
        costs = []
        for regroup in (False, True):
            with flops.BudgetContext(flop_budget=2**35) as ctx:
                a, p, f1, r1, f2, r2, w = [fnp.array(v) for v in values]
                xt = f1 @ r1.T
                yt = f2 @ r2.T
                if regroup:
                    ap = a * p
                    linear = 3 * fnp.sum((ap @ r2) * f2, axis=1) + fnp.sum(
                        ((ap * w[None, :]) @ r1) * f1, axis=1
                    )
                else:
                    linear = fnp.sum(p * (3 * a * yt + xt * a * w[None, :]), axis=1)
                result = linear + fnp.sum(p * xt * yt, axis=1)
            outputs.append(np.asarray(result))
            costs.append(ctx.summary_dict()["flops_used"])
        relative = float(np.linalg.norm(outputs[0] - outputs[1]) / np.linalg.norm(outputs[0]))
        assert relative < 1e-6
        rows.append(
            dict(
                n=n,
                rank=r,
                dense_original_flops=costs[0],
                regrouped_flops=costs[1],
                ratio=costs[1] / costs[0],
                relative_error=relative,
            )
        )
    path = Path("results/phase2/ablation_v30_020_charged_feedback_component.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Naive regroup retaining Xt/Yt and cross term measured legally; component cost increases. Sharing contractions would be a distinct implementation, no whole estimator score measured.",
                scope="Charged exact component negative",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

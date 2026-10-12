"""L14 constrained hinge inference legal component and boundary continuity."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(1094)
    rows = []
    beta = rng.normal(size=16).astype("float32")
    for knot in (-1, 0, 1):
        features = rng.normal(size=(64, 13)).astype("float32")
        features[:, 0] = 1
        features[:, 4] = np.tile([knot - 1e-5, knot + 1e-5], 32)
        # Adjacent rows share every non-alpha feature to test continuity.
        features[1::2] = features[::2]
        features[1::2, 4] = knot + 1e-5
        hinges = np.maximum(features[:, 4, None] - np.array([-1, 0, 1], dtype="float32"), 0)
        expected = np.concatenate([features, hinges], axis=1) @ beta
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            f = fnp.array(features)
            b = fnp.array(beta)
            h = fnp.maximum(f[:, 4, None] - fnp.array([-1.0, 0.0, 1.0], dtype=fnp.float32), 0)
            pred = fnp.concatenate([f, h], axis=1) @ b
        error = float(np.max(abs(np.asarray(pred) - expected)))
        assert error < 1e-5
        rows.append(
            dict(
                knot=knot,
                inference_error=error,
                boundary_jump=float(np.max(abs(np.asarray(pred)[::2] - np.asarray(pred)[1::2]))),
                flops=ctx.summary_dict()["flops_used"],
            )
        )
    p = Path("results/phase2/ablation_v30_109_hinge_legal_component.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Legal continuous hinge inference and finite boundary behavior checked; arbitrary synthetic coefficients only. Actual fixed networkLOO negative model remains, full charged candidate/parent gate unmet.",
                scope="Charged inference subset",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

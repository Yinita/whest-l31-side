"""L07 legal actualg4 normalization feature component, scalar not substituted."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng = np.random.default_rng(1024)
    rows = []
    for n in (32, 128, 1024):
        g = rng.normal(size=n).astype("float32")
        sigma = rng.uniform(0.2, 2, size=n).astype("float32")
        phi = rng.uniform(0, 0.4, size=n).astype("float32")
        expected = np.stack([g / sigma**4, g / sigma**4 * phi], axis=1)
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            gg, ss, pp = [fnp.array(x) for x in (g, sigma, phi)]
            normalized = gg / (ss**4)
            features = fnp.stack([normalized, normalized * pp], axis=1)
        error = float(np.max(abs(np.asarray(features) - expected)))
        assert error < 2e-4
        rows.append(dict(n=n, error=error, flops=ctx.summary_dict()["flops_used"]))
    p = Path("results/phase2/ablation_v30_102_g4_legal_features.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Legal per-neuron g4row normalization and phi product cost measured; does not replace g4 by K4sigma scalar. Live final-only candidate and corrected singlethread trace not completed; negative samecycleLOO retained.",
                scope="Charged feature component",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

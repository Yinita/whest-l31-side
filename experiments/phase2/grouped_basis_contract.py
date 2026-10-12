"""H26 fixedtotalrank groupedmechanism PSD basis equivalence, synthetic only."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(56)
    rows = []
    for n in (7, 11, 19):
        groups = [rng.normal(size=(n, 3)) for _ in range(3)]
        gram = sum(x @ x.T for x in groups)
        packed = np.concatenate(groups, axis=1)
        assert np.allclose(gram, packed @ packed.T)
        omega = rng.normal(size=(n, 4))
        action = sum(x @ (x.T @ omega) for x in groups)
        q, _ = np.linalg.qr(action)
        assert q.shape == (n, 4)
        error = float(np.max(abs(action - gram @ omega)))
        assert error < 1e-10
        rows.append(dict(n=n, total_output_rank=4, operator_error=error))
    path = Path("results/phase2/ablation_v30_056_grouped_basis_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Sum of threegroup Gramoperators equals concatenated operator and yields fixedtotalrankQ, no mandatory3xoutputrank. Actual Gaussian/K3/K4 decomposition of Rres and repeatconsistent attribution still absent; no candidate or cost pass.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

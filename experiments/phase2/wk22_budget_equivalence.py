"""O17 official budget-filtered path versus expanded lean composite WK22."""

import json
from pathlib import Path

import numpy as np
from birth_source_derivation import symhub


def main():
    rng = np.random.default_rng(79)
    rows = []
    for n in (4, 7):
        c = rng.normal(size=(n, n))
        c = (c + c.T) / 2
        np.fill_diagonal(c, 0)
        k = rng.normal(size=(n, n))
        np.fill_diagonal(k, 0)
        m = rng.normal(size=(n, n))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 0)
        w = rng.normal(size=(5, n))
        legs = [(1, 1, c, 1, "C"), (2, 1, k, 2, "K21"), (1, 2, k.T, 2, "K12"), (2, 2, m, 2, "WK22")]
        official = np.zeros((n, n, n))
        composite = np.zeros_like(official)
        admitted = []
        excluded = []
        for a, b, s, cost, name in legs:
            for ar, br, sr, costr, namer in legs:
                if name != "WK22" and namer != "WK22":
                    continue
                piece = symhub(3 * w[a, :, None] * s, (w[b + ar, :, None] * w[br, None, :] * sr).T)
                composite += piece
                if cost + costr <= 3:
                    official += piece
                    admitted.append([name, namer])
                else:
                    excluded.append([name, namer])
        manual = symhub(3 * w[1, :, None] * c, (w[3, :, None] * w[2, None, :] * m).T) + symhub(
            3 * w[2, :, None] * m, (w[3, :, None] * w[1, None, :] * c).T
        )
        error = float(np.max(abs(official - manual)))
        assert error < 1e-12
        rows.append(
            dict(
                n=n,
                admitted=admitted,
                excluded=excluded,
                budget_filtered_error=error,
                composite_minus_official_norm=float(np.linalg.norm(composite - official)),
            )
        )
    path = Path("results/phase2/ablation_v30_079_wk22_budget_equivalence.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Official augmented WK22 birth admits only C*WK22 and WK22*C at cost3. Old composite birth also includes heavy*WK22 and WK22*heavy at cost4. Mechanisms are unequal, so old no_wk22 negative report cannot reject the budget-filtered mechanism. Frozen-source formula audit only; no candidate score.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

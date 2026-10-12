"""H07/H10 birth exclusion: diagonal lowrank compensation is repeated-only."""

import itertools
import json
from pathlib import Path

import numpy as np
from birth_source_derivation import symhub


def main():
    rng = np.random.default_rng(401)
    rows = []
    for n in (4, 7, 11):
        x = rng.normal(size=(n, n))
        p = rng.normal(size=n)
        u = rng.normal(size=n)
        v = rng.normal(size=n)
        full = p[:, None] + p[None, :]
        diagonal = np.diag(2 * p)
        y = (u[:, None] * (full - diagonal) * v[None, :]).T
        low = (u[:, None] * full * v[None, :]).T
        exact = symhub(x, y)
        approx = symhub(x, low)
        correction = symhub(x, (u[:, None] * diagonal * v[None, :]).T)
        assert np.allclose(exact, approx - correction)
        distinct = np.array(
            [len(set(idx)) == 3 for idx in itertools.product(range(n), repeat=3)]
        ).reshape(n, n, n)
        error = float(np.max(abs((exact - approx)[distinct])))
        assert error < 1e-12
        # Subtract all repeated coordinates before any transport: tensors coincide.
        decomp = (approx - correction) * distinct
        assert np.allclose(decomp, exact * distinct)
        w = rng.normal(size=(5, n))
        transported = np.einsum("ai,bj,ck,ijk->abc", w, w, w, exact * distinct)
        alternate = np.einsum("ai,bj,ck,ijk->abc", w, w, w, approx * distinct)
        transport_error = float(np.max(abs(transported - alternate)))
        assert transport_error < 1e-10
        rows.append(
            dict(
                n=n,
                distinct_error=error,
                transported_error=transport_error,
                repeated_correction_norm=float(np.linalg.norm(correction)),
            )
        )
    path = Path("results/phase2/ablation_h07_h10_distinct_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Diagonal compensation affects only repeated birth coordinates of identity-middle hub. Rank2 expression is equivalent for distinct-only newborn if its own repeated content is subtracted exactly before transport. Using existing V29 repeat subtraction without new source repeat updates is wrong. Charged thin factor integration remains pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

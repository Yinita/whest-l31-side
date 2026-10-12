"""O03 exact consequence of deleting one K111 birth in K211 conversion only."""

import itertools
import json
from pathlib import Path

import numpy as np
from birth_source_derivation import symhub


def main():
    rng = np.random.default_rng(653)
    rows = []
    for n in (4, 7, 11):
        mu = rng.normal(size=n)
        birth = symhub(rng.normal(size=(n, n)), rng.normal(size=(n, n)))
        mask = np.fromfunction(lambda a, b, c: (a != b) & (a != c) & (b != c), (n, n, n))
        # If pK211 stays frozen but K111 birth is deleted, K211 gains this mismatch.
        mismatch = 2 * np.einsum("a,abc->bc", mu, birth * mask)
        explicit = np.zeros((n, n))
        for b, c in itertools.product(range(n), repeat=2):
            if b != c:
                explicit[b, c] = sum(
                    2 * mu[a] * birth[a, b, c] for a in range(n) if a not in (b, c)
                )
        error = float(np.max(abs(explicit - mismatch)))
        assert error < 1e-10
        rows.append(dict(n=n, mismatch_norm=float(np.linalg.norm(mismatch)), formula_error=error))
    p = Path("results/phase2/ablation_v30_065_birth_deletion_mismatch.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Deleting a K111 birth only from cumulant conversion while leaving pK211 production unchanged adds exact2 sum mu*birth mismatch. Coupled production/conversion deletion is required; no official truncation map or live candidate.",
                scope="Matched-source cancellation contract",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

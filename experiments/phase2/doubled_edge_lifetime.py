"""O15 early doubled-edge birth finite-lifetime synthetic signal diagnostic."""

import json
from pathlib import Path

import numpy as np
from birth_source_derivation import symhub


def main():
    rng = np.random.default_rng(77)
    rows = []
    for n in (4, 7, 11):
        raw = rng.normal(size=(n, n))
        cov = raw @ raw.T / n
        np.fill_diagonal(cov, 0)
        w1, w2, w3 = rng.normal(size=(3, n))
        left = 3 * w2[:, None] * (cov * cov / 2)
        right = (w3[:, None] * w1[None, :] * cov).T
        tensor = symhub(left, right)
        signals = []
        for step in range(3):
            weights = rng.normal(size=(n, n)) / np.sqrt(n)
            tensor = np.einsum("ai,bj,ck,ijk->abc", weights, weights, weights, tensor)
            signals.append(float(np.linalg.norm(np.einsum("iii->i", tensor))))
            gate = rng.uniform(0.1, 1, size=n)
            tensor *= gate[:, None, None] * gate[None, :, None] * gate[None, None, :]
        rows.append(dict(n=n, d3_norms=signals))
    path = Path("results/phase2/ablation_v30_077_doubled_edge_lifetime.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Synthetic doublededge path can create nonzeroearly transportedD3; does not measure actualnetwork signedscore gain. Previousfinalonly negative remains; fullhistory gate not unlocked without real earlybirth signal/cost and exact repeated compensation.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

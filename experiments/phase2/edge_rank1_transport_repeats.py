"""H29/H30 rank1 edge exact CP transport after own distinct subtraction."""

import json
from pathlib import Path

import numpy as np
from k111_edge_rank_contract import symcp


def main():
    rng = np.random.default_rng(593)
    rows = []
    for n, r in [(4, 2), (7, 3), (11, 4)]:
        a, b, c = rng.normal(size=(3, n, r))
        v = rng.normal(size=n)
        # Three symmetric types, coefficient1/3; encode into first leg.
        types = [
            (a * v[:, None] / 3, b * v[:, None], c),
            (a * v[:, None] / 3, b, c * v[:, None]),
            (a / 3, b * v[:, None], c * v[:, None]),
        ]
        represented = sum(symcp(*t) for t in types)
        mask = np.fromfunction(lambda i, j, k: (i != j) & (i != k) & (j != k), (n, n, n))
        repeats = represented * (~mask)
        w = rng.normal(size=(n, n)) / np.sqrt(n)
        direct = np.einsum("ai,bj,ck,ijk->abc", w, w, w, represented * mask, optimize=True)
        cp = sum(symcp(*(w @ f for f in t)) for t in types)
        rep = np.einsum("ai,bj,ck,ijk->abc", w, w, w, repeats, optimize=True)
        err = float(np.max(abs(cp - rep - direct)))
        assert err < 1e-10
        rows.append(dict(n=n, source_rank=r, cp_columns=3 * r, transport_error=err))
    out = Path("results/phase2/ablation_v30_059_060_edge_transport_repeats.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Rank1 edge 3R CP columns and exact own repeats survive linear transport. Lowrank covariance approximation, coverage audit against existing feedback, lawful weights and production timing remain unverified. One-layer deletion is model truncation.",
                scope="Synthetic complete represented-source transport; no candidate score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

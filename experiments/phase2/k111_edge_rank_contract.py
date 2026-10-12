"""O13/H29 fixed rank1 edge times CP K111 exact representation and exclusions."""

import itertools
import json
from pathlib import Path

import numpy as np


def symcp(a, b, c):
    return (
        sum(np.einsum("it,jt,kt->ijk", x, y, z) for x, y, z in itertools.permutations((a, b, c)))
        / 6
    )


def main():
    rng = np.random.default_rng(53)
    rows = []
    for n, rank in [(4, 2), (7, 3), (11, 4)]:
        a, b, c = rng.normal(size=(3, n, rank))
        vector = rng.normal(size=n)
        edge = vector[:, None] * vector[None, :]
        tensor = symcp(a, b, c)
        direct = tensor * (edge[:, :, None] + edge[:, None, :] + edge[None, :, :]) / 3
        terms = []
        for x, y, z in itertools.permutations((a, b, c)):
            for first, second in ((0, 1), (0, 2), (1, 2)):
                factors = [x.copy(), y.copy(), z.copy()]
                factors[first] *= vector[:, None]
                factors[second] *= vector[:, None]
                terms.append(np.einsum("it,jt,kt->ijk", *factors) / 18)
        represented = sum(terms)
        error = float(np.max(abs(represented - direct)))
        assert error < 1e-11
        offedge = edge.copy()
        np.fill_diagonal(offedge, 0)
        distinct = np.array(
            [len(set(i)) == 3 for i in itertools.product(range(n), repeat=3)]
        ).reshape(n, n, n)
        exact_off = tensor * (offedge[:, :, None] + offedge[:, None, :] + offedge[None, :, :]) / 3
        assert np.allclose(exact_off * distinct, represented * distinct)
        rows.append(
            dict(
                n=n,
                rank=rank,
                max_error=error,
                naive_columns=18 * rank,
                unique_symmetric_types=3,
                compressed_exact_cp_columns=3 * rank,
            )
        )
    path = Path("results/phase2/ablation_v30_075_rank1_edge_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Fixedrank1edge times CPsource produces3 symmetric CPtypes, eachrankR; diagonal edge difference vanishes only after exact birthdistinct exclusion. No livecovariance dominantmode/signal/approximationerror gate measured, so this does not accept rank1 proxy or complete mixeddiagram.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

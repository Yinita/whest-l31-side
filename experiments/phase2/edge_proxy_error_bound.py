"""O14 bound tensor error of covariance edge approximation for fixedK111."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(76)
    rows = []
    for n, rank in [(4, 2), (7, 3), (11, 4)]:
        a, b, c = rng.normal(size=(3, n, rank))
        t = (
            sum(
                np.einsum("it,jt,kt->ijk", x, y, z) for x, y, z in itertools.permutations((a, b, c))
            )
            / 6
        )
        raw = rng.normal(size=(n, n))
        edge = raw @ raw.T
        eigenvalues, vectors = np.linalg.eigh(edge)
        proxy = eigenvalues[-1] * np.outer(vectors[:, -1], vectors[:, -1])
        diff = edge - proxy
        error = t * (diff[:, :, None] + diff[:, None, :] + diff[None, :, :]) / 3
        bound = np.max(abs(diff)) * np.linalg.norm(t)
        observed = np.linalg.norm(error)
        assert observed <= bound + 1e-10
        rows.append(
            dict(
                n=n,
                edge_top_energy_fraction=float(eigenvalues[-1] ** 2 / np.sum(eigenvalues**2)),
                observed_tensor_error=float(observed),
                entrywise_norm_bound=float(bound),
            )
        )
    path = Path("results/phase2/ablation_v30_076_edge_proxy_error_bound.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="||T*(Eij+Eik+Ejk)/3||F <= ||E||max ||T||F",
                conclusion="Fixedrank1 edge approximation error bound valid; randomsynthetic covariance dominant energy not actualnetwork evidence. No true source loss/score or acceptance threshold measured.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

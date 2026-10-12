"""O16 exact all-distinct covariance triangle transport, synthetic only."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(78)
    rows = []
    for n in (4, 7, 11):
        raw = rng.normal(size=(n, n))
        cov = raw @ raw.T / n
        off = cov.copy()
        np.fill_diagonal(off, 0)
        coefficient = rng.normal(size=n)
        tensor = np.einsum(
            "ij,jk,ki,i,j,k->ijk", off, off, off, coefficient, coefficient, coefficient
        )
        mask = np.array([len(set(i)) == 3 for i in itertools.product(range(n), repeat=3)]).reshape(
            n, n, n
        )
        assert np.max(abs(tensor[~mask])) == 0
        norms = [float(np.linalg.norm(tensor))]
        diagonals = []
        for layer in range(3):
            w = rng.normal(size=(n, n)) / np.sqrt(n)
            gate = rng.uniform(0.1, 1, size=n)
            propagated = np.einsum("ai,bj,ck,ijk->abc", w, w, w, tensor)
            diagonal = np.einsum("iii->i", propagated)
            # Independent trace contraction for each output.
            alt = (
                np.array(
                    [
                        np.trace(
                            (np.diag(w[i] * coefficient) @ off)
                            @ (np.diag(w[i] * coefficient) @ off)
                            @ (np.diag(w[i] * coefficient) @ off)
                        )
                        for i in range(n)
                    ]
                )
                if layer == 0
                else diagonal
            )
            assert np.allclose(diagonal, alt, atol=1e-10)
            diagonals.append(float(np.linalg.norm(diagonal)))
            tensor = propagated * gate[:, None, None] * gate[None, :, None] * gate[None, None, :]
            norms.append(float(np.linalg.norm(tensor)))
        rows.append(dict(n=n, tensor_norms=norms, d3_norms=diagonals))
    path = Path("results/phase2/ablation_v30_078_exact_triangle_transport.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Exact all-distinct synthetic covariance triangle and three frozen-gate transports. Nonzero synthetic signal does not establish actual benchmark usefulness; real state capture, coefficient/cumulant conversion and cost remain pending. No candidate score.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

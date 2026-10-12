"""L25 exact CP source D3 contraction and mean-correction first-order adjoint."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(125)
    rows = []
    for n, rank in [(4, 2), (7, 3), (11, 4)]:
        a, b, c = rng.normal(size=(3, n, rank))
        w = rng.normal(size=(n, n))
        tensor = (
            sum(
                np.einsum("it,jt,kt->ijk", x, y, z) for x, y, z in itertools.permutations((a, b, c))
            )
            / 6
        )
        transported = np.einsum("ai,bj,ck,ijk->abc", w, w, w, tensor)
        direct = np.einsum("iii->i", transported)
        thin = np.sum((w @ a) * (w @ b) * (w @ c), axis=1)
        error = float(np.max(abs(direct - thin)))
        assert error < 1e-10
        phi = rng.uniform(size=n)
        alpha = rng.normal(size=n)
        sigma = rng.uniform(0.5, 2, size=n)
        # Gaussian derivative E[ReLU'''(N(mu,sigma²))] = -alpha phi(alpha)/sigma².
        derivative = -alpha * phi / (6 * sigma**2)
        adjoint = rng.normal(size=n)
        proxy = float(adjoint @ (derivative * thin))
        raw_tensor_score = float(
            np.einsum("a,ai,aj,ak,ijk->", adjoint * derivative, w, w, w, tensor)
        )
        assert abs(proxy - raw_tensor_score) < 1e-10
        rows.append(
            dict(n=n, rank=rank, d3_error=error, adjoint_score_error=abs(proxy - raw_tensor_score))
        )
    path = Path("results/phase2/ablation_v30_120_source_d3_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="D3_source=sum_r(WA)_ir(WB)_ir(WC)_ir; first-order local mean coefficient=-alpha*phi/(6sigma²)",
                conclusion="Exact CP source contraction and frozen mean-only score. Dense/Hub source representations, repeated compensation, indirect covariance response and ranking outcomes pending; reference-free proxy not full deletion sensitivity.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

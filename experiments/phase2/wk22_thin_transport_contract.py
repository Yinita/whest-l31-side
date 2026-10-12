"""Complete transported thin WK22 hub with exact repeated-coordinate subtraction."""

import itertools
import json
from pathlib import Path

import numpy as np
from birth_source_derivation import symhub


def cp_slices(a, b, c):
    d3 = np.sum(a * b * c, axis=1)
    d21 = ((a * b) @ c.T + (a * c) @ b.T + (b * c) @ a.T) / 3
    return d3, d21


def main():
    rng = np.random.default_rng(407)
    rows = []
    for n in (4, 7, 11):
        x = rng.normal(size=(n, n))
        np.fill_diagonal(x, 0)
        p, u, v = rng.normal(size=(3, n))
        left = np.stack([v, v * p], axis=1)
        right = np.stack([u * p, u], axis=1)
        y = left @ right.T
        # y[:,t] = sum_s left[:,s] right[t,s], hub middle identity[:,t].
        a = (x[:, :, None] * right[None, :, :]).reshape(n, 2 * n)
        b = np.repeat(np.eye(n), 2, axis=1)
        c = np.tile(left, (1, n))
        tensor = (
            sum(
                np.einsum("ir,jr,kr->ijk", aa, bb, cc)
                for aa, bb, cc in itertools.permutations((a, b, c))
            )
            / 6
        )
        full = symhub(x, y)
        assert np.allclose(tensor, full, atol=1e-11)
        d3, d21 = cp_slices(a, b, c)
        assert np.allclose(d21, np.array([[full[i, i, j] for j in range(n)] for i in range(n)]))
        repeat = np.zeros_like(full)
        for i, j, k in itertools.product(range(n), repeat=3):
            if i == j == k:
                repeat[i, j, k] = d3[i]
            elif i == j:
                repeat[i, j, k] = d21[i, k]
            elif i == k:
                repeat[i, j, k] = d21[i, j]
            elif j == k:
                repeat[i, j, k] = d21[j, i]
        w = rng.normal(size=(n, n))
        fast3, fast21 = cp_slices(w @ a, w @ b, w @ c)
        repout = np.einsum("ai,bj,ck,ijk->abc", w, w, w, repeat, optimize=True)
        exact = np.einsum("ai,bj,ck,ijk->abc", w, w, w, full - repeat, optimize=True)
        fast3 -= np.einsum("iii->i", repout)
        fast21 -= np.array([[repout[i, i, j] for j in range(n)] for i in range(n)])
        err3 = float(np.max(abs(fast3 - np.einsum("iii->i", exact))))
        err21 = float(
            np.max(abs(fast21 - np.array([[exact[i, i, j] for j in range(n)] for i in range(n)])))
        )
        assert max(err3, err21) < 1e-9
        rows.append(dict(n=n, cp_columns=2 * n, d3_error=err3, d21_error=err21))
    out = Path("results/phase2/ablation_h07_h10_thin_transport_contract.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Exact rank2 WK22 hub expands to 2n CP columns; complete repeated subtraction and transported D3/D21 verified. Rank2 matrix is not rank2 tensor. Compressed approximation still needs independent projection and charged score checks.",
                cost_at_1024=dict(
                    cp_columns=2048,
                    three_dense_leg_transport_flops=6 * 1024**2 * 2048,
                    three_d21_products_flops=6 * 1024**2 * 2048,
                ),
                scope="Offline correctness only; explicit repeat tensor is oracle diagnostic, excluded from candidate",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

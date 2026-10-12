"""O18 explicit symmetric CP B1 vs exact V29 dslice compensation formulas."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(8012)
    rows = []
    for n in (4, 7, 11):
        a, p, xt, yt = rng.normal(size=(4, n, n))
        w = rng.normal(size=n)
        x = 3 * a + xt
        y = a * w[None, :] + yt
        tensor = (
            sum(np.einsum("ir,jr,kr->ijk", *perm) for perm in itertools.permutations([x, p, y])) / 6
        )
        d3 = np.einsum("ir,ir,ir->i", x, p, y)
        d21 = ((x * p) @ y.T + (p * y) @ x.T + (x * y) @ p.T) / 3
        ap = a * p
        lp = a * a * w[None, :] + a * yt + xt * a * w[None, :] / 3 + xt * yt / 3
        la = 2 * ap * w[None, :] + xt * p * w[None, :] / 3 + p * yt
        feedback = (ap + xt * p / 3) @ yt.T + (ap * w[None, :] + p * yt) @ xt.T / 3
        grouped = la @ a.T + lp @ p.T + feedback
        assert np.allclose(d21, np.einsum("iij->ij", tensor))
        assert np.allclose(grouped, d21)
        assert np.allclose(d3, np.einsum("iii->i", tensor))
        np.fill_diagonal(a, 0)
        identity = np.eye(n)
        x = 3 * a + xt
        y = a * w[None, :] + yt
        birth = (
            sum(
                np.einsum("ir,jr,kr->ijk", *perm)
                for perm in itertools.permutations([x, identity, y])
            )
            / 6
        )
        birth21 = (np.diag(xt)[:, None] * y.T + x * y + np.diag(yt)[:, None] * x.T) / 3
        assert np.allclose(birth21, np.einsum("iij->ij", birth))
        assert np.allclose(np.diag(xt) * np.diag(yt), np.einsum("iii->i", birth))
        rows.append(
            dict(
                n=n,
                transport_d21_maxerror=float(np.max(abs(grouped - d21))),
                birth_d21_maxerror=float(np.max(abs(birth21 - np.einsum("iij->ij", birth)))),
            )
        )
    out = Path("results/phase2/ablation_v30_080_r02_b1_repeat_contract.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                rows=rows,
                terms={"base": 3, "linearY": 3, "linearX": 1, "feedback_cross": 1},
                scope="FullB1 symmetricCP transport and birth repeatedindex compensation verified; B2shadowed/WK22 omitted paths still lack officialisomorphism, O18partial",
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

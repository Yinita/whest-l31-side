"""H15 exact capped K211 from a radial rank2-plus-diagonal core."""

import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(451)
    rows = []
    for n in (4, 7, 11):
        g = rng.normal(size=n)
        metric = rng.normal(size=(n, n))
        metric = (metric + metric.T) / 2
        # Radial fourth cumulant lift = (Gij Mkl + ... six terms)/6.
        core = np.diag(g)
        tensor = (
            np.einsum("ij,kl->ijkl", core, metric)
            + np.einsum("ik,jl->ijkl", core, metric)
            + np.einsum("il,jk->ijkl", core, metric)
            + np.einsum("jk,il->ijkl", core, metric)
            + np.einsum("jl,ik->ijkl", core, metric)
            + np.einsum("kl,ij->ijkl", core, metric)
        ) / 6
        actual = np.zeros((n, n))
        for b, c in itertools.product(range(n), repeat=2):
            if b != c:
                actual[b, c] = sum(tensor[a, a, b, c] for a in range(n) if a != b and a != c)
        # For three distinct coordinates only Gaa Mbc remains.
        expected = (g.sum() - g[:, None] - g[None, :]) * metric / 6
        np.fill_diagonal(expected, 0)
        err = float(np.max(abs(actual - expected)))
        assert err < 1e-11
        rows.append(
            dict(
                n=n,
                capped_error=err,
                capped_rank=int(np.linalg.matrix_rank(actual)),
                core_rank=int(np.linalg.matrix_rank(core)),
            )
        )
    p = Path("results/phase2/ablation_v30_045_radial_capped_contract.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="For G=diag(g), K_aabc distinct = g_a M_bc/6; cap=(sum(g)-g_b-g_c) M_bc/6, zero diagonal",
                conclusion="Exact mixed-coordinate radial production slice, including a=b/c removal. Dense M may make capped matrix fullrank despite diagonal core; current nonlinear core selection and full official normalization map pending.",
                scope="Synthetic tensor contraction; no candidate score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

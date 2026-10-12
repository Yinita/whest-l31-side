"""Exact fourth-cumulant31/22 slices of Sym(G tensor M)."""

import json
from pathlib import Path

import numpy as np


def lift(g, m):
    return (
        sum(
            np.einsum(spec, g, m)
            for spec in [
                "ij,kl->ijkl",
                "ik,jl->ijkl",
                "il,jk->ijkl",
                "jk,il->ijkl",
                "jl,ik->ijkl",
                "kl,ij->ijkl",
            ]
        )
        / 6
    )


def main():
    rng = np.random.default_rng(49)
    rows = []
    for n in [4, 7, 11]:
        w = rng.normal(size=(n, n))
        m = w @ w.T
        g = rng.normal(size=(n, n))
        g = (g + g.T) / 2
        t = lift(g, m)
        k31 = (np.diag(g)[:, None] * m + np.diag(m)[:, None] * g) / 2
        k22 = (
            np.diag(g)[:, None] * np.diag(m)[None, :]
            + np.diag(m)[:, None] * np.diag(g)[None, :]
            + 4 * g * m
        ) / 6
        explicit31 = np.array([[t[i, i, i, j] for j in range(n)] for i in range(n)])
        explicit22 = np.array([[t[i, i, j, j] for j in range(n)] for i in range(n)])
        e31 = float(abs(k31 - explicit31).max())
        e22 = float(abs(k22 - explicit22).max())
        assert max(e31, e22) < 1e-11
        rows.append(dict(n=n, k31_error=e31, k22_error=e22))
    p = Path("results/phase2/ablation_h19_h20_metric_slices.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                scope="ActualWmetric fullsixsymterms; no Wick scaling/candidate yet",
                rows=rows,
                formulas={"31": "(Gii*Mij+Mii*Gij)/2", "22": "(Gii*Mjj+Mii*Gjj+4Gij*Mij)/6"},
                limitations=[
                    "ProxyG representation must be fixed explicitly",
                    "Wick31/Wick22 normalization and distinctmask still require code mapping",
                    "Computing full M adds2n^3FLOPs per layer",
                ],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

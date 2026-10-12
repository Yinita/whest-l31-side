"""Actual W tensor transport versus core+metric transport, offline only."""

import itertools
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
    rng = np.random.default_rng(95)
    rows = []
    for n, scale in itertools.product([3, 4, 7], [1, 2]):
        g = rng.normal(size=(n, n))
        g = (g + g.T) / 2
        w = rng.normal(size=(n, n)) / np.sqrt(n)
        before = lift(g, np.eye(n) * scale)
        direct = np.einsum("ai,bj,ck,dl,ijkl->abcd", w, w, w, w, before, optimize=True)
        gp = w @ g @ w.T
        metric = scale * w @ w.T
        represented = lift(gp, metric)
        error = float(abs(direct - represented).max())
        marginal = np.array([direct[i, i, i, i] for i in range(n)])
        diagonal = np.diag(gp) * np.diag(metric)
        diagerror = float(abs(marginal - diagonal).max())
        wrong = float(abs(direct - lift(gp, np.eye(n) * scale)).max())
        assert max(error, diagerror) < 1e-11
        rows.append(
            dict(
                n=n,
                birth_metric=scale,
                max_tensor_error=error,
                marginal_error=diagerror,
                reset_metric_error=wrong,
                trace_error=float(
                    abs(np.einsum("ijkk->ij", direct) - np.einsum("ijkk->ij", represented)).max()
                ),
            )
        )
    p = Path("results/phase2/ablation_v30_095_actual_transport.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                scope="Offline full n4 tensor actualW transport; identity birth metric and transported W W.T distinguished",
                rows=rows,
                formula="Sym(G tensor M) -> Sym((W G W.T) tensor (W M W.T)); K4_i=Gii*Mii",
                score_applicable=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print("maxerror", max(r["max_tensor_error"] for r in rows))


if __name__ == "__main__":
    main()

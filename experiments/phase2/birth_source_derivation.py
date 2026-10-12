"""Exact missing hub-factor birth contributions under arbitrary Wick factors."""

import itertools
import json
from pathlib import Path

import numpy as np


def symhub(x, y):
    identity = np.eye(len(x))
    return (
        sum(
            np.einsum("it,jt,kt->ijk", a, b, c)
            for a, b, c in itertools.permutations((x, identity, y))
        )
        / 6
    )


def main():
    rng = np.random.default_rng(37)
    rows = []
    for n in [4, 7]:
        c = rng.normal(size=(n, n))
        c = (c + c.T) / 2
        np.fill_diagonal(c, 0)
        d = rng.normal(size=(n, n))
        np.fill_diagonal(d, 0)
        wk22 = rng.normal(size=(n, n))
        wk22 = (wk22 + wk22.T) / 2
        np.fill_diagonal(wk22, 0)
        w1, w2, w3, w4 = rng.normal(size=(4, n))
        k21 = d / 2
        k12 = k21.T
        b1x = 3 * (w1[:, None] * c + w2[:, None] * k21)
        b1y = (
            w2[:, None] * w1[None, :] * c
            + w3[:, None] * w1[None, :] * k21
            + w2[:, None] * w2[None, :] * k12
            + w3[:, None] * w2[None, :] * wk22
        ).T
        b2x = 3 * (w1[:, None] * k12 + w2[:, None] * wk22)
        b2y = (
            w3[:, None] * w1[None, :] * c
            + w4[:, None] * w1[None, :] * k21
            + w3[:, None] * w2[None, :] * k12
            + w4[:, None] * w2[None, :] * wk22
        ).T
        tensor = symhub(b1x, b1y) + symhub(b2x, b2y)
        # Bilinearity identity: enumerate each X/Y product independently.
        xs = [3 * w1[:, None] * c, 3 * w2[:, None] * k21]
        ys = [
            (w2[:, None] * w1[None, :] * c).T,
            (w3[:, None] * w1[None, :] * k21).T,
            (w2[:, None] * w2[None, :] * k12).T,
            (w3[:, None] * w2[None, :] * wk22).T,
        ]
        x2s = [3 * w1[:, None] * k12, 3 * w2[:, None] * wk22]
        y2s = [
            (w3[:, None] * w1[None, :] * c).T,
            (w4[:, None] * w1[None, :] * k21).T,
            (w3[:, None] * w2[None, :] * k12).T,
            (w4[:, None] * w2[None, :] * wk22).T,
        ]
        pieces = [symhub(x, y) for x, y in itertools.product(xs, ys)] + [
            symhub(x, y) for x, y in itertools.product(x2s, y2s)
        ]
        error = float(abs(sum(pieces) - tensor).max())
        assert error < 1e-11
        # Every pair's exact repeated content; this will constrain restoration candidates.
        detail = []
        for i, (x, y) in enumerate(
            list(itertools.product(xs, ys)) + list(itertools.product(x2s, y2s))
        ):
            t = pieces[i]
            d3 = np.diag(x) * np.diag(y)
            d21 = (np.diag(x)[:, None] * y.T + x * y + np.diag(y)[:, None] * x.T) / 3
            err = float(
                abs(np.array([[t[a, a, b] for b in range(n)] for a in range(n)]) - d21).max()
            )
            detail.append(
                dict(
                    piece=i,
                    repeated_error=err,
                    d3_error=float(abs(np.einsum("iii->i", t) - d3).max()),
                )
            )
        rows.append(dict(n=n, full_expansion_error=error, details=detail))
    p = Path("results/phase2/ablation_h07_h12_birth_derivation.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                scope="All16 arbitrary lean B1/B2 hub products and repeated content; not factor cost or endtoend restored candidate",
                rows=rows,
                limits=[
                    "K21 matrix here exact arbitrary input; V29 lowrank feedback approximate",
                    "Complete official diagram coefficient map and lowrankWK22 representation gate still pending",
                ],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

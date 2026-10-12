"""Independent edge expansion of centered Gaussian K211 through degree two."""

import itertools
import json
from pathlib import Path

import numpy as np


def moment(derivatives, cov):
    total = 0.0
    edges = [(0, 1), (0, 2), (1, 2)]
    for powers in itertools.product(range(3), repeat=3):
        if sum(powers) > 2:
            continue
        degrees = [0, 0, 0]
        coefficient = 1.0
        for (i, j), power in zip(edges, powers):
            degrees[i] += power
            degrees[j] += power
            coefficient *= cov[i, j] ** power / (2 if power == 2 else 1)
        if max(degrees) > 2:
            continue
        total += coefficient * np.prod([derivatives[i][degrees[i]] for i in range(3)])
    return total


def main():
    rng = np.random.default_rng(4703)
    rows = []
    for n in (4, 7, 11):
        p = rng.uniform(0.05, 0.95, n)
        q = rng.uniform(0.01, 1, n)
        m = rng.uniform(0.1, 2, n)
        variance = rng.uniform(0.1, 2, n)
        c = rng.normal(size=(n, n))
        c = (c + c.T) / 2
        np.fill_diagonal(c, 0)
        a = 2 * m * (1 - p)
        b = 2 * p - 2 * m * q - 2 * p * p
        explicit = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                for i in range(n):
                    if i in (j, k):
                        continue
                    sub = c[np.ix_([i, j, k], [i, j, k])]
                    centered_square = [variance[i], a[i], 2 * p[i] - 2 * m[i] * q[i]]
                    value = moment([centered_square, [0, p[j], q[j]], [0, p[k], q[k]]], sub)
                    # Remove variance(a)*Cov(b,c), including degree-two edge term.
                    value -= variance[i] * (p[j] * p[k] * c[j, k] + q[j] * q[k] * c[j, k] ** 2 / 2)
                    value -= 2 * p[i] ** 2 * p[j] * p[k] * c[i, j] * c[i, k]
                    explicit[j, k] += value
        v = c @ a
        cap = p[:, None] * p[None, :] * ((c * b[None, :]) @ c)
        cap += c * (
            q[:, None] * p[None, :] * (v[:, None] - c * a[None, :])
            + q[None, :] * p[:, None] * (v[None, :] - c * a[:, None])
        )
        np.fill_diagonal(cap, 0)
        error = float(np.max(abs(cap - explicit)))
        assert error < 1e-11
        # Orthogonal complement to the existing off-diagonal metric direction.
        post_cov = c * p[:, None] * p[None, :]
        core = cap * 6 / (n + 4)
        coefficient = np.sum(core * post_cov) / np.sum(post_cov**2)
        residual = core - coefficient * post_cov
        orthogonality = float(abs(np.sum(residual * post_cov)))
        assert orthogonality < 1e-11
        w = rng.normal(size=(n, n))
        transported = w @ residual @ w.T
        direct = np.einsum("ia,ab,ib->i", w, residual, w)
        assert np.allclose(direct, np.diag(transported))
        rows.append(
            dict(
                n=n,
                cumulant_max_error=error,
                residual_metric_inner_product=orthogonality,
                transport_diagonal_max_error=float(np.max(abs(direct - np.diag(transported)))),
            )
        )
    out = Path("results/phase2/ablation_v30_047_r03_centering_projection_contract.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="Exact degree-two Gaussian capped cumulant, distinct indices, harmonic cA and offdiagonal projection; excludes nonGaussian/source contributions and higher covariance degrees",
                score_before=None,
                score_after=None,
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

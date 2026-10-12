"""L25 coupled Gaussian mean/variance Jacobian, independently finite-differenced."""

import json
from math import erf
from pathlib import Path

import numpy as np


def ndtr(x):
    return (1 + np.vectorize(erf)(x / np.sqrt(2))) / 2


def forward(mu, variance, w):
    m = w.T @ mu
    v = (w * w).T @ variance
    sigma = np.sqrt(v)
    alpha = m / sigma
    phi = np.exp(-alpha * alpha / 2) / np.sqrt(2 * np.pi)
    gate = ndtr(alpha)
    mean = sigma * phi + m * gate
    second = (m * m + v) * gate + m * sigma * phi
    return mean, second - mean * mean


def main():
    rng = np.random.default_rng(125)
    rows = []
    for n in (4, 7, 11):
        w = rng.normal(size=(n, n)) / np.sqrt(n)
        mu = rng.normal(size=n)
        var = rng.uniform(0.4, 1.5, size=n)
        dm, dv = rng.normal(size=(2, n))
        m = w.T @ mu
        v = (w * w).T @ var
        sigma = np.sqrt(v)
        alpha = m / sigma
        phi = np.exp(-alpha * alpha / 2) / np.sqrt(2 * np.pi)
        gate = ndtr(alpha)
        y, _ = forward(mu, var, w)
        # d E[relu²]/dm=2E[relu], d E[relu²]/dv=Phi.
        jm = gate
        jv = phi / (2 * sigma)
        km = 2 * y * (1 - gate)
        kv = gate - 2 * y * jv
        pre_dm = w.T @ dm
        pre_dv = (w * w).T @ dv
        tangent = np.concatenate([jm * pre_dm + jv * pre_dv, km * pre_dm + kv * pre_dv])
        eps = 1e-6
        plus = forward(mu + eps * dm, var + eps * dv, w)
        minus = forward(mu - eps * dm, var - eps * dv, w)
        fd = np.concatenate([(a - b) / (2 * eps) for a, b in zip(plus, minus)])
        error = float(np.max(abs(tangent - fd)))
        assert error < 1e-8
        test = rng.normal(size=2 * n)
        tm, tv = test[:n], test[n:]
        adj_mu = w @ (jm * tm + km * tv)
        adj_var = (w * w) @ (jv * tm + kv * tv)
        dual = float(abs(test @ tangent - adj_mu @ dm - adj_var @ dv))
        assert dual < 1e-10
        rows.append(dict(n=n, finite_difference_max_error=error, duality_error=dual))
    p = Path("results/phase2/ablation_v30_120_coupled_gaussian_jacobian.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Coupled mean/diagonal variance propagation adjoint verified; this goes beyond mean-only gates, but omits off-diagonal covariance and cumulants. Full source ranking is still incomplete.",
                scope="Synthetic derivative contract, no benchmark score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

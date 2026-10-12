"""Degree5/6 exact marginal cumulants for Gaussian ReLU, quadrature oracle."""

import json
from pathlib import Path

from mpmath import mp


def main():
    mp.dps = 60
    rows = []
    for alpha in [-4, -1, 0, 1, 4]:
        a = mp.mpf(alpha)
        phi = mp.exp(-a * a / 2) / mp.sqrt(2 * mp.pi)
        cdf = (1 + mp.erf(a / mp.sqrt(2))) / 2
        # truncated standard normal integrals I_j(lower=-alpha)
        lower = -a
        ints = [cdf, phi]
        for degree in range(2, 7):
            ints.append(lower ** (degree - 1) * phi + (degree - 1) * ints[degree - 2])
        moments = [mp.mpf(1)] + [
            sum(mp.binomial(k, j) * a ** (k - j) * ints[j] for j in range(k + 1))
            for k in range(1, 7)
        ]
        cumulants = [mp.mpf(0)]
        for k in range(1, 7):
            cumulants.append(
                moments[k]
                - sum(
                    mp.binomial(k - 1, j - 1) * cumulants[j] * moments[k - j] for j in range(1, k)
                )
            )
        errors = []
        for k in [5, 6]:
            direct = mp.quad(
                lambda z: max(a + z, mp.mpf(0)) ** k * mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi),
                [-a, mp.inf],
            )
            errors.append(float(abs(direct - moments[k])))
        rows.append(
            dict(
                alpha=alpha,
                kappa5=float(cumulants[5]),
                kappa6=float(cumulants[6]),
                moment5_quadrature_error=errors[0],
                moment6_quadrature_error=errors[1],
            )
        )
    assert (
        max(max(r["moment5_quadrature_error"], r["moment6_quadrature_error"]) for r in rows) < 1e-40
    )
    p = Path("results/phase2/ablation_degree5_6_marginal_derivation.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                scope="Exact Gaussian ReLU marginal cumulants only; not official propagated K5/K6 core closure",
                recurrence="m_k=sum binom(k,j)*alpha^(k-j)*I_j; I_j=(-alpha)^(j-1)*phi+(j-1)*I_(j-2); kappa_k=m_k-sum binom(k-1,j-1)kappa_j*m_(k-j)",
                rows=rows,
                limitation="Cannot insert these Gaussian newborn values as complete transported marginal K5/K6 without closure derivation",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

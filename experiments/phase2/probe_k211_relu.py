"""Offline check of Gaussian ReLU K211 star and capped contraction."""
import json
from pathlib import Path

import numpy as np
from probe_k31_relu import moments, pdf
from scipy.integrate import quad
from scipy.special import ndtr


def k211_conditional(mu, cj, ck):
    first, second, _ = moments(mu, 1.0)
    mj, mk = moments(0.3, 0.8)[0], moments(-0.2, 1.1)[0]
    sj, sk = np.sqrt(0.8**2 - cj**2), np.sqrt(1.1**2 - ck**2)
    def integrate(kind):
        def integrand(x):
            yi = max(mu + x, 0) - first
            j = moments(0.3 + cj * x, sj)[0] - mj
            k = moments(-0.2 + ck * x, sk)[0] - mk
            value = (yi**2 * j * k, j * k, yi * j, yi * k)[kind]
            return value * pdf(x)
        return sum(quad(integrand, a, b, epsabs=1e-14, epsrel=1e-10)[0] for a, b in ((-np.inf, -mu), (-mu, np.inf)))
    return integrate(0) - (second - first**2) * integrate(1) - 2 * integrate(2) * integrate(3)


def main():
    rows = []
    for mu in (-1.0, 0.0, 1.0, 2.0):
        first = moments(mu, 1.0)[0]
        p = ndtr(mu)
        b = 2 * (p * (1 - p) - first * pdf(mu))
        analytic = b * ndtr(0.3 / 0.8) * ndtr(-0.2 / 1.1)
        eps = 2e-4
        numeric = sum(signj * signk * k211_conditional(mu, signj * eps, signk * eps) for signj in (-1, 1) for signk in (-1, 1)) / (4 * eps**2)
        error = abs(analytic - numeric)
        assert error < 1e-6
        rows.append({'mu': mu, 'analytic_star_coefficient': float(analytic), 'quadrature_mixed_derivative': numeric, 'abs_error': error})
    rng = np.random.default_rng(0)
    n = 7
    raw = rng.normal(size=(n, n))
    c = (raw + raw.T) / 2
    np.fill_diagonal(c, 0)
    a, b, p, w2 = rng.normal(size=(4, n))
    v = c @ a
    fast = p[:, None] * p[None, :] * ((c * b[None, :]) @ c)
    fast += c * (w2[:, None] * p[None, :] * (v[:, None] - c * a[None, :]) + w2[None, :] * p[:, None] * (v[None, :] - c * a[:, None]))
    np.fill_diagonal(fast, 0)
    slow = np.zeros_like(c)
    for j in range(n):
        for k in range(n):
            if j != k:
                slow[j, k] = sum(b[i] * p[j] * p[k] * c[i, j] * c[i, k] + a[i] * w2[j] * p[k] * c[i, j] * c[j, k] + a[i] * w2[k] * p[j] * c[i, k] * c[j, k] for i in range(n) if i not in (j, k))
    contraction_error = float(np.max(np.abs(fast - slow)))
    assert contraction_error < 1e-12
    report = {'mode': 'offline algebra only, not benchmark improvement', 'rows': rows, 'capped_contraction_max_error': contraction_error,
              'formula': 'K_iijk=B_i*Phi_j*Phi_k*C_ij*C_ik + A_i*w2_j*Phi_k*C_ij*C_jk + A_i*w2_k*Phi_j*C_ik*C_jk + O(C^3)',
              'definitions': 'A_i=2*m1_i*(1-Phi_i); B_i=2*(Phi_i*(1-Phi_i)-m1_i*phi_i/sigma_i); w2_i=phi_i/sigma_i',
              'derivation': 'Connected second-order Gaussian covariance expansion of log moment generating function; marginal ReLU derivatives',
              'source': 'https://arxiv.org/html/2605.05179v2#S3.SS2',
              'limitations': ['Quadrature validates star centered on repeated index; other star coefficients derived analytically, not independently integrated', 'Matrix contraction verified against explicit sums', 'Weak-correlation Gaussian only; propagated non-Gaussian cumulants excluded']}
    out = Path('results/phase2/k211_relu_algebra.json')
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print('Four star checks and capped contraction passed; max quadrature error', max(x['abs_error'] for x in rows))


if __name__ == '__main__':
    main()

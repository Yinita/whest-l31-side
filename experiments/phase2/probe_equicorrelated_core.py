"""Exact one-dimensional Gaussian-mixture integrals for a ReLU K4 core."""
import json
from pathlib import Path

import numpy as np
from probe_k31_relu import moments, pdf
from scipy.integrate import quad
from scipy.special import ndtr


def exact_slices(rho):
    mean, second, third = moments(0, 1)
    variance = second - mean**2
    def integrand(x, kind):
        a, b, c = moments(np.sqrt(rho) * x, np.sqrt(1 - rho))
        u1 = a - mean
        u2 = b - 2 * mean * a + mean**2
        u3 = c - 3 * mean * b + 3 * mean**2 * a - mean**3
        return (u1*u1, u3*u1, u2*u1*u1)[kind] * pdf(x)
    values = [quad(lambda x: integrand(x, k), -np.inf, np.inf, epsabs=1e-13, epsrel=1e-11)[0] for k in range(3)]
    cov, m31, m211 = values
    return cov, m31 - 3 * variance * cov, m211 - variance * cov - 2 * cov**2


def main():
    mean, second, _ = moments(0, 1)
    p = ndtr(0)
    a = 2 * mean * (1 - p)
    b = 2 * (p*(1-p) - mean*pdf(0))
    k31_coefficient = 3 * (1-p) * (second - 2*mean**2) * p
    k211_coefficient = b*p*p + 2*a*pdf(0)*p
    rows = []
    for rho in (0.0001, 0.001, 0.01, 0.05):
        cov, k31, k211 = exact_slices(rho)
        rows.append({'rho': rho, 'post_relu_covariance': cov, 'exact_k31': k31, 'proxy_k31': k31_coefficient*rho,
                     'exact_k211': k211, 'proxy_k211': k211_coefficient*rho**2,
                     'k31_relative_error': abs(k31_coefficient*rho-k31)/abs(k31),
                     'k211_relative_error': abs(k211_coefficient*rho**2-k211)/abs(k211),
                     'n1024_core_lambda_exact': 6*(2*k31+1022*k211)/(1028*cov),
                     'n1024_core_lambda_proxy': 6*(2*k31_coefficient*rho+1022*k211_coefficient*rho**2)/(1028*cov)})
    assert rows[0]['k31_relative_error'] < 0.002
    assert rows[0]['k211_relative_error'] < 0.002
    result = {'mode': 'offline exactly integrable equicorrelated Gaussian model, not public-network score',
              'method': 'Condition on shared standard normal, integrate independent conditional ReLU moments; cumulant definitions applied after integration',
              'source': 'https://arxiv.org/html/2605.05179v2#S3.SS4',
              'limitations': ['Positive equal correlations, zero means, unit input variances', 'Does not model random signed Wishart correlations or propagated non-Gaussian sources'],
              'rows': rows}
    path = Path('results/phase2/equicorrelated_core_algebra.json')
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print([(x['rho'], round(x['k31_relative_error'], 6), round(x['k211_relative_error'], 6)) for x in rows])


if __name__ == '__main__':
    main()

"""Offline quadrature check of a first-order Gaussian ReLU K31 proxy."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def pdf(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


def moments(mu, sigma):
    a = mu / sigma
    p = ndtr(a)
    first = mu * p + sigma * pdf(a)
    second = (mu * mu + sigma * sigma) * p + mu * sigma * pdf(a)
    third = (mu**3 + 3 * mu * sigma**2) * p + (mu**2 * sigma + 2 * sigma**3) * pdf(a)
    return first, second, third


def k31_quad(mu, sigma, other_mu, other_sigma, rho):
    first, second, third = moments(mu, sigma)
    other_first = moments(other_mu, other_sigma)[0]
    cond_sigma = other_sigma * np.sqrt(1 - rho * rho)
    mixed = []
    for power in (1, 2, 3):
        def integrand(x):
            conditional_mean = other_mu + rho * other_sigma * x
            conditional_relu = moments(conditional_mean, cond_sigma)[0]
            return (mu + sigma * x)**power * conditional_relu * pdf(x)
        mixed.append(quad(integrand, -mu / sigma, np.inf, epsabs=1e-12, epsrel=1e-12)[0])
    m11, m21, m31 = mixed
    covariance = m11 - first * other_first
    centered = (m31 - 3 * first * m21 + 3 * first**2 * m11
                - other_first * third + 3 * first * second * other_first
                - 3 * first**3 * other_first)
    return centered - 3 * (second - first**2) * covariance


def main():
    rows = []
    for mu in (-1.0, 0.0, 1.0, 2.0):
        sigma, other_mu, other_sigma = 1.0, 0.3, 0.8
        first, second, _ = moments(mu, sigma)
        analytic = 3 * (1 - ndtr(mu / sigma)) * (second - 2 * first**2) * ndtr(other_mu / other_sigma)
        epsilon = 1e-4
        derivative = (k31_quad(mu, sigma, other_mu, other_sigma, epsilon)
                      - k31_quad(mu, sigma, other_mu, other_sigma, -epsilon)) / (2 * epsilon * sigma * other_sigma)
        error = abs(analytic - derivative)
        assert error < 2e-6
        rows.append({'mu': mu, 'analytic_dK31_dCov': float(analytic), 'quadrature_derivative': derivative, 'abs_error': error})
    report = {'mode': 'offline synthetic Gaussian diagnostic; not candidate or benchmark improvement',
              'formula': 'K31_ij = 3*(1-Phi_i)*(E[Y_i^2]-2*E[Y_i]^2)*Phi_j*C_ij + O(C_ij^2)',
              'derivation': 'Gaussian covariance differentiation plus K31 centered-moment identity; ReLU marginal moments from paper S3.4',
              'source': 'https://arxiv.org/html/2605.05179v2#S3.SS4',
              'limitations': ['Joint Gaussian preactivations only', 'First order around zero cross covariance', 'Does not include propagated K3/K4 or K211', 'Cannot replace complete matrix core without feed/use representation changes'],
              'rows': rows}
    path = Path('results/phase2/k31_relu_algebra.json')
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Four quadrature checks passed, max error', max(x['abs_error'] for x in rows))


if __name__ == '__main__':
    main()

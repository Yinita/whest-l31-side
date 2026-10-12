"""Independent Gaussian pair quadrature for the cubic Hermite tail.

Offline SciPy diagnostic only. It neither reads competition targets nor changes
the estimator. pK(p,q) here is Cov(ReLU(X)**p, ReLU(Y)**q), p,q in {1,2}.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def moment(mu, sigma, power):
    a = mu / sigma
    if power == 1:
        return mu * ndtr(a) + sigma * phi(a)
    return (mu * mu + sigma * sigma) * ndtr(a) + mu * sigma * phi(a)


def wick(mu, sigma, power, order):
    a = mu / sigma
    if power == 1:
        return {1: ndtr(a), 2: phi(a) / sigma,
                3: -a * phi(a) / sigma**2}[order]
    return {1: 2 * moment(mu, sigma, 1), 2: 2 * ndtr(a),
            3: 2 * phi(a) / sigma}[order]


def covariance_quad(mi, si, mj, sj, p, q, rho):
    cond_sd = sj * math.sqrt(1 - rho * rho)
    def integrand(z):
        return (mi + si * z)**p * moment(mj + sj * rho * z, cond_sd, q) * phi(z)
    mixed, error = quad(integrand, -mi / si, np.inf,
                        epsabs=2e-13, epsrel=2e-13, limit=250)
    return mixed - moment(mi, si, p) * moment(mj, sj, q), error


def expansion(mi, si, mj, sj, p, q, rho, degree):
    c = rho * si * sj
    return sum(c**k * wick(mi, si, p, k) * wick(mj, sj, q, k)
               / math.factorial(k) for k in range(1, degree + 1))


def main():
    output = Path('results/phase2/gaussian_pair_tail_validation.json')
    if output.exists():
        raise FileExistsError(output)
    analytic_rows, cubic_rows, approximation_rows = [], [], []
    # Exact arc-cosine kernel; odd terms beyond rho are absent at zero means.
    for rho in (-0.5, -0.1, 0.1, 0.5):
        exact = (math.sqrt(1 - rho*rho) + rho*(math.pi/2 + math.asin(rho)) - 1) / (2*math.pi)
        measured, quad_error = covariance_quad(0, 1, 0, 1, 1, 1, rho)
        assert abs(measured - exact) < 3e-12
        assert expansion(0, 1, 0, 1, 1, 1, rho, 3) == expansion(0, 1, 0, 1, 1, 1, rho, 2)
        analytic_rows.append(dict(rho=rho, exact=exact, quadrature=measured,
                                  abs_error=abs(measured-exact), quadrature_error_estimate=quad_error,
                                  cubic_term=0.0))
    cases = [(-0.7, 0.8, 0.4, 1.1), (0.5, 1.2, -0.9, 0.7), (1.1, 0.9, 0.8, 1.3)]
    for mi, si, mj, sj in cases:
        for p, q in [(1,1), (2,1), (2,2)]:
            # Antisymmetry eliminates even powers; Richardson extrapolation
            # eliminates the leading fifth-order contamination to the cubic.
            def cubic_from_quad(rho):
                plus, _ = covariance_quad(mi,si,mj,sj,p,q,rho)
                minus, _ = covariance_quad(mi,si,mj,sj,p,q,-rho)
                c = rho*si*sj
                linear = c*wick(mi,si,p,1)*wick(mj,sj,q,1)
                return (0.5*(plus-minus)-linear)/c**3
            h = 0.025
            coarse, fine = cubic_from_quad(h), cubic_from_quad(h/2)
            independent = (4*fine-coarse)/3
            expected = wick(mi,si,p,3)*wick(mj,sj,q,3)/6
            assert abs(independent-expected) < 2e-7
            cubic_rows.append(dict(mu_i=mi,sigma_i=si,mu_j=mj,sigma_j=sj,
                                   powers=[p,q], expected_covariance_cubic_coefficient=expected,
                                   quadrature_cubic_coefficient=independent,
                                   abs_error=abs(independent-expected)))
            errors = []
            for rho in (0.08,0.04,0.02):
                value, quad_error = covariance_quad(mi,si,mj,sj,p,q,rho)
                old = expansion(mi,si,mj,sj,p,q,rho,2)
                new = expansion(mi,si,mj,sj,p,q,rho,3)
                errors.append((abs(value-old),abs(value-new)))
                approximation_rows.append(dict(mu_i=mi,sigma_i=si,mu_j=mj,sigma_j=sj,
                                               powers=[p,q],rho=rho,quadrature=value,
                                               degree2=old,degree3=new,
                                               degree2_abs_error=abs(value-old),
                                               degree3_abs_error=abs(value-new),
                                               quadrature_error_estimate=quad_error))
            assert errors[-1][1] < errors[-1][0]
            # For these fixed nonzero-mean cases, q3 changes O(C^3) error to O(C^4).
            assert errors[1][1]/errors[2][1] > 10
    report = dict(mode='offline independent synthetic Gaussian quadrature; not a benchmark gain',
                  formula='Delta pK(p,q) = C_ij**3/6 * E[d^3 ReLU(X_i)^p] * E[d^3 ReLU(X_j)^q]',
                  justification='Gaussian covariance (Mehler/Price) expansion, no source K3/K4 assumed',
                  wick3={'p=1':'-alpha*phi(alpha)/sigma**2','p=2':'2*phi(alpha)/sigma'},
                  source='https://arxiv.org/html/2605.05179v2#S3.SS4',
                  zero_mean_relu_arc_cosine_checks=analytic_rows,
                  independent_cubic_coefficient_checks=cubic_rows,
                  weak_correlation_approximation_checks=approximation_rows,
                  limitations=['Joint Gaussian only; does not restore all grade-4 or source interactions',
                               'Asymptotic near zero correlation; not guaranteed beneficial for large correlations',
                               'The covariance expansion can extend a deliberately truncated table; competition validation required'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 4 analytic arc-cosine checks, 9 independent cubic coefficients, 27 quadrature comparisons')
    print('Max cubic coefficient error:',max(r['abs_error'] for r in cubic_rows))


if __name__ == '__main__':
    main()

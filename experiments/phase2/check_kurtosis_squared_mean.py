"""Offline independent checks for a marginal kurtosis-squared mean term.

Polynomial checks use exact moments of Gaussian + finite symmetric noise and
recover the coefficient from their residual; they do not generate cumulants
using the candidate's Edgeworth expression. ReLU checks integrate Hermite H8.
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_hermitenorm


def double_factorial(n):
    return math.prod(range(n, 0, -2))


def gaussian_raw_moment(mu, variance, degree):
    return sum(Fraction(math.comb(degree, 2*j) * double_factorial(2*j-1))
               * mu**(degree-2*j) * variance**j for j in range(degree//2+1))


def cumulants_from_moments(moments):
    cumulants = [Fraction(0)] * len(moments)
    for degree in range(1, len(moments)):
        cumulants[degree] = moments[degree] - sum(
            Fraction(math.comb(degree-1, k-1)) * cumulants[k] * moments[degree-k]
            for k in range(1, degree))
    return cumulants


def polynomial_check(mu, gaussian_variance, support):
    moments = [sum(probability * point**d for point, probability in support)
               for d in range(9)]
    kappas = cumulants_from_moments(moments)
    assert all(kappas[d] == 0 for d in (1,3,5,7))
    variance = gaussian_variance + kappas[2]
    exact = sum(probability * gaussian_raw_moment(mu+point, gaussian_variance, 8)
                for point, probability in support)
    gaussian = gaussian_raw_moment(mu, variance, 8)
    # First-order cumulant terms have independently known Taylor coefficients.
    linear = sum(kappas[d] * Fraction(math.comb(8,d))
                 * gaussian_raw_moment(mu, variance, 8-d) for d in (4,6,8))
    residual = exact - gaussian - linear
    derivative8 = math.factorial(8)
    recovered = residual / (kappas[4]**2 * derivative8)
    assert recovered == Fraction(1,1152)
    return dict(mean=float(mu), gaussian_variance=float(gaussian_variance),
                support=[[float(x),float(p)] for x,p in support],
                cumulants={str(d):float(kappas[d]) for d in (2,4,6,8)},
                exact_eighth_moment=float(exact), gaussian_eighth_moment=float(gaussian),
                first_order_cumulant_correction=float(linear),
                residual_kurtosis_squared_contribution=float(residual),
                recovered_coefficient=str(recovered), exact_fraction_check=True)


def main():
    output = Path('results/phase2/kurtosis_squared_mean_validation.json')
    if output.exists():
        raise FileExistsError(output)
    polynomial_rows = []
    for mu, variance in [(Fraction(0),Fraction(1)),(Fraction(1,3),Fraction(4,5)),
                          (Fraction(-2,5),Fraction(7,6))]:
        for support in [[(Fraction(-1,2),Fraction(1,2)),(Fraction(1,2),Fraction(1,2))],
                        [(Fraction(x,3),Fraction(1,4)) for x in (-2,-1,1,2)]]:
            polynomial_rows.append(polynomial_check(mu,variance,support))
    relu_rows = []
    for alpha, sigma in [(-1.3,0.8),(-0.4,1.2),(0.0,1.0),(0.7,0.9),(1.8,1.4)]:
        mu = alpha*sigma
        def integrand(z):
            return (mu+sigma*z)*eval_hermitenorm(8,z)*math.exp(-z*z/2)/math.sqrt(2*math.pi)
        integral, error = quad(integrand,-alpha,np.inf,epsabs=5e-12,epsrel=5e-12,limit=250)
        recovered = integral/sigma**8
        a2=alpha*alpha
        expected=(a2**3-15*a2*a2+45*a2-15)*math.exp(-a2/2)/math.sqrt(2*math.pi)/sigma**7
        assert abs(recovered-expected)<5e-10
        relu_rows.append(dict(alpha=alpha,sigma=sigma,hermite_integral_w8=recovered,
                              signed_formula_w8=expected,abs_error=abs(recovered-expected),
                              integration_error_estimate=error/sigma**8))
    report = dict(mode='offline synthetic correctness only; no benchmark or candidate improvement',
                  mean_term='kappa4**2 * w(8,1) / 1152',
                  coefficient='1 / [2! * (4!)**2]',
                  w8='phi(alpha) * [alpha**6 - 15*alpha**4 + 45*alpha**2 - 15] / sigma**7',
                  primary_source='https://arxiv.org/html/2605.05179v2',
                  paper_sections=['S.3.2 diagram combinatorics','S.3.4 Proposition S.3.8',
                                  'S.4.1 equations 13 and 14','S.6.2 asymptotic expansion limitation'],
                  formal_truncation={'hermite_order':8,'K3_max_hermite_order':5,
                                     'vector_partition_blocks':[[4],[4]],'equation14_grade':5,
                                     'excluded_by_formal_K3':True},
                  exact_fraction_polynomial_checks=polynomial_rows,
                  independent_relu_hermite_integrals=relu_rows,
                  limitations=['Selective beyond-K3 term, not a complete K4/K5 estimator',
                               'Using existing approximate g4row does not improve its accuracy',
                               'ReLU Edgeworth expansion may be asymptotic or divergent at fixed width',
                               'Adding the mean term changes moment-to-cumulant centering downstream unless explicitly isolated'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 6 exact-fraction Gaussian+finite-noise polynomial coefficients and 5 independent ReLU H8 integrals')
    print('Max ReLU w8 integral error:', max(r['abs_error'] for r in relu_rows))


if __name__ == '__main__':
    main()

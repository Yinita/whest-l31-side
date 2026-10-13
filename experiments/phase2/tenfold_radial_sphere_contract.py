"""Exact homogeneity/radial tensor and spherical K4/K6 contracts, no targets.

This prepares a new input-law candidate. It does not run the parent estimator,
does not measure adjusted scores, and does not assert a production budget pass.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def radius_moment(n, degree):
    return math.exp(math.lgamma((n+degree)/2)-math.lgamma(n/2)
                    +degree/2*(math.log(2)-math.log(n)))


def gaussian_relu(mu, sigma):
    alpha = mu/sigma
    phi = math.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
    cdf = (1+math.erf(alpha/math.sqrt(2)))/2
    return sigma*phi+mu*cdf


def spherical_relu(n, mu, sigma, nodes):
    upper = math.sqrt(n)
    lower = max(-upper, -mu/sigma)
    if lower >= upper:
        return 0.0
    roots, weights = np.polynomial.legendre.leggauss(nodes)
    x = lower+(upper-lower)*(roots+1)/2
    norm = math.exp(math.lgamma(n/2)-math.lgamma((n-1)/2))/math.sqrt(math.pi*n)
    density = norm*np.maximum(1-x*x/n, 0)**((n-3)/2)
    return float((upper-lower)/2*np.sum(weights*density*(mu+sigma*x)))


def edgeworth(n, mu, sigma, second_order=False):
    alpha = mu/sigma
    phi = math.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
    k4 = -6*sigma**4/(n+2)
    out = gaussian_relu(mu, sigma)+k4*(alpha*alpha-1)*phi/(24*sigma**3)
    if second_order:
        k6 = 240*sigma**6/((n+2)*(n+4))
        out += k6*(alpha**4-6*alpha*alpha+3)*phi/(720*sigma**5)
        out += k4*k4*(alpha**6-15*alpha**4+45*alpha*alpha-15)*phi/(1152*sigma**7)
    return out


def centered_moments(y, weights):
    mean = weights@y
    dy = y-mean
    cov = np.einsum("r,ri,rj->ij", weights, dy, dy)
    k3 = np.einsum("r,ri,rj,rk->ijk", weights, dy, dy, dy)
    k4 = np.einsum("r,ri,rj,rk,rl->ijkl", weights, dy, dy, dy, dy)
    k4 -= (np.einsum("ij,kl->ijkl", cov, cov)+np.einsum("ik,jl->ijkl", cov, cov)
           +np.einsum("il,jk->ijkl", cov, cov))
    return mean, cov, k3, k4


def radial_tensor_formula(mean, cov, k3, k4, n):
    d = len(mean)
    a1, a3, a4 = [radius_moment(n, p) for p in (1, 3, 4)]
    g1 = a1*mean
    g2 = cov+(1-a1*a1)*np.outer(mean, mean)
    g3, g4 = np.zeros_like(k3), np.zeros_like(k4)
    for i, j, k in itertools.product(range(d), repeat=3):
        cu = cov[i,j]*mean[k]+cov[i,k]*mean[j]+cov[j,k]*mean[i]
        g3[i,j,k] = a3*k3[i,j,k]+(a3-a1)*cu+(a3-3*a1+2*a1**3)*mean[i]*mean[j]*mean[k]
    for i, j, k, ell in itertools.product(range(d), repeat=4):
        km = (k3[i,j,k]*mean[ell]+k3[i,j,ell]*mean[k]
              +k3[i,k,ell]*mean[j]+k3[j,k,ell]*mean[i])
        cc = cov[i,j]*cov[k,ell]+cov[i,k]*cov[j,ell]+cov[i,ell]*cov[j,k]
        cmm = (cov[i,j]*mean[k]*mean[ell]+cov[i,k]*mean[j]*mean[ell]
               +cov[i,ell]*mean[j]*mean[k]+cov[j,k]*mean[i]*mean[ell]
               +cov[j,ell]*mean[i]*mean[k]+cov[k,ell]*mean[i]*mean[j])
        g4[i,j,k,ell] = (a4*k4[i,j,k,ell]+(a4-a1*a3)*km+(a4-1)*cc
                         +(a4-2*a1*a3-1+2*a1*a1)*cmm
                         +(a4-4*a1*a3-3+12*a1*a1-6*a1**4)*mean[i]*mean[j]*mean[k]*mean[ell])
    return g1, g2, g3, g4


def tensor_contract():
    # The finite angular law is synthetic. Radial independence identities apply
    # to any angular law; we do not call this a standard Gaussian expectation.
    rng = np.random.default_rng(261013901)
    angles = rng.normal(size=(7, 4))
    angles *= 2/np.linalg.norm(angles, axis=1)[:, None]
    weights = np.arange(1, 8, dtype=float)
    weights /= weights.sum()
    w0, w1 = rng.normal(size=(4, 5)), rng.normal(size=(5, 3))
    y = np.maximum(np.maximum(angles@w0, 0)@w1, 0)
    moments = centered_moments(y, weights)
    formula = radial_tensor_formula(*moments, 4)
    # Independent scalar chi4 quadrature, then direct centered output tensors.
    roots, quadrature = np.polynomial.legendre.leggauss(256)
    radius = 6*(roots+1)
    radial_weights = 6*quadrature*(radius**3/2)*np.exp(-radius*radius/2)
    assert abs(radial_weights.sum()-1) < 1e-13
    mixed_y = (radius[:, None, None]/2*y[None, :, :]).reshape(-1, 3)
    mixed_weights = (radial_weights[:, None]*weights[None, :]).ravel()
    direct = centered_moments(mixed_y, mixed_weights)
    errors = [float(np.max(np.abs(a-b))) for a,b in zip(formula, direct)]
    relative = [float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-20)) for a,b in zip(formula, direct)]
    assert max(relative) < 5e-12
    scaled = np.maximum(np.maximum((angles*1.37)@w0, 0)@w1, 0)
    assert np.max(np.abs(scaled-1.37*y)) < 1e-13
    return {"max_absolute_errors_mean_cov_K3_K4": errors,
            "relative_errors_mean_cov_K3_K4": relative,
            "angular_points": 7, "radial_quadrature_nodes": 256,
            "scope": "Synthetic nonGaussian finite angular law; direct independent chi4 radial quadrature, no sampled reference"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    assert not out.exists()
    rows = []
    for n in (8, 32, 64, 256, 1024):
        a1 = radius_moment(n, 1)
        exact_first_relu = math.sqrt(n)*math.exp(math.lgamma(n/2)-math.lgamma((n+1)/2))/(2*math.sqrt(math.pi))
        assert abs(a1*exact_first_relu-1/math.sqrt(2*math.pi)) < 4e-13
        for alpha in (-2, -1, 0, 1, 2):
            sigma, mu = 1.3, 1.3*alpha
            exact_lo = spherical_relu(n, mu, sigma, 192)
            exact_hi = spherical_relu(n, mu, sigma, 384)
            assert abs(exact_hi-exact_lo) < 2e-9
            gaussian = gaussian_relu(mu, sigma)
            ew4, ew6 = edgeworth(n, mu, sigma), edgeworth(n, mu, sigma, True)
            rows.append({"input_width": n, "alpha": alpha, "sigma": sigma,
                         "sphere_relu_mean": exact_hi,
                         "quadrature_192_384_delta": abs(exact_hi-exact_lo),
                         "before_Gaussian_mean_error": gaussian-exact_hi,
                         "after_K4_mean_error": ew4-exact_hi,
                         "after_K4_K6_K4square_mean_error": ew6-exact_hi,
                         "radial_factor": a1,
                         "restored_Gaussian_firstrelu_error_if_zero_alpha": a1*ew4-gaussian if alpha == 0 else None,
                         "nonzeroalpha_caveat": "Homogeneity restoration applies to complete bias-free network, not a fixed affine mu+sigma*sphere scalar when mu!=0"})
    report = {"version": "tenfold-radial-sphere-contract-r01", "complete": True,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "tensor_contract": tensor_contract(), "firstrelu_rows": rows,
              "input_sphere_cumulants": {"K2": "I", "K3": "0",
                  "K4_abcd": "-2/(n+2)*(delta_ab*delta_cd+delta_ac*delta_bd+delta_ad*delta_bc)",
                  "marginal_K4": "-6*sigma**4/(n+2)",
                  "marginal_K6": "240*sigma**6/((n+2)*(n+4))"},
              "before": "Gaussian K3 chain with implicit zero initial K4",
              "after_proposed": "One spherical input cumulant chain plus exact radial homogeneity restoration; not implemented production",
              "no_adjusted_score": True,
              "limitations": ["Exact radial formulas do not make spherical approximate cumulant chain exact",
                              "Nonzero initial K4 needs full firstlayer USE and K4toK3 feed with covariance metric; existing mean-only patch insufficient",
                              "K6/K4square scalar gate diagnostic is not full mixed tensor K6 closure",
                              "No competition flopscope/RPC/time/resource evidence measured",
                              "Fixed LAM/REF_R tables trained for Gaussian law need separate attribution under sphere law"]}
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"version": report["version"], "tensor_contract": report["tensor_contract"],
                      "width1024_rows": [r for r in rows if r["input_width"] == 1024]}, indent=2))


if __name__ == "__main__":
    main()

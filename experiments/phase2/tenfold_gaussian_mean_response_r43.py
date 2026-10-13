"""Permitted Gaussian ReLU mean transport response, no labels/fitting.

Holding variance fixed is a local response model. True shifts before ReLU also
change covariance and higher sources; this helper does not transform them.
"""
import flopscope as flops
import flopscope.numpy as fnp


def gaussian_relu_mean(mu,sigma):
    alpha=mu/sigma
    return sigma*flops.stats.norm.pdf(alpha)+mu*flops.stats.norm.cdf(alpha)


def exact_response(mu,sigma,h):
    return gaussian_relu_mean(mu+h,sigma)-gaussian_relu_mean(mu,sigma)


def linear_response(mu,sigma,h):
    return flops.stats.norm.cdf(mu/sigma)*h


def quadratic_response(mu,sigma,h):
    return linear_response(mu,sigma,h)+.5*flops.stats.norm.pdf(mu/sigma)/sigma*(h*h)

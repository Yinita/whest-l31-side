"""Wright2024 Gaussian closure, ReLU covariance series through order12.

Full layerwise mean/covariance algorithm, finite-series approximation;
not exact deep-network distribution or infinite-series evaluation.
"""
import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        n = mlp.weights[0].shape[0]
        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0, 1))
        rows = []
        for layer, weights in enumerate(mlp.weights):
            w = weights.astype(fnp.float32)
            mu = w.T @ mu
            pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            var = fnp.maximum(fnp.diag(pre), 1e-10)
            sigma = fnp.sqrt(var)
            alpha = mu / sigma
            phi = flops.stats.norm.pdf(alpha).astype(fnp.float32)
            gate = flops.stats.norm.cdf(alpha).astype(fnp.float32)
            mean = sigma * phi + mu * gate
            rows.append(mean)
            if layer == len(mlp.weights) - 1:
                break
            scale = sigma[:, None] * sigma[None, :]
            rho = fnp.clip(pre / scale, -1.0, 1.0)
            cov = rho * gate[:, None] * gate[None, :]
            power = rho * rho
            hprev = fnp.ones_like(alpha)
            h = alpha
            for order in range(2, 13):
                hermite = hprev if order == 2 else h
                if order >= 4:
                    hprev, h = h, alpha * h - (order - 3) * hprev
                    hermite = h
                derivative = phi * hermite
                cov = cov + power * derivative[:, None] * derivative[None, :] / math.factorial(order)
                power = power * rho
            cov = cov * scale
            second = (mu * mu + var) * gate + mu * sigma * phi
            fnp.fill_diagonal(cov, fnp.maximum(second - mean * mean, 0.0))
            cov = flops.as_symmetric(cov, symmetry=(0, 1))
            mu = mean
        return fnp.stack(rows)

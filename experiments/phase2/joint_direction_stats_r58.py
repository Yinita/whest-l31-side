"""Offline stable scalar-direction joint cumulant statistics, no estimator.

Each projected moment and previous-neuron marginal uses the same MC samples.
No full covariance or third/fourth tensor is constructed.
"""
import numpy as np


def empty(shape):
    return dict(count=0, mean=np.zeros(shape, dtype=np.float64),
                m2=np.zeros(shape, dtype=np.float64),
                m3=np.zeros(shape, dtype=np.float64),
                m4=np.zeros(shape, dtype=np.float64))


def block(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    centered = values-mean
    square = centered*centered
    return dict(count=len(values), mean=mean, m2=square.sum(axis=0),
                m3=(square*centered).sum(axis=0), m4=(square*square).sum(axis=0))


def merge(left, right):
    a, b = int(left['count']), int(right['count'])
    assert a >= 0 and b > 0
    n = a+b
    delta = right['mean']-left['mean']
    old2, old3 = left['m2'], left['m3']
    return dict(count=n, mean=left['mean']+delta*b/n,
        m2=old2+right['m2']+delta**2*a*b/n,
        m3=old3+right['m3']+delta**3*a*b*(a-b)/n**2
            +3*delta*(a*right['m2']-b*old2)/n,
        m4=left['m4']+right['m4']+delta**4*a*b*(a*a-a*b+b*b)/n**3
            +6*delta**2*(a*a*right['m2']+b*b*old2)/n**2
            +4*delta*(a*right['m3']-b*old3)/n)


def cumulants(stats):
    """Unbiased k-statistics; central plug-in values retained for audit."""
    n = int(stats['count']); assert n > 3
    c2, c3, c4 = [stats[k]/n for k in ('m2', 'm3', 'm4')]
    k2 = n*c2/(n-1)
    k3 = n*n*c3/((n-1)*(n-2))
    k4 = n*n*((n+1)*c4-3*(n-1)*c2*c2)/((n-1)*(n-2)*(n-3))
    return np.stack([stats['mean'], k2, k3, k4], axis=-1)


def joint_targets(stats, directions, width):
    """directions: layer×previous_neuron×canonical_output; stats packed rows.

    J_p = k_p(sum_i W_i X_i) - sum_i W_i**p k_p(X_i), p2..4.
    Mean residual is a linearity/numerics diagnostic, never a joint target.
    """
    values = cumulants(stats)
    d = directions.shape[-1]
    assert values.shape == (len(directions), width+2*d, 4)
    prior = values[:, :width]
    projected = values[:, width:width+d]
    actual_f32 = values[:, width+d:]
    diagonal = np.stack([
        np.einsum('lnd,ln->ld', directions**p, prior[:, :, p-1])
        for p in range(1, 5)], axis=-1)
    return dict(previous_post4=prior, projected_pre4=projected,
                actual_f32_pre4=actual_f32, diagonal_pre4=diagonal,
                joint_pre4=projected-diagonal,
                projection_float32_difference4=actual_f32-projected)


def readout_effect(pre, errors):
    """Fixed first-order ReLU K2/K3/K4 sensitivity, diagnostic only."""
    mu, var, d3, g4, phi = [pre[..., k] for k in range(5)]
    sigma=np.sqrt(var);alpha=mu/sigma
    coefficients=np.stack([phi/(2*sigma), -alpha*phi/(6*sigma**2),
                            (alpha*alpha-1)*phi/(24*sigma**3)],axis=-1)
    return np.sum(coefficients*errors[..., 1:],axis=-1)

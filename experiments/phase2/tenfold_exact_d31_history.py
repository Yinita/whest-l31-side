"""Exact shared-leg 31-support birth and CP4/complete CP3 feed contractions.

Only matrices: H=4 Sym(Q,Q,Q,R), F=Sym(X,Q,Q)+Sym(U,Q,R).
Sym averages permutations. No truth, clipping or rank/age selection.
"""

import flopscope.numpy as fnp


def zero_diag(a):
    return a - fnp.diag(fnp.diag(a))


def use_slices(q, r):
    q2 = q * q
    qr = q * r
    d4 = 4 * fnp.sum(q2 * qr, axis=1)
    d31 = 3 * (q * qr) @ q.T + (q2 * q) @ r.T
    prod = 2 * qr @ q2.T
    d22 = prod + prod.T
    return d4, zero_diag(d31), zero_diag(d22)


def feed_slices(q, r, x, u, need_d21=True):
    d3 = fnp.sum(x * q * q + u * q * r, axis=1)
    if not need_d21:
        return d3, None
    d21 = ((q * q) @ x.T + 2 * (x * q) @ q.T + (u * q) @ r.T + (u * r) @ q.T + (q * r) @ u.T) / 3
    return d3, zero_diag(d21)


def nonlinear_feed(q, r, gain, w2):
    """Returns postq/r and newpostX/U. NewX/U must not gain again."""
    return q * gain[:, None], r * gain[:, None], 3 * w2[:, None] * q * r, 3 * w2[:, None] * q * q


def caps(q, r, z):
    """Distinct F_jk=sum_i z_i H_iijk; P_jk=sum_i z_i H_ijjk."""
    q2 = q * q
    qr = q * r
    a = qr.T @ z
    b = q2.T @ z
    d = q.T @ z
    e = r.T @ z
    f = 2 * (q * a) @ q.T + (r * b) @ q.T + (q * b) @ r.T
    p = 2 * (qr * d) @ q.T + (q2 * e) @ q.T + (q2 * d) @ r.T
    _, d31, d22 = use_slices(q, r)
    # Exclusion j/k subtracts ordinary repeated slices; diagonals discarded.
    f = f - z[:, None] * d31 - z[None, :] * d31.T
    p = p - z[:, None] * d31 - z[None, :] * d22
    return zero_diag(f), zero_diag(p)

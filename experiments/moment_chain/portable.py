"""Portable Hermite-step (mean, cov) ReLU closure + stabilizer features.

Numpy-only mirror of jamesrahenry/arc-whitebox-replication moment_chain
(chain_port_refit.py / estimator_chain.py). Used for offline DAgger fitting.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

import numpy as np

SQ2PI = np.sqrt(2.0 * np.pi)
K_H_DEFAULT = 10
WIDTH = 256
DEPTH = 32


def phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / SQ2PI


def Phi(x: np.ndarray) -> np.ndarray:
    """A&S 7.1.26 normal CDF — portable to fnp (no erf)."""
    z = x / np.sqrt(2.0)
    s = np.sign(z)
    az = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    e = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-az * az)
    return 0.5 * (1.0 + s * e)


def hermite_coeffs(a: np.ndarray, K: int) -> list[np.ndarray]:
    """a_k(alpha) for relu(alpha+Z), k=0..K."""
    out = [phi(a) + a * Phi(a), Phi(a)]
    He_prev = np.ones_like(a)
    He_cur = -a
    fact = 2.0
    out.append(He_prev * phi(a) / fact)
    for k in range(3, K + 1):
        fact *= k
        out.append(He_cur * phi(a) / fact)
        He_prev, He_cur = He_cur, (-a) * He_cur - (k - 2) * He_prev
    return out


def step_np(
    m_act: np.ndarray, S_act: np.ndarray, W: np.ndarray, K: int = K_H_DEFAULT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Post-ReLU (mean, cov) through weight W. Returns m,S,mu_pre,sig,Spre,a."""
    mu = m_act @ W
    Spre = W.T @ S_act @ W
    var = np.clip(np.diag(Spre), 1e-30, None)
    sig = np.sqrt(var)
    a = mu / sig
    rho = np.clip(Spre / np.outer(sig, sig), -0.9999, 0.9999)
    coeffs = hermite_coeffs(a, K)
    m_post = sig * coeffs[0]
    S_post = np.zeros_like(Spre)
    rk = rho.copy()
    for k in range(1, K + 1):
        ck = coeffs[k]
        S_post = S_post + np.outer(ck, ck) * rk
        rk = rk * rho
    S_post = S_post * np.outer(sig, sig)
    d = var * ((1.0 + a * a) * Phi(a) + a * phi(a)) - m_post**2
    S_post[np.diag_indices(len(mu))] = np.clip(d, 1e-30, None)
    return m_post, S_post, mu, sig, Spre, a


def probe_cumulants(weights: list[np.ndarray], n_probe: int, seed: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    w = weights[0].shape[0]
    x = rng.standard_normal((n_probe, w))
    out = []
    for W in weights:
        z = x @ W
        mu = z.mean(0)
        zc = z - mu
        s2 = (zc**2).mean(0)
        out.append(
            (
                (zc**3).mean(0),
                (zc**4).mean(0) - 3.0 * s2**2,
                (zc**2).T @ zc / n_probe,
            )
        )
        x = np.maximum(z, 0.0)
    return out


def mean_features(a: np.ndarray, sig: np.ndarray, k3: np.ndarray, k4: np.ndarray) -> np.ndarray:
    pa = phi(a)
    c3 = -(k3 / 6.0) * (a * pa / sig**2)
    c4 = (k4 / 24.0) * (pa * (a * a - 1.0) / sig**3)
    return np.stack(
        [
            c3,
            c4,
            pa * sig,
            a * pa * sig,
            a * a * pa * sig,
            c3 * a,
            c4 * a,
            (k3 / sig**2) * pa,
        ],
        axis=1,
    )


def var_features(a: np.ndarray, sig: np.ndarray, k3: np.ndarray, k4: np.ndarray) -> np.ndarray:
    pa = phi(a)
    cm2 = k3 * pa / (3.0 * sig) - k4 * a * pa / (12.0 * sig**2)
    return np.stack(
        [
            cm2,
            sig**2 * pa,
            sig**2 * a * pa,
            sig**2 * a * a * pa,
            k3 * pa / sig,
            k4 * pa / sig**2,
        ],
        axis=1,
    )


def tcross_feats(
    mu_pre: np.ndarray, sig_pre: np.ndarray, Spre: np.ndarray, kiij: np.ndarray, a: np.ndarray
) -> list[np.ndarray]:
    rho = np.clip(Spre / np.outer(sig_pre, sig_pre), -0.999, 0.999)
    s_cond = sig_pre[None, :] * np.sqrt(np.clip(1.0 - rho**2, 1e-12, None))
    m_cond0 = mu_pre[None, :] - rho * sig_pre[None, :] * a[:, None]
    G = (phi(a)[:, None] / sig_pre[:, None]) * Phi(m_cond0 / s_cond)
    T1 = 0.5 * (kiij * G + (kiij * G).T)
    return [T1, T1 * rho]


def load_coefs_b64(path: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    blob = path.read_text().strip()
    flat = struct.unpack("<512d", base64.b64decode(blob))
    out = []
    for i in range(DEPTH):
        base = i * 16
        Am = np.asarray(flat[base : base + 8], dtype=np.float64)
        Av = np.asarray(flat[base + 8 : base + 14], dtype=np.float64)
        Ao = np.asarray(flat[base + 14 : base + 16], dtype=np.float64)
        out.append((Am, Av, Ao))
    return out


def save_coefs_b64(
    coefs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path
) -> None:
    flat: list[float] = []
    for Am, Av, Ao in coefs:
        flat.extend(np.asarray(Am, dtype=np.float64).ravel().tolist())
        flat.extend(np.asarray(Av, dtype=np.float64).ravel().tolist())
        flat.extend(np.asarray(Ao, dtype=np.float64).ravel().tolist())
    assert len(flat) == 512
    blob = base64.b64encode(struct.pack("<512d", *flat)).decode()
    path.write_text(blob)


def run_chain(
    weights: list[np.ndarray],
    probe: list[tuple],
    coefs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
    K: int = K_H_DEFAULT,
    mean_only: bool = False,
) -> np.ndarray:
    """Return stacked post-ReLU means (depth, width)."""
    w = weights[0].shape[0]
    m = np.zeros(w)
    S = np.eye(w)
    means = []
    for l, W in enumerate(weights):
        m_hat, S_hat, mu_pre, sig_pre, Spre, a = step_np(m, S, W, K=K)
        if coefs is None:
            m, S = m_hat, S_hat
        else:
            k3, k4, kiij = probe[l]
            Am, Av, Ao = coefs[l]
            Fm = mean_features(a, sig_pre, k3, k4)
            m = m_hat + Fm @ Am
            if mean_only:
                S = S_hat
            else:
                Fo = tcross_feats(mu_pre, sig_pre, Spre, kiij, a)
                S = S_hat + Ao[0] * Fo[0] + Ao[1] * Fo[1]
                S[np.diag_indices(w)] = np.diag(S_hat) + var_features(a, sig_pre, k3, k4) @ Av
        means.append(np.maximum(m, 0.0))
    return np.stack(means, axis=0)

"""O21/O22 exact scalar radial harmonic convention and actual metric transport."""

import json
from pathlib import Path

import numpy as np


def pairings(indices):
    if not indices:
        yield []
        return
    first = indices[0]
    for j in range(1, len(indices)):
        rest = indices[1:j] + indices[j + 1 :]
        for pairs in pairings(rest):
            yield [(first, indices[j])] + pairs


def lift(metric):
    n = len(metric)
    tensor = np.zeros((n,) * 6)
    symbols = "abcdef"
    for pairs in pairings(list(range(6))):
        specs = [symbols[i] + symbols[j] for i, j in pairs]
        tensor += np.einsum(",".join(specs) + "->abcdef", metric, metric, metric)
    return tensor / 15


def main():
    rng = np.random.default_rng(84)
    rows = []
    for n in (2, 3, 4):
        identity = np.eye(n)
        radial = lift(identity)
        w = rng.normal(size=(n, n)) / np.sqrt(n)
        actual = np.einsum(
            "ai,bj,ck,dl,em,fp,ijklmp->abcdef", w, w, w, w, w, w, radial, optimize=True
        )
        metric = w @ w.T
        represented = lift(metric)
        error = float(np.max(abs(actual - represented)))
        assert error < 1e-10
        marginal = np.array([actual[(i,) * 6] for i in range(n)])
        assert np.allclose(marginal, np.diag(metric) ** 3)
        trace = float(np.einsum("iijjkk", radial))
        expected = n * (n + 2) * (n + 4) / 15
        assert abs(trace - expected) < 1e-10
        rows.append(
            dict(n=n, tensor_transport_error=error, radial_triple_trace=trace, normalizer=expected)
        )
    path = Path("results/phase2/ablation_v30_083_084_radial_k6_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                convention="T6=q*Sym(M tensor M tensor M), average15pairings; marginal=q*Mii³; identitytripletrace=q*n(n+2)(n+4)/15",
                conclusion="Radial pure-scalar normalization and actualW metric transport exact. This defines representation only; full nonlinearK6 production/other harmonic components and lawfulstate cost still pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

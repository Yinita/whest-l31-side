"""L15 independent finite difference of frozen tanh model analytical gradient."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(110)
    x = rng.normal(size=(17, 13))
    target = rng.normal(size=17)
    params = [
        rng.normal(scale=0.2, size=(13, 8)),
        rng.normal(scale=0.2, size=8),
        rng.normal(scale=0.2, size=8),
        rng.normal(scale=0.2, size=1),
    ]
    w, b, v, c = params
    hidden = np.tanh(x @ w + b)
    error = hidden @ v + c[0] - target
    dy = 2 * error / len(x)
    dh = dy[:, None] * v[None, :] * (1 - hidden * hidden)
    grads = [x.T @ dh, dh.sum(axis=0), hidden.T @ dy, np.array([dy.sum()])]

    def loss():
        return np.mean(
            (np.tanh(x @ params[0] + params[1]) @ params[2] + params[3][0] - target) ** 2
        )

    errors = []
    for p, g in zip(params, grads):
        for index in np.ndindex(p.shape):
            old = p[index]
            p[index] = old + 1e-6
            plus = loss()
            p[index] = old - 1e-6
            minus = loss()
            p[index] = old
            errors.append(abs((plus - minus) / 2e-6 - g[index]))
    assert max(errors) < 1e-8
    path = Path("results/phase2/ablation_v30_110_gradient_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                parameters=len(errors),
                max_absolute_gradient_error=float(max(errors)),
                conclusion="All121 analytical gradients match independent central finite differences. CV remains negative; no charged candidate or acceptance claim.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

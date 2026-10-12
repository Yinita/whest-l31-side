"""H19/H20 freeze transported proxy G and constant-metric limiting checks."""

import json
from pathlib import Path

import numpy as np
from metric_slice_derivation import lift


def main():
    rng = np.random.default_rng(492)
    rows = []
    for n in (4, 7, 11):
        w = rng.normal(size=(n, n))
        c = rng.normal(size=(n, n))
        c = (c + c.T) / 2
        var = np.diag(c).copy()
        g = rng.normal(size=n)
        lam = 0.13
        prev = np.diag(g) + lam * (c - np.diag(var))
        proxy = w @ prev @ w.T
        dg = (w * w) @ (g - var * lam) + lam * np.diag(w @ c @ w.T)
        assert np.allclose(dg, np.diag(proxy))
        m = w @ w.T
        t = lift(proxy, m)
        k31 = (dg[:, None] * m + np.diag(m)[:, None] * proxy) / 2
        k22 = (
            dg[:, None] * np.diag(m)[None, :] + np.diag(m)[:, None] * dg[None, :] + 4 * proxy * m
        ) / 6
        error = max(
            float(
                np.max(abs(k31 - np.array([[t[i, i, i, j] for j in range(n)] for i in range(n)])))
            ),
            float(
                np.max(abs(k22 - np.array([[t[i, i, j, j] for j in range(n)] for i in range(n)])))
            ),
        )
        assert error < 1e-10
        constant = 3 * np.eye(n)
        old22 = (dg[:, None] + dg[None, :]) * 0.5
        new22 = (dg[:, None] * 3 + 3 * dg[None, :] + 4 * proxy * constant) / 6
        np.fill_diagonal(old22, 0)
        np.fill_diagonal(new22, 0)
        assert np.allclose(old22, new22)
        rows.append(
            dict(
                n=n,
                max_slice_error=error,
                diagonal_mapping_error=float(np.max(abs(dg - np.diag(proxy)))),
            )
        )
    path = Path("results/phase2/ablation_h19_h20_proxy_mapping.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                proxy_definition="G_pre = W[diag(g_prev)+lambda(C_prev-diag(var_prev))]W.T; diagonal equals frozen dG. Use full G_pre, do not replace its offdiagonal by lambda*C_pre.",
                cost="Extra full G_pre sandwich + metricGram ~6n^3 dense FLOPs/layer before Strassen/copies; actual cost still must be measured.",
                constant_metric_limit="M=METRIC_C*I gives frozen WK22 offdiag=(dGi+dGj)*METRIC_C/6; WK31 gives METRIC_C*Gij/2, but V29 lambda*Coff is an additional proxy assumption.",
                conclusion="Exact full proxy mapping and six-sym slice contract established; legal candidate and network score pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

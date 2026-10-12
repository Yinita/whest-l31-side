"""O28 pre-registered first-layer diagram scale, no target or candidate tuning."""
# ruff: noqa: E402 -- BLAS thread environment must precede numpy import

import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path

import numpy as np


def main():
    rows = []
    for n in (16, 32, 64, 128):
        for seed in (12026001, 12026002, 12026003):
            rng = np.random.default_rng(seed)
            w = (rng.standard_normal((n, n)) * np.sqrt(2 / n)).astype(np.float32)
            cov = w.T @ w
            sigma = np.sqrt(np.diag(cov))
            corr = cov / (sigma[:, None] * sigma[None, :])
            np.fill_diagonal(corr, 0)
            # alpha=0 at first preactivation: w1=1/2, w2=phi(0)/sigma.
            w1 = np.full(n, 0.5)
            w2 = 1 / (np.sqrt(2 * np.pi) * sigma)
            # Exact all-distinct leading tree K111, covariance input Gaussian.
            c = cov.copy()
            np.fill_diagonal(c, 0)
            tree = (
                w2[:, None, None]
                * w1[None, :, None]
                * w1[None, None, :]
                * c[:, :, None]
                * c[:, None, :]
                + w1[:, None, None]
                * w2[None, :, None]
                * w1[None, None, :]
                * c[:, :, None]
                * c[None, :, :]
                + w1[:, None, None]
                * w1[None, :, None]
                * w2[None, None, :]
                * c[:, None, :]
                * c[None, :, :]
            )
            idx = np.arange(n)
            distinct = (
                (idx[:, None, None] != idx[None, :, None])
                & (idx[:, None, None] != idx[None, None, :])
                & (idx[None, :, None] != idx[None, None, :])
            )
            tree *= distinct
            edge = tree * (
                w1[:, None, None] * w2[None, :, None] * w2[None, None, :] * c[None, :, :]
                + w2[:, None, None] * w1[None, :, None] * w2[None, None, :] * c[:, None, :]
                + w2[:, None, None] * w2[None, :, None] * w1[None, None, :] * c[:, :, None]
            )
            rows.append(
                dict(
                    width=n,
                    seed=seed,
                    corr_offdiag_rms=float(np.sqrt(np.mean(corr[corr != 0] ** 2))),
                    tree_distinct_rms=float(np.sqrt(np.mean(tree[distinct] ** 2))),
                    capped211_proxy_rms=float(np.sqrt(np.mean((tree * c[:, None, :] * c[None, :, :])[distinct] ** 2))),
                    edge_distinct_rms=float(np.sqrt(np.mean(edge[distinct] ** 2))),
                )
            )
    fits = {}
    for metric in ("corr_offdiag_rms", "tree_distinct_rms", "edge_distinct_rms", "capped211_proxy_rms"):
        slopes = []
        for seed in (12026001, 12026002, 12026003):
            select = [r for r in rows if r["seed"] == seed]
            slopes.append(
                float(
                    np.polyfit(
                        np.log([r["width"] for r in select]), np.log([r[metric] for r in select]), 1
                    )[0]
                )
            )
        fits[metric] = dict(
            seed_slopes=slopes,
            mean=float(np.mean(slopes)),
            spread=float(np.std(slopes, ddof=1)),
            uncertainty="Three paired seeds only; spread is not a population confidence interval",
        )
    p = Path("results/phase2/ablation_v30_090_r02_extended_width_law.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                fits=fits,
                scope="First-layer Gaussian lawful state, widths32/64/128; no references or score",
                conclusion="Measured tree and K111×edge scale only. Does not establish deep-network capped211 behavior or 1024 improvement; Capped211 proxy explicitly differs from complete official production; no deep-state closure claim.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""Exact factor regrouping for two linear feedback D3 terms; cross term stays."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(20)
    rows = []
    for n, r in [(4, 2), (7, 3), (11, 4)]:
        a, p = rng.normal(size=(2, n, n))
        f1, r1, f2, r2 = rng.normal(size=(4, n, r))
        w = rng.normal(size=n)
        xt = f1 @ r1.T
        yt = f2 @ r2.T
        original = np.sum(p * (3 * a * yt + xt * a * w[None, :]), axis=1)
        ap = a * p
        regrouped = 3 * np.sum((ap @ r2) * f2, axis=1) + np.sum(
            ((ap * w[None, :]) @ r1) * f1, axis=1
        )
        err = float(abs(original - regrouped).max())
        assert err < 1e-11
        rows.append(dict(n=n, rank=r, error=err))
    out = Path("results/phase2/ablation_v30_020_feedback_cost.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                scope="Exact factor linearD3feedback only; XtYt cross term remains materialized",
                rows=rows,
                formula="3rowsum(P*A*F2R2T)=3rowsum((AP@R2)*F2); rowsum(P*A*w2*F1R1T)=rowsum(((AP*w2)@R1)*F1)",
                cost="Two new n²r GEMMs while original Xt/Yt must remain for cross term: +4n²r leading billed FLOPs unless shared other contractions; no cost-saving proof",
                conclusion="Factor formula correct, naive replacement expected cost increase; implementation test still pending",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

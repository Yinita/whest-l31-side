"""O06-O09 syntheticfullcore USE/FEED path isolation contracts, no oraclecandidate."""

import json
from pathlib import Path

import numpy as np
from metric_slice_derivation import lift


def main():
    rng = np.random.default_rng(689)
    rows = []
    for n in (4, 7, 11):
        core = rng.normal(size=(n, n))
        core = (core + core.T) / 2
        metric = rng.normal(size=(n, n))
        metric = metric @ metric.T / n
        proxy = np.diag(np.diag(core))
        full = lift(core, metric)
        old = lift(proxy, metric)
        use = np.array([full[(i,) * 4] - old[(i,) * 4] for i in range(n)])
        # Offdiag residual has zero marginal use at this step, yet nonzero31/211 feed.
        assert np.max(abs(use)) < 1e-12
        feed = np.array(
            [
                full[i, i, j, k] - old[i, i, j, k]
                for i in range(n)
                for j in range(n)
                for k in range(n)
                if len({i, j, k}) == 3
            ]
        )
        w = rng.normal(size=(n, n)) / np.sqrt(n)
        rotated = w @ (core - proxy) @ w.T
        next_marginal = np.diag(rotated) * np.diag(w @ metric @ w.T)
        rows.append(
            dict(
                n=n,
                current_use_difference=float(np.linalg.norm(use)),
                distinct_feed_difference=float(np.linalg.norm(feed)),
                next_use_difference=float(np.linalg.norm(next_marginal)),
            )
        )
    path = Path("results/phase2/ablation_v30_068_071_kernel_path_split.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Synthetic samecore singlepath diagnostic: offdiagcore removal has zerocurrentmarginal but nonzerofeed and nexttransportmarginal. This is not complete officialproduction oracle or network replacement; realstate/fullkernel/contract gates pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

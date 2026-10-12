"""Legal K31/K211 conversion from finite distribution moments, independent cumulants."""

import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.probe_k211_conversion import cumulant


def main():
    output = Path("results/phase2/ablation_v30_063_064_r02_charged_conversion.json")
    assert not output.exists()
    rng = np.random.default_rng(6402)
    rows = []
    for n in (4, 7):
        x = rng.normal(size=(53, n)) ** 2 + rng.normal(size=n)
        weights = rng.uniform(size=53)
        weights /= weights.sum()
        mean = x.T @ weights
        centered = x - mean
        covariance = (centered * weights[:, None]).T @ centered
        p21 = np.empty((n, n))
        p31 = np.empty((n, n))
        p111 = np.empty((n, n, n))
        p211 = np.empty_like(p111)
        exact31 = np.zeros((n, n))
        exactcap = np.zeros_like(exact31)
        for i in range(n):
            for j in range(n):
                p21[i, j] = cumulant([x[:, i] ** 2, x[:, j]], weights)
                p31[i, j] = cumulant([x[:, i] ** 3, x[:, j]], weights)
                exact31[i, j] = (
                    cumulant([x[:, i], x[:, i], x[:, i], x[:, j]], weights) if i != j else 0
                )
                for k in range(n):
                    p111[i, j, k] = cumulant([x[:, i], x[:, j], x[:, k]], weights)
                    p211[i, j, k] = cumulant([x[:, i] ** 2, x[:, j], x[:, k]], weights)
                    if len({i, j, k}) == 3:
                        exactcap[j, k] += cumulant([x[:, i], x[:, i], x[:, j], x[:, k]], weights)
        with flops.BudgetContext(flop_budget=2**30) as ctx:
            mu, cov, pk21, pk31, pk111, pk211 = [
                fnp.array(a.astype("float32")) for a in (mean, covariance, p21, p31, p111, p211)
            ]
            raw2 = fnp.array((x * x).T @ weights).astype(fnp.float32)
            k31 = pk31 - 3 * mu[:, None] * pk21 + (6 * mu * mu - 3 * raw2)[:, None] * cov
            fnp.fill_diagonal(k31, 0)
            converted = (
                pk211 - 2 * mu[:, None, None] * pk111 - 2 * cov[:, :, None] * cov[:, None, :]
            )
            cap = (
                fnp.sum(converted, axis=0)
                - fnp.stack([converted[j, j, :] for j in range(n)], axis=0)
                - fnp.stack([converted[k, :, k] for k in range(n)], axis=1)
            )
            fnp.fill_diagonal(cap, 0)
        errors = [
            float(np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected))
            for actual, expected in ((k31, exact31), (cap, exactcap))
        ]
        assert max(errors) < 1e-5
        rows.append(
            dict(
                n=n,
                k31_relative_error=errors[0],
                k211_cap_relative_error=errors[1],
                flops=ctx.summary_dict()["flops_used"],
            )
        )
    output.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="Legal conversion arithmetic only; cumulant inputs offline finite support, no benchmark labels or full candidate",
                limitations=[
                    "K211 dense input bounded small only",
                    "Production approximation errors remain separate",
                    "No score improvement claimed",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()

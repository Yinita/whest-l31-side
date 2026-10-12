"""H16 full exact power-cumulant K31 conversion and projected offdiag coefficient."""

import json
from pathlib import Path

import numpy as np
from probe_k211_conversion import cumulant


def main():
    rng = np.random.default_rng(46)
    rows = []
    for n in (4, 7):
        x = rng.normal(size=(n, 79)) ** 2 + rng.normal(size=(n, 1))
        weights = rng.uniform(size=79)
        weights /= weights.sum()
        means = x @ weights
        errors = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cov = cumulant([x[i], x[j]], weights)
                p31 = cumulant([x[i] ** 3, x[j]], weights)
                p21 = cumulant([x[i] ** 2, x[j]], weights)
                formula = (
                    p31
                    - 3 * means[i] * p21
                    + (6 * means[i] ** 2 - 3 * (weights @ (x[i] ** 2))) * cov
                )
                exact = cumulant([x[i], x[i], x[i], x[j]], weights)
                errors.append(abs(formula - exact))
        assert max(errors) < 1e-10
        rows.append(dict(n=n, max_conversion_error=max(errors)))
    path = Path("results/phase2/ablation_v30_046_k31_conversion_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="K31=pK31−3mu_i*pK21+(6mu_i²−3E[Xi²])*K11; core offdiag receives6*(K31+K31.T)/(n+4)=12Sym(K31)/(n+4)",
                conclusion="Exact finite-distribution conversion and identitymetric core coefficient; productionformula, existingproxy removal/lowrankstate and chargedcandidate pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

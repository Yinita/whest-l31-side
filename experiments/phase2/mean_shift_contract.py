"""L16 offline central-moment contract; no learned candidate or score claim."""

import json
import math
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(111)
    rows = []
    for n in (3, 7, 19):
        x = rng.normal(size=(400, n)) ** 2
        weights = rng.uniform(size=400)
        weights /= weights.sum()
        mu = weights @ x
        z = x - mu
        delta = rng.normal(scale=0.1, size=n)
        # A changed estimate of the mean is distinct from translating X.
        for j in range(n):
            moments = [float(weights @ z[:, j] ** k) for k in range(5)]
            for k in (2, 3, 4):
                formula = sum(
                    math.comb(k, r) * (-delta[j]) ** (k - r) * moments[r] for r in range(k + 1)
                )
                direct = float(weights @ (z[:, j] - delta[j]) ** k)
                rows.append(abs(formula - direct))
        translated = x + delta
        translated_mu = weights @ translated
        before_cov = (z * weights[:, None]).T @ z
        after_z = translated - translated_mu
        after_cov = (after_z * weights[:, None]).T @ after_z
        assert np.allclose(before_cov, after_cov, atol=1e-12)
        # Recentring the same distribution around mu+delta gives nonzero first moment.
        assert np.allclose(weights @ (x - (mu + delta)), -delta)
    result = dict(
        max_moment_error=max(rows),
        checks=len(rows),
        conclusion="Cumulants of order >=2 are invariant under deterministic translation, but changing only the estimated mean of the same distribution is not translation: its residual has mean -delta. L16 needs an explicit corrected distribution model and layer8 CV benefit before candidate integration.",
        translation_model="X_new = X + delta: mean shifts; covariance and cumulants >=2 unchanged; ReLU-support/nonnegativity must still be checked.",
        recentering_model="E[(X-mu-delta)^k] = sum_r binom(k,r)(-delta)^(k-r) E[(X-mu)^r]; first central moment is -delta.",
        adjusted_score=None,
        acceptance="Algebra only; no candidate, score, resource or learning acceptance",
    )
    path = Path("results/phase2/ablation_v30_111_mean_shift_contract.json")
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()

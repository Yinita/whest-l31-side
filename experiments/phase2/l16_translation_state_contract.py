"""L16 translated-output distribution state semantics, distinct from rawmoments frozen."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(116)
    rows = []
    for n in (4, 7, 11):
        samples = rng.normal(size=(4096, n)) ** 2
        delta = rng.normal(size=n) * 0.01
        mean = samples.mean(axis=0)
        centered = samples - mean
        translated = samples + delta
        centered_after = translated - translated.mean(axis=0)
        error = float(np.max(abs(centered_after - centered)))
        assert error < 1e-12
        cov_before = centered.T @ centered / len(samples)
        cov_after = centered_after.T @ centered_after / len(samples)
        raw_second_before = np.mean(samples * samples, axis=0)
        raw_second_after = np.mean(translated * translated, axis=0)
        expected = raw_second_before + 2 * delta * mean + delta * delta
        assert np.allclose(raw_second_after, expected)
        rows.append(
            dict(
                n=n,
                centered_sample_error=error,
                covariance_error=float(np.max(abs(cov_before - cov_after))),
                raw_second_error=float(np.max(abs(raw_second_after - expected))),
            )
        )
    p = Path("results/phase2/ablation_v30_111_translation_state_contract.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="If corrected output distribution is explicitly a deterministic translation Y+delta, centered cumulants remain unchanged while raw moments shift consistently. This interpretation is mathematically coherent but may violate postReLU nonnegative support and does not capture true error covariance. Actuallearned middle-layer mechanism and positiveCV gate absent.",
                scope="Distribution assumption contract, not candidate or score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

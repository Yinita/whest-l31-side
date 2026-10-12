"""O10 synthetic representation residual spectral mode partition."""

import json
from pathlib import Path

import numpy as np


def main():
    source = Path("results/phase2/ablation_v30_067_core_representation_bound.json")
    original = json.loads(source.read_text())
    rows = []
    for r in original["rows"]:
        s = np.array(r["residual_singular_values"])
        energy = s * s
        cumulative = np.cumsum(energy) / energy.sum()
        rows.append(
            dict(
                n=r["n"],
                ranks_for_50_90_99pct=[
                    int(np.searchsorted(cumulative, x) + 1) for x in (0.5, 0.9, 0.99)
                ],
                first_mode_fraction=float(cumulative[0]),
                cumulative_energy=cumulative.tolist(),
            )
        )
    path = Path("results/phase2/ablation_v30_072_synthetic_core_modes.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Synthetic finite-support offdiagonal residual mode energy only. Does not separate actual network mean-error floor, reference uncertainty, or officialproduction kernel gap. O05 actual benchmark prerequisite remains unmet.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

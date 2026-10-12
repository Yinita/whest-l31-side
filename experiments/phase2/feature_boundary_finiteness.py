"""L33 feature normalization contract against frozen actual development cache."""

import json
from pathlib import Path

import numpy as np


def main():
    rows = []
    for i in range(8):
        f = np.load(f"benchmark/data/v29_features_v30_096/network_{i}.npz")["features"]
        sigma = f[:, :, 3]
        skew = f[:, :, 9] / sigma**3
        k4 = f[:, :, 11] / sigma**4
        alpha = f[:, :, 4]
        derived = np.stack(
            [skew, k4, alpha * alpha * f[:, :, 6], alpha**3 * f[:, :, 6], alpha**4 * f[:, :, 6]],
            axis=2,
        )
        assert np.isfinite(derived).all()
        rows.append(
            dict(
                network_id=i,
                sigma_min=float(sigma.min()),
                sigma_max=float(sigma.max()),
                derived_max_abs=float(abs(derived).max()),
                finite=True,
            )
        )
    p = Path("results/phase2/ablation_v30_128_feature_finiteness.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="All frozen development derived features finite with actual sigma ranges. Does not prove arbitrary-network variancefloor or heldout contract; this is observed cache validation. Charged feature construction in L03 now integrated; fullCV running serially.",
                scope="Actual feature boundary diagnostic",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

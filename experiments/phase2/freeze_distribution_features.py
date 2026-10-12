"""O27 freeze target-free shift features before reserved-shard state collection."""

import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    cache = Path("benchmark/data/v29_features_v30_096")
    rows = []
    names = ["variance", "alpha", "standardized_D3", "standardized_K4_scalar", "Phi"]
    quantiles = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    for i in range(8):
        # Only lawful cached features read; predictions and labels not accessed.
        features = np.load(cache / f"network_{i}.npz")["features"]
        for layer in range(16):
            f = features[layer]
            arrays = [f[:, 2], f[:, 4], f[:, 9] / f[:, 3] ** 3, f[:, 11] / f[:, 3] ** 4, f[:, 7]]
            rows.append(
                dict(
                    network_id=i,
                    layer=layer,
                    quantiles={
                        name: np.quantile(x, quantiles).tolist() for name, x in zip(names, arrays)
                    },
                )
            )
    p = Path("results/phase2/ablation_v30_089_frozen_shift_features.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                names=names,
                quantile_levels=quantiles,
                development_rows=rows,
                source_sha256=hashlib.sha256(
                    Path("estimators/k3_v29/estimator.py").read_bytes()
                ).hexdigest(),
                comparison_rule="Fixed features and quantile grid; reserved state collection diagnostic only; no target regression or post-inspection holdout reuse",
                status="development feature protocol frozen; reserved-shard state trace not yet collected",
                limitation="K4 scalar feature is V29 proxy, not full per-neuron kurtosis; standardized quantities diagnostic",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

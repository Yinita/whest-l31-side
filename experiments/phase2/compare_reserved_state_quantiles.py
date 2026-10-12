"""O27 frozen feature shift diagnostic, no labels and no candidate selection."""

import json
from pathlib import Path

import numpy as np


def main():
    protocol = json.loads(
        Path("results/phase2/ablation_v30_089_frozen_shift_features.json").read_text()
    )
    reserved = json.loads(
        Path("results/phase2/ablation_v30_089_reserved_state_only.json").read_text()
    )
    rows = []
    for layer in range(16):
        for name in protocol["names"]:
            dev = np.array(
                [r["quantiles"][name] for r in protocol["development_rows"] if r["layer"] == layer]
            )
            other = np.array([r["layers"][layer][name] for r in reserved["rows"]])
            scale = max(float(np.mean(dev[:, 5] - dev[:, 3])), 1e-12)
            rows.append(
                dict(
                    layer=layer,
                    feature=name,
                    development_mean_quantiles=dev.mean(axis=0).tolist(),
                    reserved_mean_quantiles=other.mean(axis=0).tolist(),
                    mean_absolute_quantile_shift=float(
                        np.mean(abs(dev.mean(axis=0) - other.mean(axis=0)))
                    ),
                    shift_over_development_iqr=float(
                        np.mean(abs(dev.mean(axis=0) - other.mean(axis=0))) / scale
                    ),
                )
            )
    p = Path("results/phase2/ablation_v30_089_distribution_shift.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Frozen lawful state-feature quantiles compared between public network shards, target-free. K4 is scalar proxy; sample8 shifts descriptive, not statistical significance. No tuning after this diagnostic may claim state distribution remains unseen.",
                scope="O27 complete fixed feature distribution comparison; no raw/adjusted score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

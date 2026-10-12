"""Report held-network explainability and heterogeneous correction effects."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    screens = json.loads((root / "results/phase2/ablation_learning_loocv_r01.json").read_text())
    baseline = screens["L03"]
    r2 = 1 - baseline["after_raw"] / baseline["before_raw"]
    output = root / "results/phase2/ablation_v30_113_predictability.json"
    assert not output.exists()
    output.write_text(
        json.dumps(
            dict(
                scope="NetworkCV13features readout explainability, original references; no charged candidate",
                cv_r_squared=r2,
                required_for_tenfold=0.9,
                gap=0.9 - r2,
                per_network=baseline["per_network"],
                conclusion="Negative held-network explained variance; simple linear features do not approach10x target",
                adjusted_score=None,
                reason="Need charged estimator to report adjusted; offline raw only",
            ),
            indent=2,
        )
        + "\n"
    )
    output = root / "results/phase2/ablation_v30_126_heterogeneous_gains.json"
    assert not output.exists()
    rows = []
    for key, r in screens.items():
        ratios = [row["raw_ratio"] for row in r["per_network"]]
        rows.append(
            dict(
                key=key,
                improved_networks=sum(x < 1 for x in ratios),
                worsened_networks=sum(x >= 1 for x in ratios),
                best_ratio=min(ratios),
                worst_ratio=max(ratios),
                ratios=ratios,
            )
        )
    output.write_text(
        json.dumps(
            dict(
                scope="All offline learning screens, network heterogeneous raw effects; not independent confirmation",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

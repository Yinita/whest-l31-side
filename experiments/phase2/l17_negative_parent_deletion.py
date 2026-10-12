"""L17 deletion diagnostic on negative parent; explicitly does not pass positive gate."""

import json
from pathlib import Path

import numpy as np
from learning_feature_screens import evaluate


def main():
    data = [dict(np.load(f"benchmark/data/v29_features_v30_096/network_{i}.npz")) for i in range(8)]
    parent = evaluate(data, list(range(13)))
    groups = {
        "threshold": [4, 5, 6, 7, 12],
        "skew": [9, 10],
        "kurtosis": [11],
        "amplitude": [1, 2, 3, 8],
        "constant": [0],
    }
    rows = []
    for name, deleted in groups.items():
        columns = [i for i in range(13) if i not in deleted]
        result = evaluate(data, columns)
        rows.append(
            dict(
                group=name,
                deleted_columns=deleted,
                remaining_columns=columns,
                before_raw=parent["after_raw"],
                after_raw=result["after_raw"],
                ratio=result["after_raw"] / parent["after_raw"],
                per_network=result["per_network"],
            )
        )
    p = Path("results/phase2/ablation_v30_112_negative_parent_deletion.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                parent_cv_raw=parent["after_raw"],
                uncorrected_raw=parent["before_raw"],
                conclusion="Five fixed group deletions refitted under networkLOO on negative L03 parent. Original task requires a reliably positive CV parent; gate remains unmet, diagnostic cannot satisfy adoption or final confirmation.",
                scope="Negative-parent attribution only; adjusted unavailable",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

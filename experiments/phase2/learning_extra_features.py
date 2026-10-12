"""Offline predefined L07-L09 feature blocks; no reference used to construct."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from datasets import Dataset

from experiments.phase2.learning_feature_screens import evaluate


def main():
    root = Path(__file__).resolve().parents[2]
    data = [
        dict(np.load(root / f"benchmark/data/v29_features_v30_096/network_{i}.npz"))
        for i in range(8)
    ]
    ds = Dataset.from_file(
        str(root / "benchmark/data/mini_first8/prepared/mini/data.arrow")
    ).select_columns(["weights"])
    spec = importlib.util.spec_from_file_location(
        "beta_v29", root / "estimators/k3_v29/estimator.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for i, item in enumerate(data):
        f = item["features"]
        weights = np.asarray(ds[i]["weights"], dtype=np.float32)
        extra = np.zeros((16, 1024, 6), dtype=np.float32)
        # V29 NO_CORR default1: actual previous original correction delta is zero.
        previous_delta = np.zeros(1024, dtype=np.float32)
        propagated = weights[-1].T @ previous_delta
        extra[-1, :, 2] = propagated
        extra[-1, :, 3] = propagated * f[-1, :, 7]
        transported_var = (weights[-1] ** 2).T @ f[-2, :, 2]
        extra[-1, :, 4] = transported_var
        extra[-1, :, 5] = f[-1, :, 2] - transported_var
        item["features"] = np.concatenate([f, extra], axis=2)
    out = {}
    # L07 cannot recover actual g4row from K4_sigma scalar; do not substitute.
    for key, columns in {
        "L08": list(range(13)) + [15, 16],
        "L09": list(range(13)) + [17, 18],
    }.items():
        r = evaluate(data, columns)
        r.update(
            added_feature_scope="Predefined original correction delta or variance transport; legal estimator ops still pending",
            adjusted_score=None,
        )
        out[key] = r
        print(key, r["after_raw"] / r["before_raw"])
    output = root / "results/phase2/ablation_learning_extra_r01.json"
    assert not output.exists()
    output.write_text(
        json.dumps(
            dict(results=out, L07="Pending actual g4row trace; scalar K4_sigma is not equivalent"),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""Same-run L03 versus L07 raw-only network LOO, reference read after trace."""

import json
from pathlib import Path

import numpy as np
from learning_feature_screens import evaluate


def main():
    root = Path(__file__).resolve().parents[2]
    data = []
    for i in range(8):
        traced = dict(np.load(root / f"benchmark/data/v29_g4_v30_102/network_{i}.npz"))
        original = dict(np.load(root / f"benchmark/data/v29_features_v30_096/network_{i}.npz"))
        data.append(
            dict(
                features=np.concatenate([traced["base_features"], traced["features"]], axis=2),
                predictions=traced["predictions"],
                reference=original["reference"],
            )
        )
    control = evaluate(data, list(range(13)))
    candidate = evaluate(data, list(range(15)))
    result = dict(
        control_L03=control,
        candidate_L07=candidate,
        incremental_raw_ratio=candidate["after_raw"] / control["after_raw"],
        adjusted_score=None,
        conclusion="Real per-neuron g4row incremental features with same-cycle13 feature control. Development LOO only; parent reliable-gain and charged candidate gates still apply.",
    )
    path = root / "results/phase2/ablation_v30_102_g4_cv.json"
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(result["incremental_raw_ratio"])


if __name__ == "__main__":
    main()

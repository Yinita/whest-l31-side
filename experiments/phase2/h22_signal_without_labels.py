"""O29 candidate signal ledger without substituting a weak reference."""

import json
from pathlib import Path

import numpy as np


def main():
    candidate = json.loads(Path("results/phase2/ablation_v30_052_r04_mini8_rpc.json").read_text())
    baseline = json.loads(Path("results/phase2/v29_mini8_rpc_predictions.json").read_text())
    rows = []
    for a, b in zip(candidate["per_mlp"], baseline["per_mlp"]):
        assert a["mlp_name"] == b["mlp_name"]
        delta = np.asarray(a["predictions"], dtype=np.float64) - np.asarray(
            b["predictions"], dtype=np.float64
        )
        rows.append(
            dict(
                network_id=a["mlp_index"],
                name=a["mlp_name"],
                final_delta_rms=float(np.sqrt(np.mean(delta[-1] ** 2))),
                final_delta_mean=float(np.mean(delta[-1])),
                layer_delta_rms=np.sqrt(np.mean(delta**2, axis=1)).tolist(),
            )
        )
    p = Path("results/phase2/ablation_v30_091_h22_prediction_signal.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Frozen H22-V29 prediction signal measured without labels. Two independent strong references absent; cannot separate realized reference noise or assign a confidence interval.",
                scope="Candidate delta only; O29 remains partial",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

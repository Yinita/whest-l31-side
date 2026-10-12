"""Offline first-order layer error transport decomposition, oracle only."""

import json
from pathlib import Path

import numpy as np
from datasets import Dataset


def main():
    root = Path(__file__).resolve().parents[2]
    data = Dataset.from_file(
        str(root / "benchmark/data/mini_first8/prepared/mini/data.arrow")
    ).select_columns(["weights", "mlp_name"])
    rows = []
    for index, row in enumerate(data):
        cache = dict(np.load(root / f"benchmark/data/v29_features_v30_096/network_{index}.npz"))
        errors = (cache["reference"] - cache["predictions"]).astype(np.float64)
        weights = np.asarray(row["weights"], dtype=np.float32)
        for layer in range(1, 16):
            transported = cache["features"][layer, :, 7] * (
                weights[layer].astype(np.float64).T @ errors[layer - 1]
            )
            residual = errors[layer] - transported
            rows.append(
                dict(
                    network_id=index,
                    name=row["mlp_name"],
                    layer=layer,
                    error_mse=float(np.mean(errors[layer] ** 2)),
                    transport_mse=float(np.mean(transported**2)),
                    local_residual_mse=float(np.mean(residual**2)),
                    transport_error_cross=float(np.mean(transported * residual)),
                )
            )
    output = root / "results/phase2/ablation_v30_119_error_transport.json"
    assert not output.exists()
    output.write_text(
        json.dumps(
            dict(
                scope="Reference residuals oracle, first-order Phi*(W.T@e_prev); nonlinear truncation and reference noise mixed; not candidate",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

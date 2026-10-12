"""Fixed-binning residual localization using frozen development features."""

import json
from pathlib import Path

import numpy as np


def summarize(error, mask):
    values = error[mask].astype(np.float64)
    return dict(
        count=len(values),
        mean=float(values.mean()) if len(values) else None,
        variance=float(values.var()) if len(values) else None,
        mse=float(np.mean(values**2)) if len(values) else None,
    )


def main():
    root = Path(__file__).resolve().parents[2]
    data = [
        dict(np.load(root / f"benchmark/data/v29_features_v30_096/network_{i}.npz"))
        for i in range(8)
    ]
    alpha_rows = []
    skew_rows = []
    layer_rows = []
    alpha_edges = np.array([-np.inf, -4, -2, -1, 0, 1, 2, 4, np.inf])
    for held, item in enumerate(data):
        f = item["features"][-1]
        e = item["reference"][-1] - item["predictions"][-1]
        for j in range(len(alpha_edges) - 1):
            alpha_rows.append(
                dict(
                    network_id=held,
                    lower=str(alpha_edges[j]),
                    upper=str(alpha_edges[j + 1]),
                    **summarize(e, (f[:, 4] >= alpha_edges[j]) & (f[:, 4] < alpha_edges[j + 1])),
                )
            )
        train_skew = np.concatenate(
            [
                abs(d["features"][-1, :, 9] / d["features"][-1, :, 3] ** 3)
                for i, d in enumerate(data)
                if i != held
            ]
        )
        boundaries = np.quantile(train_skew, [0.25, 0.5, 0.75])
        skew = abs(f[:, 9] / f[:, 3] ** 3)
        bins = np.searchsorted(boundaries, skew, side="right")
        for j in range(4):
            skew_rows.append(
                dict(
                    network_id=held,
                    bin=j,
                    training_boundaries=boundaries.tolist(),
                    **summarize(e, bins == j),
                )
            )
        delta = (item["reference"] - item["predictions"]).astype(np.float64)
        for layer in range(16):
            layer_rows.append(
                dict(
                    network_id=held,
                    layer=layer,
                    raw_mse=float(np.mean(delta[layer] ** 2)),
                    mean_error=float(delta[layer].mean()),
                )
            )
    output = root / "results/phase2/ablation_v30_117_118_error_bins.json"
    assert not output.exists()
    output.write_text(
        json.dumps(
            dict(
                scope="Original development prediction errors; fixedalpha bins and held-network-excluded skew boundaries; localization not causal or candidate gain",
                alpha_rows=alpha_rows,
                skew_rows=skew_rows,
                layer_rows=layer_rows,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

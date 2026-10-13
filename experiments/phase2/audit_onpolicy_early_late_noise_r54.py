"""TRAIN24 marginal MC noise scale versus frozen H18 layer residuals.

Descriptive plug-in delta-method diagnostic; no reference noise subtraction,
model tuning, independence claims or achieved score.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    base = args.directory
    label_path = base / "research_labels.npz"
    labels = np.load(label_path, allow_pickle=False)
    ids = list(range(0, 384, 16))
    assert labels["idx"][:24].tolist() == ids
    raw = np.moveaxis(labels["moments"][:24, :, :15].astype(float), 1, -1)
    m1, m2, m3, m4 = [raw[..., i] for i in range(4)]
    variance = m2 - m1 * m1
    fourth_central = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
    central = np.stack(
        [
            m1,
            variance,
            m3 - 3 * m1 * m2 + 2 * m1**3,
            fourth_central - 3 * variance**2,
        ],
        axis=-1,
    )
    rows = []
    for position, idx in enumerate(ids):
        path = base / f"baseline_idx{idx:04d}.npz"
        trace = np.load(path, allow_pickle=False)
        assert int(trace["idx"]) == idx
        state = trace["states"].astype(float)
        mse = np.mean((central[position] - state) ** 2, axis=1)
        mean_noise = variance[position] / 1e8
        variance_noise = (fourth_central[position] - variance[position] ** 2) / 1e8
        assert np.min(mean_noise) >= 0 and np.min(variance_noise) >= 0
        noise = np.stack([mean_noise, variance_noise], axis=-1).mean(axis=1)
        rows.append(
            dict(
                idx=idx,
                residual_mse=mse[:, :2].tolist(),
                plugin_nominal_mc_marginal_noise=noise.tolist(),
                baseline_trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    mse = np.asarray([row["residual_mse"] for row in rows]).mean(axis=0)
    noise = np.asarray([row["plugin_nominal_mc_marginal_noise"] for row in rows]).mean(axis=0)
    result = dict(
        version="r54-early-versus-late-train24-noise-diagnostic",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        train_indices=ids,
        channels=["mean", "variance"],
        by_layer=[
            dict(
                layer=layer,
                residual_mse=mse[layer].tolist(),
                plugin_nominal_mc_noise=noise[layer].tolist(),
                residual_over_nominal_noise=(mse[layer] / noise[layer]).tolist(),
            )
            for layer in range(15)
        ],
        rows=rows,
        scope="MC N1e8 plug-in marginal moments; shared sample correlations across neurons/layers; no sampling CI, noise subtraction or model tuning",
        labels_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["by_layer"]))


if __name__ == "__main__":
    main()

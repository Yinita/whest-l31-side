"""Immutable L03 remaining folds, unchanged official rows/references per singleton."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from whestbench import load_dataset


def main():
    data = [dict(np.load(f"benchmark/data/v29_features_v30_096/network_{i}.npz")) for i in range(8)]
    dataset = load_dataset("benchmark/data/mini_first8", split="mini")
    root = Path("benchmark/data/l03_singletons")
    root.mkdir(exist_ok=False)
    template = (
        Path("estimators/ablation_v30_098_fold0/estimator.py")
        .read_text()
        .split("\nFOLD0_COEFFICIENT =")[0]
    )
    metadata = json.loads(Path("benchmark/data/mini_first8/metadata.json").read_text())
    provenance = json.loads(Path("benchmark/data/mini_first8/provenance.json").read_text())
    for held in range(1, 8):
        train = [i for i in range(8) if i != held]
        x = np.concatenate([data[i]["features"][-1] for i in train]).astype(np.float64)
        y = np.concatenate(
            [data[i]["reference"][-1] - data[i]["predictions"][-1] for i in train]
        ).astype(np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        constant = std < 1e-12
        mean[constant] = 0
        std[constant] = 1
        z = (x - mean) / std
        beta = np.linalg.solve(z.T @ z + 1e-3 * np.eye(13), z.T @ y)
        text = (
            template
            + "\nFOLD0_COEFFICIENT = "
            + repr((beta / std).astype(np.float32).tolist())
            + "\nFOLD0_INTERCEPT = "
            + repr(float(np.float32(-mean @ (beta / std))))
            + "\n"
        )
        out = Path(f"estimators/ablation_v30_098_fold{held}")
        out.mkdir(exist_ok=False)
        candidate = out / "estimator.py"
        candidate.write_text(text)
        shutil.copy("estimators/k3_v29/LICENSE", out / "LICENSE")
        (out / "provenance.json").write_text(
            json.dumps(
                dict(
                    sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    train_ids=train,
                    allowed_eval_ids=[held],
                    mechanism="Same L03 frozen networkLOO final-only affine correction",
                ),
                indent=2,
            )
            + "\n"
        )
        singleton = root / f"network_{held}"
        singleton.mkdir()
        dataset.select([held]).save_to_disk(str(singleton / "prepared/mini"))
        meta = json.loads(json.dumps(metadata))
        meta["splits"]["mini"]["n_mlps"] = 1
        meta["prepared_splits"] = {"mini": {"path": "prepared/mini"}}
        (singleton / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        prov = dict(
            provenance,
            network_ids=[str(held)],
            n_mlps=1,
            selection="One unchanged official row for matched heldfold; no new reference",
        )
        (singleton / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")


if __name__ == "__main__":
    main()

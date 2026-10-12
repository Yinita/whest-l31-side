"""Materialize pre-registered weights without generating or inspecting labels."""

# ruff: noqa: E402 -- freeze BLAS threads before importing numerical libraries
import hashlib
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import flopscope.numpy as fnp
import numpy as np
from whestbench.generation import sample_mlp


def main():
    protocol_path = Path("results/phase2/ablation_v30_115_independent_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    output = Path("benchmark/data/independent_v30_115")
    output.mkdir(exist_ok=False)
    rows = []
    for index, seed in enumerate(protocol["weight_rng_seeds"]):
        mlp = sample_mlp(
            protocol["width"],
            protocol["depth"],
            rng=fnp.random.default_rng(seed),
            seed=protocol["grader_seeds"][index],
        )
        path = output / f"network_{index}.npz"
        weights = np.stack([np.asarray(w) for w in mlp.weights])
        assert weights.shape == (16, 1024, 1024)
        assert np.isfinite(weights).all()
        np.savez(path, weights=weights, grader_seed=mlp.seed)
        rows.append(
            dict(
                network_id=index,
                weight_seed=seed,
                grader_seed=mlp.seed,
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                mean=float(weights.mean()),
                variance=float(weights.var()),
            )
        )
        print(f"Frozen independent weights {index}", flush=True)
    manifest = dict(
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        rows=rows,
        labels_generated=False,
        status="weights materialized; strong references outstanding",
        scope="Offline data generation, no candidate selection or stage acceptance",
    )
    report = Path("results/phase2/ablation_v30_115_independent_weights.json")
    assert not report.exists()
    report.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

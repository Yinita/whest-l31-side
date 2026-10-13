"""Export immutable, CRC-checked official weights without running an estimator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import zipfile
from pathlib import Path

import flopscope.numpy as fnp
from whestbench import load_dataset, mlp_at


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-mlps", type=int, default=8)
    args = parser.parse_args()
    if args.n_mlps <= 0:
        parser.error("Positive n-mlps required")
    dataset = Path(args.dataset).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        parser.error("Use a fresh output directory")
    provenance = json.loads((dataset / "provenance.json").read_text())
    data = load_dataset(str(dataset), split="mini")
    if len(data) < args.n_mlps:
        parser.error("Insufficient official rows")
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for index in range(args.n_mlps):
        # mlp_at applies the SDK's official estimator seed derivation.
        mlp = mlp_at(data, index)
        path = output / f"mlp_{index}.npz"
        fnp.savez(str(path), weights=fnp.stack(mlp.weights))
        with zipfile.ZipFile(path) as archive:
            corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"Bad exported member {corrupt} in {path}")
        entries.append({"index": index, "name": mlp.name, "seed": mlp.seed,
                        "weights": str(path), "sha256": digest(path)})
        print(f"Exported network {index}: {mlp.name}", flush=True)
    manifest = {"dataset": str(dataset), "dataset_commit": provenance["revision"],
                "dataset_provenance_sha256": digest(dataset / "provenance.json"),
                "script_sha256": digest(Path(__file__)),
                "versions": {k: importlib.metadata.version(k)
                             for k in ("flopscope", "whestbench")},
                "n_mlps": args.n_mlps, "complete": True,
                "scope": "Input cache only; no candidate or reference-based selection"}
    (output / "inputs.json").write_text(json.dumps(entries, indent=2) + "\n")
    manifest["inputs_sha256"] = digest(output / "inputs.json")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

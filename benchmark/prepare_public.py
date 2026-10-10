"""Cache one pinned official Phase 2 Arrow shard without changing its targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download

REPO = "aicrowd/arc-whestbench-public-2026"
REVISION = "aa99830fdc09fad15407b10e8e3459d3e18bba0a"
FILE = "prepared/mini/data-00000-of-00014.arrow"
ROOT = Path(__file__).resolve().parent / "data" / "mini_first8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=range(14), default=0)
    args = parser.parse_args()
    root = ROOT if args.shard == 0 else ROOT.parent / f"mini_shard{args.shard:02d}"
    file = f"prepared/mini/data-{args.shard:05d}-of-00014.arrow"
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi().repo_info(REPO, repo_type="dataset", revision=REVISION, files_metadata=True)
    entry = next(f for f in info.siblings if f.rfilename == file)
    path = root / "prepared" / "mini" / "data.arrow"
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.lfs.sha256
    valid = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    if not valid:
        chunk = 8 * 1024 * 1024
        ranges = [(i, min(i + chunk - 1, entry.size - 1)) for i in range(0, entry.size, chunk)]
        staging = path.with_suffix(".partial")
        parts = path.parent / f"parts-{expected}"
        parts.mkdir(exist_ok=True)

        def download(bounds):
            start, end = bounds
            part = parts / f"{start}-{end}.bin"
            digest = part.with_suffix(".sha256")
            if part.exists() and digest.exists() and part.stat().st_size == end - start + 1:
                if hashlib.sha256(part.read_bytes()).hexdigest() == digest.read_text().strip():
                    return part.stat().st_size
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{file}?part={start}"
            for attempt in range(3):
                try:
                    with httpx.Client(timeout=90, follow_redirects=True) as client:
                        response = client.get(url, headers={"Range": f"bytes={start}-{end}"})
                        response.raise_for_status()
                        content = response.content
                        if response.status_code != 206 or len(content) != end - start + 1:
                            raise RuntimeError(f"Unexpected range response for offset {start}")
                        temporary = part.with_suffix(".tmp")
                        temporary.write_bytes(content)
                        temporary.replace(part)
                        digest.write_text(hashlib.sha256(content).hexdigest() + "\n")
                    break
                except (httpx.TransportError, httpx.HTTPStatusError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            return len(content)

        total = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(download, bounds) for bounds in ranges]
            for future in as_completed(futures):
                total += future.result()
                print(f"Downloaded {total / 1e6:.0f}/{entry.size / 1e6:.0f} MB", flush=True)
        with staging.open("wb") as out:
            for start, end in ranges:
                out.write((parts / f"{start}-{end}.bin").read_bytes())
        actual = hashlib.sha256(staging.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Official shard SHA-256 mismatch")
        staging.replace(path)
    dataset = Dataset.from_file(str(path))
    if not len(dataset):
        raise RuntimeError("Empty official shard")
    dataset.save_to_disk(str(root / "prepared" / "mini8"))
    metadata_path = hf_hub_download(REPO, "metadata.json", repo_type="dataset", revision=REVISION)
    metadata = json.loads(Path(metadata_path).read_text())
    metadata["splits"] = {"mini": metadata["splits"]["mini"]}
    metadata["splits"]["mini"]["n_mlps"] = len(dataset)
    metadata["prepared_splits"] = {"mini": {"path": "prepared/mini8"}}
    metadata["default_split"] = "mini"
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    provenance = {
        "repo": REPO,
        "revision": REVISION,
        "file": file,
        "sha256": expected,
        "selection": "first 8 official Mini rows; unchanged targets" if args.shard == 0 else f"official Mini shard {args.shard}; unchanged targets; reserved from local candidate tuning",
        "network_ids": [str(x) for x in dataset["mlp_id"]],
        "independence_caveat": "Public networks; separation from upstream author tuning has not been established.",
        "n_mlps": len(dataset),
        "columns": dataset.column_names,
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = Path(__file__).resolve().parents[1] / "results" / "phase2" / ("mini8_dataset.json" if args.shard == 0 else f"mini_shard{args.shard:02d}_dataset.json")
    report.write_text(json.dumps(provenance, indent=2) + "\n")
    print("Prepared", root)


if __name__ == "__main__":
    main()

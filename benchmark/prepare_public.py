"""Cache the first official Phase 2 Arrow shard as an auditable Mini-8 subset."""

from __future__ import annotations

import hashlib
import json
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
    ROOT.mkdir(parents=True, exist_ok=True)
    info = HfApi().repo_info(REPO, repo_type="dataset", revision=REVISION, files_metadata=True)
    entry = next(f for f in info.siblings if f.rfilename == FILE)
    path = ROOT / "prepared" / "mini" / "data.arrow"
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.lfs.sha256
    valid = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    if not valid:
        chunk = 8 * 1024 * 1024
        ranges = [(i, min(i + chunk - 1, entry.size - 1)) for i in range(0, entry.size, chunk)]
        staging = path.with_suffix(".partial")
        with staging.open("wb") as out:
            out.truncate(entry.size)

        def download(bounds):
            start, end = bounds
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILE}?part={start}"
            with httpx.Client(timeout=90, follow_redirects=True) as client:
                response = client.get(url, headers={"Range": f"bytes={start}-{end}"})
                response.raise_for_status()
                content = response.content
                if response.status_code != 206 or len(content) != end - start + 1:
                    raise RuntimeError(f"Unexpected range response for offset {start}")
                with staging.open("r+b") as out:
                    out.seek(start)
                    out.write(content)
            return len(content)

        total = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(download, bounds) for bounds in ranges]
            for future in as_completed(futures):
                total += future.result()
                print(f"Downloaded {total / 1e6:.0f}/{entry.size / 1e6:.0f} MB", flush=True)
        actual = hashlib.sha256(staging.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Official shard SHA-256 mismatch")
        staging.replace(path)
    dataset = Dataset.from_file(str(path))
    if len(dataset) != 8:
        raise RuntimeError(f"Unexpected first shard size: {len(dataset)}")
    dataset.save_to_disk(str(ROOT / "prepared" / "mini8"))
    metadata_path = hf_hub_download(REPO, "metadata.json", repo_type="dataset", revision=REVISION)
    metadata = json.loads(Path(metadata_path).read_text())
    metadata["splits"] = {"mini": metadata["splits"]["mini"]}
    metadata["splits"]["mini"]["n_mlps"] = 8
    metadata["prepared_splits"] = {"mini": {"path": "prepared/mini8"}}
    metadata["default_split"] = "mini"
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    provenance = {
        "repo": REPO,
        "revision": REVISION,
        "file": FILE,
        "sha256": expected,
        "selection": "first 8 official Mini rows; unchanged targets",
        "n_mlps": len(dataset),
        "columns": dataset.column_names,
    }
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = Path(__file__).resolve().parents[1] / "results" / "phase2" / "mini8_dataset.json"
    report.write_text(json.dumps(provenance, indent=2) + "\n")
    print("Prepared", ROOT)


if __name__ == "__main__":
    main()

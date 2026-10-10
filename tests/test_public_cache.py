"""Verify resumable data chunks without network or changing official targets."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("corrupt", [False, True])
def test_resume_checks_chunk_digest(tmp_path, monkeypatch, corrupt):
    spec = importlib.util.spec_from_file_location("public_cache", Path(__file__).parents[1] / "benchmark/prepare_public.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    content = b"official-arrow-fixture"
    sha = hashlib.sha256(content).hexdigest()
    root = tmp_path / "cache"
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"splits": {"mini": {}}}))
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "HfApi", lambda: SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(siblings=[SimpleNamespace(rfilename=module.FILE, size=len(content), lfs=SimpleNamespace(sha256=sha))])))
    monkeypatch.setattr(module, "hf_hub_download", lambda *a, **k: str(metadata))
    monkeypatch.setattr(module, "__file__", str(tmp_path / "benchmark/prepare_public.py"))
    (tmp_path / "results/phase2").mkdir(parents=True)
    monkeypatch.setattr("sys.argv", ["prepare_public.py"])

    class Dataset:
        column_names = ["mlp_id"]
        def __len__(self):
            return 1
        def __getitem__(self, key):
            return [123]
        def save_to_disk(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module.Dataset, "from_file", lambda path: Dataset())
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url, headers):
            calls.append(url)
            return SimpleNamespace(status_code=206, content=content, raise_for_status=lambda: None)

    monkeypatch.setattr(module.httpx, "Client", Client)
    module.main()
    arrow = root / "prepared/mini/data.arrow"
    assert arrow.read_bytes() == content
    assert len(calls) == 1
    arrow.unlink()  # Simulate interrupted final assembly, retaining validated chunks.
    if corrupt:
        next((arrow.parent / f"parts-{sha}").glob("*.bin")).write_bytes(b"x" * len(content))
    module.main()
    assert arrow.read_bytes() == content
    assert len(calls) == (2 if corrupt else 1)

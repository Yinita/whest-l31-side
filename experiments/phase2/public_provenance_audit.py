"""O26 upstream training disclosure audit from frozenpublic README only."""

import hashlib
import json
from pathlib import Path


def main():
    source = Path("/tmp/whest-p2-k3-reference/README.md")
    text = source.read_text()
    words = ("fit", "train", "public", "mini", "calibrat", "network", "tune")
    hits = [
        dict(line=i, text=line)
        for i, line in enumerate(text.splitlines(), 1)
        if any(w in line.lower() for w in words)
    ]
    path = Path("results/phase2/ablation_v30_088_upstream_split_audit.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                hits=hits,
                conclusion="Frozenpublic README disclosure excerpts only. Exactupstream per-network trainingIDs and heldout separation are not established. Publicreservedshard cannot be asserted unseen to author; localL21 remains sealed until chosen candidate.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

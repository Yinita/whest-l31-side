"""Record network-level selection policy without reading confirmation targets."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/phase2/ablation_v30_097_selection_protocol.json"
    assert not output.exists()
    provenance = json.loads((root / "benchmark/data/mini_shard01/provenance.json").read_text())
    protocol = dict(
        development_ids=list(range(8)),
        folds=[
            dict(train=[i for i in range(8) if i != held], validation=[held]) for held in range(8)
        ],
        public_confirmation_ids=list(range(8, 16)),
        selection="Network-equal cross-validation adjusted score; freeze one candidate hash before confirmation; no neuron-level split",
        final_unseen_ids=None,
        independence_caveat="Upstream V29 tuned on public dumps; public shard01 upstream independence unknown",
        confirmation_target_access="This script reads only provenance, no confirmation target arrays; does not establish absence of access by upstream or other historical code",
        confirmation_provenance_sha256=hashlib.sha256(
            (root / "benchmark/data/mini_shard01/provenance.json").read_bytes()
        ).hexdigest(),
        data_commit=provenance["revision"],
        status="Selection policy frozen; feature cache L01 prerequisite remains pending",
    )
    output.write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""L02/L33 audit model fold exclusions, candidate immutable hashes and feature shape."""

import hashlib
import json
from pathlib import Path


def main():
    rows = []
    for held in range(8):
        folder = Path(f"estimators/ablation_v30_098_fold{held}")
        provenance = json.loads((folder / "provenance.json").read_text())
        assert held not in provenance["train_ids"]
        assert set(provenance["train_ids"]) == set(range(8)) - {held}
        assert provenance["allowed_eval_ids"] == [held]
        actual = hashlib.sha256((folder / "estimator.py").read_bytes()).hexdigest()
        assert actual == provenance["sha256"]
        rows.append(dict(held=held, train=provenance["train_ids"], sha256=actual))
    output = Path("results/phase2/ablation_v30_097_128_loo_source_audit.json")
    assert not output.exists()
    output.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="All8 candidate train/eval exclusions and immutable source hashes verified. Provenance proof is separate from dataset/reference independence and whole8fold acceptance; public confirmation unopened for score.",
                scope="Read-only frozen model contract",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

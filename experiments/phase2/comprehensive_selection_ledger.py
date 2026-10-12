"""L35 evidence selection ledger including prequeue and offline comparisons."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    entries = []
    for path in sorted((root / "results/phase2").glob("*.json")):
        if not path.stat().st_size or "selection_ledger" in path.name:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        record = dict(
            path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        if (
            isinstance(data, dict)
            and "results" in data
            and isinstance(data["results"], dict)
            and "adjusted_final_layer_score" in data["results"]
        ):
            r = data["results"]
            record.update(
                type="benchmark",
                adjusted=r["adjusted_final_layer_score"],
                raw=r.get("final_layer_mse"),
                failures=r.get("n_failed_mlps"),
                network_count=len(r.get("per_mlp", [])),
            )
        elif any(tag in path.name for tag in ["loocv", "_cv", "predictability", "extra_r01"]):
            record.update(type="offline_selection", official_adjusted=None)
        else:
            continue
        entries.append(record)
    path = root / "results/phase2/ablation_v30_129_r02_comprehensive_selection_ledger.json"
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                entries=entries,
                counts={
                    "benchmark": sum(r["type"] == "benchmark" for r in entries),
                    "offline_selection": sum(r["type"] == "offline_selection" for r in entries),
                },
                conclusion="All saved benchmark and identifiable offlineCV reports indexed, including negative/failure and prequeue screens. Report count is not independent hypotheses; no effective-test-count statistical guarantee. Independent confirmation labels remain unopened.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""129-item evidence gap inventory with real candidate/report presence."""

import hashlib
import json
from pathlib import Path


def main():
    q = json.loads(Path("experiments/phase2/ablation_queue.json").read_text())
    rows = []
    for t in q["tasks"]:
        path = Path(t["candidate_path"]) if t.get("candidate_path") else None
        exists = bool(path and path.exists())
        rows.append(
            dict(
                key=t["key"],
                version=t["version"],
                status=t["status"],
                kind=t["kind"],
                candidate_exists=exists,
                candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if exists else None,
                evidence_count=len(t.get("evidence", [])),
                full_score_report=t.get("result", {}).get("after_report"),
                acceptance=t["acceptance"],
            )
        )
    p = Path("results/phase2/ablation_gap_inventory_20261012.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                total=len(rows),
                conclusion="129 tracked items. Candidate presence/evidence count never proves completion; each acceptance gate retained. Partial items require actual remaining implementation/validation.",
                scope="Read-only completeness audit",
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

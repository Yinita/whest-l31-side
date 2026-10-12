"""Inventory all recorded queue runs without discarding negatives."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    queue = json.loads((root / "experiments/phase2/ablation_queue.json").read_text())
    entries = []
    for task in queue["tasks"]:
        for result in task.get("previous_results", []) + [task.get("result", {})]:
            if not result.get("after_report"):
                continue
            path = root / result["after_report"]
            entries.append(
                dict(
                    key=task["key"],
                    version=task["version"],
                    candidate_sha256=task.get("candidate_sha256"),
                    report=result["after_report"],
                    report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    scope=result.get("scope"),
                    score=result["after"]["adjusted_final_layer_score"],
                    independent_final_set=False,
                )
            )
    output = root / "results/phase2/ablation_v30_129_selection_ledger.json"
    assert not output.exists(), 'Use a fresh ledger revision'
    output.write_text(
        json.dumps(
            dict(
                scope="Recorded backlog benchmark reports only; earlier research screens outside queue remain in RESEARCH_LOG",
                candidate_runs=len(entries),
                entries=entries,
                stage1_target=5.779e-10,
                selection_warning="Repeated developer comparisons induce selection bias; true unseen final set not yet prepared",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

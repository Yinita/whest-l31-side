"""L35 exhaustive current queue plus separately attached RPC/CV evidence ledger."""

import hashlib
import json
from pathlib import Path


def main():
    q = json.loads(Path("experiments/phase2/ablation_queue.json").read_text())
    rows = []
    for task in q["tasks"]:
        for evidence in task.get("evidence", []):
            path = Path(evidence)
            if not path.exists() or path.suffix != ".json" or not path.stat().st_size:
                continue
            report = json.loads(path.read_text())
            results = report.get("results", report)
            if "adjusted_final_layer_score" in results:
                rows.append(
                    dict(
                        key=task["key"],
                        version=task["version"],
                        status=task["status"],
                        report=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        adjusted=results["adjusted_final_layer_score"],
                        raw=results.get("final_layer_mse"),
                        failures=results.get("n_failed_mlps"),
                        scope=results.get("mode", task.get("result", {}).get("scope")),
                    )
                )
    p = Path("results/phase2/ablation_v30_129_current_score_ledger_r03.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                count=len(rows),
                conclusion="Evidence-report inventory, not independent hypothesis count. Components and actualfold results remain separately named; no missingscore inferred and no reference independence asserted.",
                scope="Read-only current ledger checkpoint",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

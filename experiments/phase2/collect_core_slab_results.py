"""Record full Mini8 slab revisions without overwriting historical rows."""

import hashlib
import json
from pathlib import Path


def main():
    base = json.loads(Path("results/phase2/v29_mini8_local.json").read_text())["results"]
    path = Path("experiments/phase2/ablation_queue.json")
    q = json.loads(path.read_text())
    for version, key in [("017", "C17"), ("018", "C18")]:
        report = Path(f"results/phase2/ablation_v30_{version}_r02_mini8_local.json")
        after = json.loads(report.read_text())["results"]
        t = next(t for t in q["tasks"] if t["key"] == key)
        t.setdefault("previous_results", []).append(t.get("result", {}))
        t.setdefault("previous_evidence", []).append(t.get("evidence", []))
        before = dict(base, flops_used=sum(x["flops_used"] for x in base["per_mlp"]))
        current = dict(after, flops_used=sum(x["flops_used"] for x in after["per_mlp"]))
        ratios = {
            k: current[k] / before[k]
            for k in ["final_layer_mse", "adjusted_final_layer_score", "flops_used"]
        }
        t["result"] = dict(
            before_report="results/phase2/v29_mini8_local.json",
            after_report=str(report),
            before=before,
            after=current,
            ratios=ratios,
            failures=after["n_failed_mlps"],
            scope="Mini8 development; no independent confirmation",
        )
        t["status"] = (
            "completed_positive"
            if not after["n_failed_mlps"] and ratios["adjusted_final_layer_score"] < 1
            else "completed_negative"
            if not after["n_failed_mlps"]
            else "completed_failed"
        )
        candidate = Path(f"estimators/ablation_v30_{version}_r02/estimator.py")
        t["candidate_path"] = str(candidate)
        t["candidate_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        t["evidence"] = [
            str(report),
            str(report.with_suffix(".manifest.json")),
            str(report.with_suffix(".log")),
        ]
        with Path("experiments/phase2/RESEARCH_LOG.md").open("a") as log:
            log.write(
                "\n"
                + key
                + "r02 Mini8 "
                + json.dumps(dict(ratios=ratios, failures=after["n_failed_mlps"]))
                + "\n"
            )
    path.write_text(json.dumps(q, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

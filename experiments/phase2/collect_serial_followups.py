"""Finalize actual serial diagnostic evidence after resource report exists."""

import json
from pathlib import Path


def main():
    queue_path = Path("experiments/phase2/ablation_queue.json")
    q = json.loads(queue_path.read_text())
    distribution = Path("results/phase2/ablation_v30_089_distribution_shift.json")
    resource = Path("results/phase2/ablation_v30_030_r02_c01_mini8_subprocess.json")
    if distribution.exists():
        t = next(t for t in q["tasks"] if t["key"] == "O27")
        t["status"] = "completed_diagnostic"
        t.setdefault("evidence", []).extend(
            [str(distribution), "results/phase2/ablation_v30_089_reserved_state_only.json"]
        )
        t["result"] = {
            "conclusion": "Frozen state-only feature quantiles on 8 development and 8 reserved public networks compared without labels; descriptive distribution shift, not score or independent generalization",
            "score_applicable": False,
        }
    if resource.exists() and resource.stat().st_size:
        r = json.loads(resource.read_text())["results"]
        t = next(t for t in q["tasks"] if t["key"] == "C30")
        t["status"] = "completed_diagnostic"
        t.setdefault("evidence", []).append(str(resource))
        t["result"]["subprocess_failures"] = r["n_failed_mlps"]
        t["result"]["subprocess_adjusted"] = r["adjusted_final_layer_score"]
        t["result"]["conclusion"] = (
            "C01 samehash full Mini8 math/local, RPC and originalcap subprocess measured; preserve resource failures, development server cap not official deployment proof"
        )
    queue_path.write_text(json.dumps(q, ensure_ascii=False, indent=2) + "\n")
    state_path = Path("experiments/phase2/research_state.json")
    s = json.loads(state_path.read_text())
    if distribution.exists():
        s["reserved_state_followup"]["status"] = "completed target-free distribution diagnostic"
    if resource.exists() and resource.stat().st_size:
        s["resource_followup"]["status"] = "completed diagnostic; inspect retained failures"
    state_path.write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n")
    with Path("experiments/phase2/RESEARCH_LOG.md").open("a") as log:
        log.write(
            "\nSerial followup collection: reserved distribution="
            + str(distribution.exists())
            + ", originalcap C01 subprocess="
            + str(resource.exists())
            + ". Actual reports retained; resource failures do not imply algorithm score improvement or stage acceptance.\n"
        )


if __name__ == "__main__":
    main()

"""Actual heldnetwork0 score, preserving full-LOO acceptance requirement."""

import json
from pathlib import Path


def main():
    path = Path("results/phase2/ablation_v30_098_r03_fold0_mini1_local.json")
    result = json.loads(path.read_text())["results"]
    base = json.loads(Path("results/phase2/v29_mini1_control_20261012.json").read_text())["results"]
    before = base["per_mlp"][0]
    after = result["per_mlp"][0]
    qpath = Path("experiments/phase2/ablation_queue.json")
    q = json.loads(qpath.read_text())
    task = next(t for t in q["tasks"] if t["key"] == "L03")
    task.setdefault("evidence", []).append(str(path))
    task["charged_fold0_result"] = dict(
        before_raw=before["final_layer_mse"],
        after_raw=after["final_layer_mse"],
        before_adjusted=before["adjusted_final_layer_score"],
        after_adjusted=after["adjusted_final_layer_score"],
        before_flops=before["flops_used"],
        after_flops=after["flops_used"],
        failures=result["n_failed_mlps"],
        train_ids=list(range(1, 8)),
        held_id=0,
        scope="Single held-network fold; not full8fold charged CV or independent final",
    )
    qpath.write_text(json.dumps(q, ensure_ascii=False, indent=2) + "\n")
    with Path("experiments/phase2/RESEARCH_LOG.md").open("a") as f:
        f.write(
            "\nL03 actualcharged heldfold0: "
            + json.dumps(task["charged_fold0_result"], ensure_ascii=False)
            + "\n"
        )


if __name__ == "__main__":
    main()

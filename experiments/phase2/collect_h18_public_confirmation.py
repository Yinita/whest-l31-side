"""Record preregistered paired public confirmation without independent claims."""

import json
from pathlib import Path


def main():
    output = Path("results/phase2/ablation_v30_116_r01_confirmation_decision.json")
    assert not output.exists()
    before_path = Path("results/phase2/ablation_v30_116_r01_paired_v29_public.json")
    after_path = Path("results/phase2/ablation_v30_116_r01_h18_public.json")
    before = json.loads(before_path.read_text())["results"]
    after = json.loads(after_path.read_text())["results"]
    rows = []
    for b, a in zip(before["per_mlp"], after["per_mlp"], strict=True):
        assert b["mlp_name"] == a["mlp_name"]
        rows.append(
            dict(
                name=b["mlp_name"],
                before_raw=b["final_layer_mse"],
                after_raw=a["final_layer_mse"],
                before_adjusted=b["adjusted_final_layer_score"],
                after_adjusted=a["adjusted_final_layer_score"],
                before_flops=b["flops_used"],
                after_flops=a["flops_used"],
            )
        )
    result = dict(
        before_raw=before["final_layer_mse"],
        after_raw=after["final_layer_mse"],
        before_adjusted=before["adjusted_final_layer_score"],
        after_adjusted=after["adjusted_final_layer_score"],
        failures=after["n_failed_mlps"],
        control_failures=before["n_failed_mlps"],
        rows=rows,
        scope="Public confirmation; O27 states previously inspected; upstream tuning overlap unknown; not independent stage acceptance",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    queue_path = Path("experiments/phase2/ablation_queue.json")
    queue = json.loads(queue_path.read_text())
    task = next(t for t in queue["tasks"] if t["key"] == "L21")
    task["status"] = "completed_diagnostic"
    task["evidence"] = [
        str(before_path),
        str(after_path),
        str(output),
        "results/phase2/ablation_v30_116_r01_preregistration.json",
    ]
    task["public_confirmation"] = result
    task["result"] = {
        "conclusion": "Paired public confirmation executed with failures retained; reliable benefit decision in decision report, no independent acceptance"
    }
    queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n")
    with Path("experiments/phase2/RESEARCH_LOG.md").open("a") as f:
        f.write("\nL21 H18 public confirmation " + json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

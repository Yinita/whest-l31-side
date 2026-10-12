"""Collect actual per-fold charged reports without treating in-sample rows as CV."""

import json
from pathlib import Path

import numpy as np


def main():
    rows = []
    for held in range(8):
        if held == 0:
            bp = Path("results/phase2/v29_mini1_control_20261012.json")
            cp = Path("results/phase2/ablation_v30_098_r03_fold0_mini1_local.json")
        else:
            bp = Path(f"results/phase2/l03_fold{held}_paired_v29_control.json")
            cp = Path(f"results/phase2/l03_fold{held}_charged_candidate.json")
        b = json.loads(bp.read_text())["results"]
        c = json.loads(cp.read_text())["results"]
        before = b["per_mlp"][0]
        after = c["per_mlp"][0]
        assert before["mlp_name"] == after["mlp_name"]
        rows.append(
            dict(
                held_id=held,
                control_report=str(bp),
                candidate_report=str(cp),
                before_raw=before["final_layer_mse"],
                after_raw=after["final_layer_mse"],
                before_adjusted=before["adjusted_final_layer_score"],
                after_adjusted=after["adjusted_final_layer_score"],
                before_flops=before["flops_used"],
                after_flops=after["flops_used"],
                failures=c["n_failed_mlps"],
            )
        )
    output = Path("results/phase2/ablation_v30_098_r04_full_charged_cv.json")
    assert not output.exists()
    result = dict(
        rows=rows,
        before_raw=float(np.mean([r["before_raw"] for r in rows])),
        after_raw=float(np.mean([r["after_raw"] for r in rows])),
        before_adjusted=float(np.mean([r["before_adjusted"] for r in rows])),
        after_adjusted=float(np.mean([r["after_adjusted"] for r in rows])),
        failures=sum(r["failures"] for r in rows),
        scope="Development networkLOO, frozenmodels trained excluding each held; no independent final confirmation",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    qp = Path("experiments/phase2/ablation_queue.json")
    q = json.loads(qp.read_text())
    t = next(t for t in q["tasks"] if t["key"] == "L03")
    t.setdefault("evidence", []).append(str(output))
    t["charged_full_cv"] = result
    t["status"] = (
        "completed_negative"
        if result["after_adjusted"] >= result["before_adjusted"] and not result["failures"]
        else "completed_positive"
        if not result["failures"]
        else "completed_failed"
    )
    t["result"]["conclusion"] = (
        "Actual charged full8fold CV completed; independent/RPC validation required for positive adoption, negative can close tested model."
    )
    qp.write_text(json.dumps(q, ensure_ascii=False, indent=2) + "\n")
    with Path("experiments/phase2/RESEARCH_LOG.md").open("a") as f:
        f.write("\nL03 fullchargedCV " + json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

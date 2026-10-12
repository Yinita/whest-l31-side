"""Collect only completed serial artifacts; never promote unpassed gates."""

import json
from pathlib import Path


def main():
    path = Path("experiments/phase2/ablation_queue.json")
    q = json.loads(path.read_text())
    rpc = Path("results/phase2/ablation_v30_030_r01_c01_mini8_rpc_scored.json")
    if rpc.exists():
        r = json.loads(rpc.read_text())
        t = next(t for t in q["tasks"] if t["key"] == "C30")
        t["status"] = "partial_diagnostic"
        t["evidence"] = [str(rpc), "results/phase2/ablation_v30_001_r02_mini8_local.json"]
        t["result"] = {
            "conclusion": "C01 full Mini8 local/RPC measured; ordinary subprocess original resource-cap diagnostic still pending",
            "rpc_adjusted": r["adjusted_final_layer_score"],
            "rpc_raw": r["final_layer_mse"],
            "rpc_failures": r["n_failed_mlps"],
            "score_applicable": False,
        }
    pilot = Path("results/phase2/ablation_v30_114_reference_cpu_pilot.json")
    if pilot.exists():
        t = next(t for t in q["tasks"] if t["key"] == "L19")
        if str(pilot) not in t.setdefault("evidence", []):
            t["evidence"].append(str(pilot))
        t["reference_status"] = (
            "Measured weak CPU pilot only; no strong reference or stage acceptance"
        )
    path.write_text(json.dumps(q, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

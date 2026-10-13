"""Compare frozen candidates on an identical strict-RPC official cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Fresh output required")
    paths = [Path(args.before), Path(args.after)]
    before, after = [json.loads(path.read_text()) for path in paths]
    for report in (before, after):
        if report["mode"] != "strict local official-client RPC score" or not report["manifest"]["completed"]:
            raise ValueError("Only completed strict-RPC reports can be compared")
        if len(report["per_mlp"]) != report["n_mlps"]:
            raise ValueError("Missing network rows")
        manifest = report["manifest"]
        command = manifest["command"]
        predictions = Path(command[command.index("--output") + 1])
        if digest(predictions) != report["predictions_sha256"]:
            raise ValueError("Raw predictions report hash differs")
        estimator = Path(command[command.index("--estimator") + 1])
        if digest(estimator) != manifest["estimator_sha256"]:
            raise ValueError("Candidate changed since frozen evaluation")
    for key in ("dataset_commit", "dataset_provenance", "seed_protocol_version",
                "limits", "versions", "n_mlps", "score_script_sha256"):
        if before[key] != after[key]:
            raise ValueError(f"Cohort/protocol mismatch: {key}")
    for key in ("inputs_sha256", "supervisor_sha256", "worker_sha256", "versions",
                "server_max_array_bytes", "network_ids"):
        if before["manifest"][key] != after["manifest"][key]:
            raise ValueError(f"Runner/input mismatch: {key}")
    rows = []
    for b, a in zip(before["per_mlp"], after["per_mlp"]):
        for key in ("mlp_index", "mlp_name", "seed"):
            if b[key] != a[key]:
                raise ValueError(f"Network identity mismatch: {key}")
        row = {key: b[key] for key in ("mlp_index", "mlp_name", "seed")}
        fields = ("final_layer_mse", "adjusted_final_layer_score", "flops_used", "wall_time_s",
                  "residual_wall_time_s", "client_peak_rss_kib", "limits_passed", "failure_reasons")
        row["before"] = {key: b[key] for key in fields}
        row["after"] = {key: a[key] for key in fields}
        row["relative_raw"] = a["final_layer_mse"] / b["final_layer_mse"] - 1 if b["final_layer_mse"] else None
        row["relative_adjusted"] = a["adjusted_final_layer_score"] / b["adjusted_final_layer_score"] - 1 if b["adjusted_final_layer_score"] else None
        rows.append(row)
    metrics = ("final_layer_mse", "adjusted_final_layer_score", "n_failed_mlps", "mean_compute_utilization")
    result = {"scope": "Frozen paired public cohort, both strict RPC; upstream public tuning unknown; not final unseen holdout",
              "before_report": str(paths[0]), "after_report": str(paths[1]),
              "before_report_sha256": digest(paths[0]), "after_report_sha256": digest(paths[1]),
              "before_estimator_sha256": before["manifest"]["estimator_sha256"],
              "after_estimator_sha256": after["manifest"]["estimator_sha256"],
              "protocol_checks_passed": True, "n_mlps": before["n_mlps"],
              "dataset_commit": before["dataset_commit"], "inputs_sha256": before["manifest"]["inputs_sha256"],
              "before": {key: before[key] for key in metrics},
              "after": {key: after[key] for key in metrics},
              "relative_adjusted": after["adjusted_final_layer_score"] / before["adjusted_final_layer_score"] - 1,
              "relative_raw": after["final_layer_mse"] / before["final_layer_mse"] - 1,
              "per_mlp": rows, "stage_accepted": False}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("n_mlps", "before", "after", "relative_adjusted", "relative_raw")}, indent=2))


if __name__ == "__main__":
    main()

"""Aggregate completed strict-RPC cohorts for one frozen candidate without duplicates."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(paths):
    reports = [json.loads(path.read_text()) for path in paths]
    first = reports[0]
    rows, inputs, seen = [], [], set()
    estimator_hash = first["manifest"]["estimator_sha256"]
    for path, report in zip(paths, reports):
        manifest = report["manifest"]
        if report["mode"] != "strict local official-client RPC score" or not manifest["completed"]:
            raise ValueError("Completed strict-RPC scores required")
        if len(report["per_mlp"]) != report["n_mlps"]:
            raise ValueError("Missing network rows")
        for key in ("dataset_commit", "seed_protocol_version", "limits", "versions", "score_script_sha256"):
            if report[key] != first[key]:
                raise ValueError(f"Protocol mismatch: {key}")
        for key in ("estimator_sha256", "supervisor_sha256", "worker_sha256", "server_max_array_bytes", "versions"):
            if manifest[key] != first["manifest"][key]:
                raise ValueError(f"Candidate/runner mismatch: {key}")
        command = manifest["command"]
        if digest(Path(command[command.index("--estimator")+1])) != estimator_hash:
            raise ValueError("Candidate source changed")
        if digest(Path(command[command.index("--output")+1])) != report["predictions_sha256"]:
            raise ValueError("Raw predictions changed")
        for row in report["per_mlp"]:
            identity = (row["mlp_name"], row["seed"])
            if identity in seen:
                raise ValueError("Duplicate public network")
            seen.add(identity)
            rows.append(dict(row, cohort_report=str(path)))
        inputs.append({"path": str(path), "sha256": digest(path), "n_mlps": report["n_mlps"]})
    count = len(rows)
    return {
        "scope": "Same-candidate strict-RPC public coverage; public/upstream tuning unknown; not independent holdout or official deployment certification",
        "estimator_sha256": estimator_hash, "cohort_reports": inputs,
        "dataset_commit": first["dataset_commit"], "versions": first["versions"], "limits": first["limits"],
        "n_mlps": count, "n_failed_mlps": sum(not row["limits_passed"] for row in rows),
        "final_layer_mse": sum(row["final_layer_mse"] for row in rows)/count,
        "adjusted_final_layer_score": sum(row["adjusted_final_layer_score"] for row in rows)/count,
        "mean_flops": sum(row["flops_used"] for row in rows)/count,
        "max_wall_s": max(row["wall_time_s"] for row in rows),
        "max_residual_s": max(row["residual_wall_time_s"] for row in rows),
        "max_client_peak_rss_kib": max(row["client_peak_rss_kib"] for row in rows),
        "per_mlp": rows, "stage_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Fresh output required")
    result = aggregate([Path(p) for p in args.reports])
    result["aggregate_script_sha256"] = digest(Path(__file__))
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("n_mlps", "n_failed_mlps", "final_layer_mse", "adjusted_final_layer_score", "max_wall_s", "max_residual_s")}))


if __name__ == "__main__":
    main()

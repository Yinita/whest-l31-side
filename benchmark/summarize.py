"""Summarize observed local/online evidence without claiming same-suite parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def summarize(local, online, runners, submission):
    score = online["response"].get("score")
    status = online["response"].get("grading_status_cd")
    local_score = local["results"]["adjusted_final_layer_score"]
    delta = None
    if status == "graded" and score is not None:
        delta = float(score) / local_score - 1 if local_score else None
    return {
        "submission_id": submission["submission_id"],
        "round_id": online["response"].get("challenge_round_id"),
        "online_status": status,
        "online_adjusted_score": score,
        "online_score_secondary_as_reported": online["response"].get("score_secondary"),
        "local_n_mlps": len(local["results"]["per_mlp"]),
        "local_adjusted_score": local_score,
        "local_final_layer_mse": local["results"]["final_layer_mse"],
        "local_compute_utilization": local["results"]["mean_compute_utilization"],
        "local_failed_mlps": local["results"]["n_failed_mlps"],
        "runner_same_flops": runners["same_flops"],
        "runner_max_relative_mse_delta": runners["max_relative_mse_delta"],
        "runner_failed_mlps": {"left": runners["left_failed"], "right": runners["right_failed"]},
        "online_vs_local_aggregate_relative_delta": delta,
        "online_same_network_parity_verified": False,
        "interpretation": "Mini subset vs online suite: aggregate diagnostic only; network identities/reference and grader tool versions not reported by status API.",
        "submitted_estimator_sha256": submission["estimator_sha256"],
        "submitted_artifact_sha256": submission["artifact_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("local", "online", "runners", "submission", "estimator", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--rpc")
    args = parser.parse_args()
    result = summarize(
        read(args.local), read(args.online), read(args.runners), read(args.submission)
    )
    current = hashlib.sha256(Path(args.estimator).read_bytes()).hexdigest()
    result["submitted_source_matches_current"] = current == result["submitted_estimator_sha256"]
    if args.rpc:
        rpc = read(args.rpc)
        if rpc["manifest"]["estimator_sha256"] != result["submitted_estimator_sha256"]:
            raise ValueError("RPC estimator differs from submitted source")
        local_names = [r["mlp_name"] for r in read(args.local)["results"]["per_mlp"]]
        if [r["mlp_name"] for r in rpc["per_mlp"]] != local_names:
            raise ValueError("RPC networks differ from local report")
        result["rpc"] = {k: v for k, v in rpc.items() if k not in ["manifest", "per_mlp"]}
        result["rpc_flops_delta_per_network"] = [r["local_flops_delta"] for r in rpc["per_mlp"]]
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Score every strict RPC row, retaining failures as undiscounted zero fallbacks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
from whestbench import load_dataset
from whestbench.dataset import resolve_seed_context
from whestbench.domain import MLP

BUDGET = 2**41
LIMITS = {
    "flop_budget": BUDGET,
    "wall_time_limit_s": 120.0,
    "residual_wall_time_limit_s": 0.4,
    "setup_timeout_s": 5.0,
    "lambda_flops_per_second": 0.0,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_rows(items, expected: int):
    """Identity mistakes are protocol errors, never silently missing failures."""
    if not isinstance(items, list) or len(items) != expected:
        raise ValueError("Missing or extra per-network rows")
    rows = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Per-network row must be an object")
        index = item.get("mlp_index", item.get("index"))
        if type(index) is not int or not 0 <= index < expected or index in rows:
            raise ValueError(f"Duplicate or invalid network index: {index}")
        if "mlp_index" in item and "index" in item and item["mlp_index"] != item["index"]:
            raise ValueError("Conflicting network index aliases")
        name = item.get("mlp_name", item.get("name"))
        if "mlp_name" in item and "name" in item and item["mlp_name"] != item["name"]:
            raise ValueError("Conflicting network name aliases")
        rows[index] = (name, item)
    return rows


def numeric(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        return None
    return int(value) if integer else float(value)


def json_safe(value):
    """Keep malformed evidence visible without emitting nonstandard JSON NaNs."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_value": str(value)}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def local_failure(item):
    flags = (
        "budget_exhausted", "time_exhausted", "residual_wall_time_exhausted",
        "combined_budget_exhausted",
    )
    return bool(item.get("traceback") or item.get("failure") or item.get("error")
                or any(item.get(key) for key in flags))


def validate_manifest_limits(manifest):
    recorded = manifest.get("limits")
    if not isinstance(recorded, dict):
        raise ValueError("Strict RPC manifest must declare its limits")
    aliases = {
        "wall_time_limit_s": "predict_seconds",
        "residual_wall_time_limit_s": "residual_seconds",
        "setup_timeout_s": "setup_seconds",
    }
    for key, expected in LIMITS.items():
        # lambda is zero and no residual price exists in the strict supervisor.
        if key == "lambda_flops_per_second" and key not in recorded:
            continue
        alias = aliases.get(key)
        value = recorded.get(key, recorded.get(alias))
        if value != expected:
            raise ValueError(f"Strict RPC manifest limit mismatch: {key}")
        if key in recorded and alias in recorded and recorded[key] != recorded[alias]:
            raise ValueError(f"Conflicting strict RPC limit aliases: {key}")


def local_comparison(path, rpc_manifest, dataset_commit, expected, targets, seeds):
    """A local reference without a bound report hash is not candidate parity."""
    if path is None:
        return None, {"provided": False, "same_candidate_parity_verified": False}
    path = Path(path)
    local = json.loads(path.read_text())
    rows = canonical_rows(local.get("results", {}).get("per_mlp"), expected)
    for index, (name, item) in rows.items():
        if name != targets[index]["mlp_name"]:
            raise ValueError(f"Local network identity mismatch at index {index}")
        if "seed" in item and item["seed"] != seeds[index]:
            raise ValueError(f"Local network seed mismatch at index {index}")
    metadata = {
        "provided": True, "path": str(path.resolve()), "report_sha256": digest(path),
        "same_candidate_parity_verified": False, "verification_errors": [],
    }
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        metadata["verification_errors"].append("Missing local manifest")
        return None, metadata
    manifest = json.loads(manifest_path.read_text())
    metadata["manifest_path"] = str(manifest_path.resolve())
    metadata["manifest_sha256"] = digest(manifest_path)
    checks = {
        "local report hash is bound": manifest.get("report_sha256") == metadata["report_sha256"],
        "same estimator hash": bool(rpc_manifest.get("estimator_sha256")) and
        manifest.get("estimator_sha256") == rpc_manifest.get("estimator_sha256"),
        "same data commit": bool(dataset_commit) and manifest.get("dataset_commit") == dataset_commit,
        "successful local process": manifest.get("exit_code") == 0,
    }
    config = local.get("run_config", {})
    for key, value in LIMITS.items():
        checks[f"same {key}"] = config.get(key) == value
    checks["same setup seed"] = config.get("seed") == rpc_manifest.get("seed", 0)
    checks["same network count"] = config.get("n_mlps") == expected
    metadata["verification_errors"] = [key for key, passed in checks.items() if not passed]
    metadata["same_candidate_parity_verified"] = all(checks.values())
    return (rows if all(checks.values()) else None), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", default="benchmark/data/mini_first8")
    parser.add_argument("--local")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Use a fresh output filename")
    prediction_path = Path(args.predictions)
    rpc = json.loads(prediction_path.read_text())
    manifest = rpc.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Missing strict RPC manifest")
    validate_manifest_limits(manifest)
    expected = manifest.get("n_mlps")
    if type(expected) is not int or expected <= 0:
        raise ValueError("Manifest must declare a positive n_mlps")
    data = load_dataset(args.dataset, split="mini")
    if expected > len(data):
        raise ValueError("Manifest network count exceeds dataset")
    targets = data.select_columns(["mlp_name", "mlp_seed", "all_layer_means", "final_means"])
    protocol, salt = resolve_seed_context(data)
    # Apply the canonical seed conversion to tiny synthetic weights, avoiding
    # loading the dataset's large weight column merely to verify its identity.
    seeds = [MLP.from_row({"weights": [[[0.0]]], "mlp_seed": targets[index]["mlp_seed"]},
                          seed_protocol_version=protocol, seed_salt=salt).seed
             for index in range(expected)]
    provenance_path = Path(args.dataset) / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    dataset_commit = provenance.get("revision") if provenance else None
    if manifest.get("dataset_commit") != dataset_commit or not dataset_commit:
        raise ValueError("Strict RPC manifest does not match frozen dataset commit")
    # CLI dataset paths can be relative or absolute, but must resolve to the same cache.
    if manifest.get("dataset") and Path(manifest["dataset"]).resolve() != Path(args.dataset).resolve():
        raise ValueError("Strict RPC dataset path mismatch")
    inputs = canonical_rows(rpc.get("per_mlp"), expected)
    for index, (name, item) in inputs.items():
        if name != targets[index]["mlp_name"] or item.get("seed") != seeds[index]:
            raise ValueError(f"Strict RPC network name/seed mismatch at index {index}")
    local_rows, comparison = local_comparison(
        args.local, manifest, dataset_commit, expected, targets, seeds,
    )
    rows = []
    for index in range(expected):
        name, item = inputs[index]
        target = targets[index]
        truth = np.asarray(target["all_layer_means"], dtype=np.float32)
        final_truth = np.asarray(target["final_means"], dtype=np.float32)
        if truth.shape != (16, 1024) or final_truth.shape != (1024,):
            raise ValueError("Frozen target shape differs from Phase 2 contract")
        if not np.isfinite(truth).all() or not np.isfinite(final_truth).all():
            raise ValueError("Frozen reference contains non-finite values")
        reasons = []
        if manifest.get("setup_ok") is not True:
            reasons.append("setup_not_successful")
        if item.get("status") != "ok":
            reasons.append("worker_status_not_ok")
        if item.get("failure"):
            reasons.append("worker_failure_recorded")
        summary = item.get("budget_summary")
        summary = summary if isinstance(summary, dict) else {}
        flops = numeric(summary.get("flops_used"), integer=True)
        wall = numeric(summary.get("wall_time_s"))
        residual = numeric(summary.get("residual_wall_time_s"))
        if flops is None or wall is None or residual is None:
            reasons.append("missing_or_invalid_budget_summary")
        else:
            if flops > BUDGET:
                reasons.append("budget_exhausted")
            if wall > 120:
                reasons.append("time_exhausted")
            if residual > 0.4:
                reasons.append("residual_wall_time_exhausted")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                prediction = np.asarray(item.get("predictions"), dtype=np.float32)
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                reasons.append("invalid_prediction_shape_or_values")
        except (ValueError, TypeError, OverflowError):
            prediction = None
            reasons.append("invalid_prediction_shape_or_values")
        valid = not reasons
        scored_prediction = prediction if valid else np.zeros_like(truth)
        with np.errstate(over="ignore", invalid="ignore"):
            mse = float(np.mean((scored_prediction[-1] - final_truth) ** 2))
            all_mse = float(np.mean((scored_prediction - truth) ** 2))
        if valid and (not math.isfinite(mse) or not math.isfinite(all_mse)):
            reasons.append("non_finite_float32_loss")
            valid = False
            scored_prediction = np.zeros_like(truth)
            mse = float(np.mean(final_truth ** 2))
            all_mse = float(np.mean(truth ** 2))
        multiplier = max(0.1, flops / BUDGET) if valid else 1.0
        row = {
            "mlp_index": index, "mlp_name": name, "seed": item["seed"],
            "worker_status": item.get("status"), "failure": item.get("failure"),
            "failure_reasons": reasons, "zero_fallback": not valid, "limits_passed": valid,
            "final_layer_mse": mse,
            "all_layers_mse": all_mse,
            "score_multiplier": multiplier, "adjusted_final_layer_score": mse * multiplier,
            "flops_used": flops, "wall_time_s": wall, "residual_wall_time_s": residual,
            "budget_summary": item.get("budget_summary"),
            "client_peak_rss_kib": item.get("client_peak_rss_kib"),
        }
        if local_rows is not None:
            control = local_rows[index][1]
            control_failed = local_failure(control)
            row["local_comparison_status"] = (
                "both_passed" if valid and not control_failed else
                "rpc_passed_local_failed" if valid else
                "both_failed" if control_failed else "rpc_failed_local_passed"
            )
            if valid and not control_failed:
                before = numeric(control.get("final_layer_mse"))
                cost = numeric(control.get("flops_used"), integer=True)
                row["local_mse_relative_delta"] = mse / before - 1 if before else None
                row["local_flops_delta"] = flops - cost if cost is not None else None
        rows.append(row)
    known_costs = [row["flops_used"] for row in rows if row["flops_used"] is not None]
    deltas = [abs(row["local_mse_relative_delta"]) for row in rows
              if row.get("local_mse_relative_delta") is not None]
    result = {
        "schema_version": "1.0", "mode": "strict local official-client RPC score",
        "manifest": manifest, "predictions_sha256": digest(prediction_path),
        "score_script_sha256": digest(Path(__file__)),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "flopscope", "whestbench")},
        "dataset": str(Path(args.dataset).resolve()), "dataset_commit": dataset_commit,
        "seed_protocol_version": protocol,
        "dataset_provenance": provenance, "limits": LIMITS,
        "n_mlps": expected, "n_failed_mlps": sum(not row["limits_passed"] for row in rows),
        "final_layer_mse": float(np.mean([row["final_layer_mse"] for row in rows])),
        "all_layers_mse": float(np.mean([row["all_layers_mse"] for row in rows])),
        "adjusted_final_layer_score": float(np.mean([row["adjusted_final_layer_score"] for row in rows])),
        "mean_compute_utilization": float(np.mean(known_costs)) / BUDGET
        if len(known_costs) == expected else None,
        "n_unknown_flop_counts": expected - len(known_costs),
        "local_comparison": comparison,
        "max_relative_mse_delta_vs_local": max(deltas) if deltas else None,
        "per_mlp": rows,
        "limitation": "Local development server, not an online grade or full hardened-grader equivalence."
        " Missing costs remain unknown; every failed row uses zero prediction and multiplier 1.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("n_mlps", "n_failed_mlps", "final_layer_mse", "adjusted_final_layer_score",
                       "n_unknown_flop_counts", "local_comparison")}, indent=2))


if __name__ == "__main__":
    main()

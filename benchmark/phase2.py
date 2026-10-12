"""Reproducible Phase 2 runs and comparisons; never submits automatically."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "hf://aicrowd/arc-whestbench-public-2026@v2-phase2"
BUDGET = 2**41


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(left, right):
    """Require the same networks and scoring configuration before reporting deltas."""
    keys = (
        "flop_budget",
        "lambda_flops_per_second",
        "wall_time_limit_s",
        "residual_wall_time_limit_s",
        "seed",
        "dataset",
    )
    for key in keys:
        if left.get("run_config", {}).get(key) != right.get("run_config", {}).get(key):
            raise ValueError(f"Run configuration differs: {key}")
    for key in ("whestbench_version", "flopscope_version"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Tool version differs: {key}")
    a, b = left["results"]["per_mlp"], right["results"]["per_mlp"]
    if not a or len(a) != len(b):
        raise ValueError("Network counts differ or are empty")
    rows = []
    if any(not row.get("mlp_name") for row in a + b):
        raise ValueError("Missing network identity")
    for x, y in zip(a, b):
        for key in ("mlp_name", "seed"):
            if x.get(key) != y.get(key):
                raise ValueError(f"Network order differs: {key}")
        row = {"mlp_name": x.get("mlp_name"), "seed": x.get("seed")}
        for key in ("final_layer_mse", "adjusted_final_layer_score", "flops_used"):
            if key not in x or key not in y:
                raise ValueError(f"Missing per-network metric: {key}")
            u, v = float(x[key]), float(y[key])
            if not math.isfinite(u) or not math.isfinite(v):
                raise ValueError(f"Non-finite metric: {key}")
            row[key] = {
                "left": u,
                "right": v,
                "delta": v - u,
                "relative_delta": (v - u) / u if u else None,
            }
        rows.append(row)
    return {
        "n_mlps": len(rows),
        "left_failed": left["results"].get("n_failed_mlps"),
        "right_failed": right["results"].get("n_failed_mlps"),
        "per_mlp": rows,
        "same_flops": all(row["flops_used"]["delta"] == 0 for row in rows),
        "max_relative_mse_delta": max(
            abs(row["final_layer_mse"]["relative_delta"] or 0) for row in rows
        ),
        "interpretation": "Paired same-network comparison; not an online/private-suite parity claim.",
    }


def run(args):
    estimator = Path(args.estimator).resolve()
    output = Path(args.output).resolve()
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise ValueError("Use a fresh output filename; benchmark records are immutable")
    command = [
        "uv",
        "run",
        "--frozen",
        "whest",
        "run",
        "--estimator",
        str(estimator),
        "--dataset",
        args.dataset,
        "--split",
        "mini",
        "--n-mlps",
        str(args.n_mlps),
        "--runner",
        args.runner,
        "--max-threads",
        "1",
        "--seed",
        "0",
        "--flop-budget",
        str(BUDGET),
        "--lambda-flops-per-second",
        "0",
        "--wall-time-limit",
        "120",
        "--setup-timeout",
        "5",
        "--residual-wall-time-limit",
        "0.4",
        "--detail",
        "full",
        "--format",
        "json",
    ]
    if args.streaming:
        command.append("--streaming")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(tuple(f"V{i}_" for i in range(17, 30))):
            del env[key]
    env.update(
        {
            "HF_HUB_DISABLE_XET": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": output.stem,
        "command": command,
        "estimator_sha256": digest(estimator),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("flopscope", "whestbench")},
        "dataset": args.dataset,
        "seed": 0,
        "max_threads": 1,
    }
    local_provenance = Path(args.dataset) / "provenance.json"
    if local_provenance.exists():
        manifest["dataset_provenance"] = read(local_provenance)
        manifest["dataset_commit"] = manifest["dataset_provenance"]["revision"]
    else:
        try:
            from huggingface_hub import HfApi

            manifest["dataset_commit"] = HfApi().repo_info(
                "aicrowd/arc-whestbench-public-2026", repo_type="dataset", revision="v2-phase2"
            ).sha
        except Exception as exc:
            raise RuntimeError("Cannot resolve dataset revision; run not started") from exc
    save(output.with_suffix(".manifest.json"), manifest)
    with output.open("w") as out, output.with_suffix(".log").open("w") as err:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    manifest.update(
        {
            "exit_code": result.returncode,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "report_sha256": digest(output),
            "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        }
    )
    save(output.with_suffix(".manifest.json"), manifest)
    if result.returncode:
        raise RuntimeError(f"Benchmark failed; inspect {output} and its .log")
    report = read(output)
    metrics = report.get("results", {})
    print(
        json.dumps(
            {
                k: metrics.get(k)
                for k in (
                    "adjusted_final_layer_score",
                    "final_layer_mse",
                    "mean_compute_utilization",
                    "n_failed_mlps",
                )
            },
            ensure_ascii=False,
        )
    )


def status(args):
    from whestbench.aicrowd_client import AIcrowdClient
    from whestbench.aicrowd_config import resolve_api_key

    payload = AIcrowdClient(api_key=resolve_api_key(None)).get_submission_status(args.submission_id)
    save(
        args.output,
        {
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "submission_id": args.submission_id,
            "response": payload,
        },
    )
    print(
        json.dumps(
            {
                k: payload.get(k)
                for k in ("grading_status_cd", "score", "score_secondary", "grading_message")
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--estimator", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--n-mlps", type=int, default=8)
    r.add_argument("--runner", choices=["local", "subprocess"], default="subprocess")
    r.add_argument("--streaming", action="store_true")
    r.add_argument("--dataset", default=DATASET)
    c = sub.add_parser("compare")
    c.add_argument("left")
    c.add_argument("right")
    c.add_argument("--output", required=True)
    s = sub.add_parser("status")
    s.add_argument("submission_id", type=int)
    s.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "run":
        if args.n_mlps <= 0:
            parser.error("--n-mlps must be positive")
        run(args)
    elif args.action == "compare":
        a, b = read(args.left), read(args.right)
        manifests = []
        for name in (args.left, args.right):
            sidecar = Path(name).with_suffix(".manifest.json")
            if not sidecar.exists():
                raise ValueError(f"Missing manifest: {sidecar}")
            meta = read(sidecar)
            if meta["report_sha256"] != digest(name):
                raise ValueError(f"Report SHA-256 differs: {name}")
            manifests.append(meta)
        for key in ("versions", "dataset_commit", "dataset_provenance"):
            if manifests[0].get(key) != manifests[1].get(key):
                raise ValueError(f"Manifest configuration differs: {key}")
        result = compare(a, b)
        result["source_manifests"] = manifests
        save(args.output, result)
        print(f"Compared {result['n_mlps']} networks")
    else:
        status(args)


if __name__ == "__main__":
    main()

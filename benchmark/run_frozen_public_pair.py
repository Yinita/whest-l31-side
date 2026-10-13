"""Execute a frozen public cohort serially, keeping resumable phase evidence.

No parameter selection, dataset download, online upload, or research-state edits
are performed. Only completed outputs can be resumed; partial outputs block
resumption so a live/incomplete experiment cannot be overwritten.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("prepare_rpc_inputs.py", "run_rpc_strict.py", "rpc_strict_score.py",
         "compare_strict_cohort.py", "aggregate_strict_public.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    protocol_path = ROOT/"experiments/phase2/DEVELOPMENT_PROTOCOL_20261013.json"
    protocol = json.loads(protocol_path.read_text())
    cohort = protocol[args.cohort]
    if not cohort["no_model_selection_on_cohort"]:
        raise ValueError("Candidate selection must be frozen")
    if not args.run_id.isalnum():
        raise ValueError("Simple alphanumeric run id required")
    dataset = Path(cohort["dataset"])
    provenance = json.loads((dataset/"provenance.json").read_text())
    if provenance["revision"] != cohort["dataset_commit"]:
        raise ValueError("Data commit mismatch")
    if provenance["sha256"] != cohort["arrow_sha256"]:
        raise ValueError("Frozen Arrow hash mismatch")
    n = provenance["n_mlps"]
    if n != cohort["n_mlps"] or provenance["network_ids"] != cohort["network_ids"]:
        raise ValueError("Cohort identities differ")
    before, after = cohort["candidates"]
    for candidate in (before, after):
        if digest(Path(candidate["path"])) != candidate["sha256"]:
            raise ValueError("Frozen candidate hash mismatch")
    inputs = Path(cohort["input_manifest"]).parent/"inputs.json"
    prefix = ROOT/"results/phase2"/f"frozen_pair_{args.cohort}_{args.run_id}"
    workflow = prefix.with_suffix(".json")
    # Cohort-wide lock also blocks a second run-id racing the same outputs.
    lock = prefix.parent/f"frozen_pair_{args.cohort}.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, f"{os.getpid()} {utc()}\n".encode())
    os.close(fd)
    tool_hashes = {name: digest(ROOT/"benchmark"/name) for name in TOOLS}
    report = {"status": "running", "pid": os.getpid(), "started_at_utc": utc(),
              "script_sha256": digest(Path(__file__)), "tool_hashes": tool_hashes,
              "protocol_sha256": digest(protocol_path), "cohort": cohort,
              "n_mlps": n, "phases": [], "current_phase": None}

    def save():
        temporary = workflow.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(workflow)

    def verify_frozen():
        for candidate in (before, after):
            if digest(Path(candidate["path"])) != candidate["sha256"]:
                raise ValueError("Candidate source changed mid-workflow")
        for name, sha in tool_hashes.items():
            if digest(ROOT/"benchmark"/name) != sha:
                raise ValueError("Benchmark tool changed mid-workflow")

    def execute(name, command, output, complete):
        verify_frozen()
        if output.exists():
            if not complete(json.loads(output.read_text())):
                raise ValueError(f"Partial output blocks resumption: {output}")
            report["phases"].append({"name": name, "status": "resumed_completed",
                                     "output": str(output), "sha256": digest(output)})
            save()
            return
        report["current_phase"] = name
        save()
        log = prefix.parent/f"{prefix.name}_{name}.log"
        phase = {"name": name, "command": command, "started_at_utc": utc(), "log": str(log)}
        with log.open("w") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            phase["pid"] = child.pid
            report["current_child"] = phase
            save()
            returncode = child.wait()
        phase.update(returncode=returncode, finished_at_utc=utc())
        if returncode or not output.exists() or not complete(json.loads(output.read_text())):
            phase["status"] = "failed_or_incomplete"
            report["phases"].append(phase)
            raise RuntimeError(f"Phase failed: {name}; preserved {log}")
        phase.update(status="completed", output=str(output), sha256=digest(output))
        report["phases"].append(phase)
        report.pop("current_child", None)
        save()
        print(name, "completed", flush=True)

    try:
        save()
        execute("inputs", ["uv", "run", "--frozen", "python", "benchmark/prepare_rpc_inputs.py",
                           "--dataset", str(dataset), "--n-mlps", str(n), "--output-dir", str(inputs.parent)],
                inputs.parent/"manifest.json", lambda r: r.get("complete") is True and r.get("n_mlps") == n)
        manifest = json.loads((inputs.parent/"manifest.json").read_text())
        if digest(inputs) != manifest["inputs_sha256"] or manifest["dataset_commit"] != cohort["dataset_commit"]:
            raise ValueError("Input manifest changed")
        scored_paths = []
        for candidate, score_key in ((before, "before_report"), (after, "after_report")):
            score = Path(cohort[score_key])
            prediction = score.with_name(score.name.replace("_score.json", ".json"))
            execute(f"{candidate['name']}_predict", ["uv", "run", "--frozen", "python", "benchmark/run_rpc_strict.py",
                    "--estimator", candidate["path"], "--inputs", str(inputs), "--dataset", str(dataset),
                    "--n-mlps", str(n), "--output", str(prediction)], prediction,
                    lambda r: r.get("manifest", {}).get("completed") is True and len(r.get("per_mlp", [])) == n)
            execute(f"{candidate['name']}_score", ["uv", "run", "--frozen", "python", "benchmark/rpc_strict_score.py",
                    "--predictions", str(prediction), "--dataset", str(dataset), "--output", str(score)], score,
                    lambda r: r.get("manifest", {}).get("completed") is True and r.get("n_mlps") == n)
            scored_paths.append(score)
        comparison = prefix.parent/f"{prefix.name}_before_after.json"
        execute("paired_compare", ["uv", "run", "--frozen", "python", "benchmark/compare_strict_cohort.py",
                "--before", str(scored_paths[0]), "--after", str(scored_paths[1]), "--output", str(comparison)],
                comparison, lambda r: r.get("protocol_checks_passed") is True)
        report.update(status="completed", finished_at_utc=utc(), comparison=str(comparison), current_phase=None)
        save()
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=utc())
        save()
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

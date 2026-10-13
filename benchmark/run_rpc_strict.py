"""Supervise official RPC packages without modifying their compute or meters."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import whestbench

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {"flop_budget": 2**41, "predict_seconds": 120, "residual_seconds": 0.4,
          "setup_seconds": 5, "response_grace_seconds": 5}


class SupervisionError(RuntimeError):
    def __init__(self, phase, reason, details=None):
        super().__init__(reason)
        self.phase, self.details = phase, details


class Session:
    def __init__(self, estimator, output, env, ordinal, expected_hashes):
        self.tmp = tempfile.TemporaryDirectory(prefix="whest-strict-rpc-")
        self.server = self.worker = None
        self.buffer = b""
        self.selector = selectors.DefaultSelector()
        self.logs = []
        url = f"ipc://{self.tmp.name}/server.sock"
        self.ordinal = ordinal
        lifecycle = output.with_suffix(f".session{ordinal}.lifecycle.json")
        try:
            server_log = output.with_suffix(f".session{ordinal}.server.log").open("wb")
            worker_log = output.with_suffix(f".session{ordinal}.worker.log").open("wb")
            self.logs.extend([server_log, worker_log])
            self.server = subprocess.Popen(
                [sys.executable, "-m", "flopscope_server", "--url", url, "--timeout", "180"],
                env=env, stdout=server_log, stderr=server_log, start_new_session=True,
            )
            deadline = time.monotonic() + 10
            while not Path(self.tmp.name, "server.sock").exists():
                if self.server.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("RPC server startup failed")
                time.sleep(0.05)
            worker_env = dict(env, FLOPSCOPE_SERVER_URL=url)
            self.worker = subprocess.Popen(
                [str(ROOT / "benchmark/data/client-venv/bin/python"), "-u",
                 str(ROOT / "benchmark/rpc_strict_worker.py"), "--sdk-root",
                 str(Path(whestbench.__file__).parent), "--estimator", str(estimator)],
                env=worker_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=worker_log, start_new_session=True,
            )
            lifecycle.write_text(json.dumps({"server_pid": self.server.pid,
                                           "client_pid": self.worker.pid,
                                           "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
            os.set_blocking(self.worker.stdout.fileno(), False)
            self.selector.register(self.worker.stdout, selectors.EVENT_READ)
            self.ready = self.receive("ready", 30)
            if self.ready["manifest"]["worker_sha256"] != expected_hashes["worker_sha256"]:
                raise SupervisionError("source_mismatch", "Executed worker hash changed")
            self.send({"command": "start_setup"})
            try:
                self.setup = self.receive("setup_done", LIMITS["setup_seconds"])
            except TimeoutError as exc:
                raise SupervisionError("setup_timeout", str(exc)) from exc
            if self.setup.get("status") != "ok":
                raise SupervisionError("setup_error", "Candidate setup failed", self.setup)
            if self.setup["setup_elapsed_s"] > LIMITS["setup_seconds"]:
                raise SupervisionError("setup_timeout", "Setup exceeded 5s", self.setup)
            if self.setup["estimator_sha256"] != expected_hashes["estimator_sha256"]:
                raise SupervisionError("source_mismatch", "Executed candidate hash changed")
        except BaseException:
            self.close()
            raise

    def send(self, value):
        self.worker.stdin.write((json.dumps(value) + "\n").encode())
        self.worker.stdin.flush()

    def receive(self, expected, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                event = json.loads(line)
                if event.get("event") in {"setup_error", "worker_error"}:
                    raise SupervisionError(event["event"], "Worker failed", event)
                if expected == "predict_ready" and event.get("event") == "result":
                    raise SupervisionError("input_error", "Input construction failed", event)
                if event.get("event") != expected:
                    raise RuntimeError(f"Expected {expected}, received {event}")
                return event
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{expected} response exceeded {timeout}s")
            if not self.selector.select(remaining):
                raise TimeoutError(f"{expected} response exceeded {timeout}s")
            chunk = os.read(self.worker.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"Worker EOF (exit {self.worker.poll()}) while awaiting {expected}")
            self.buffer += chunk

    def predict(self, entry):
        self.send({"command": "predict", "entry": entry})
        ready = self.receive("predict_ready", 120)
        for key, source in [("mlp_index", "index"), ("mlp_name", "name"), ("seed", "seed")]:
            if ready[key] != entry[source]:
                raise SupervisionError("identity_mismatch", "Worker input identity mismatch", ready)
        self.send({"command": "go"})
        try:
            result = self.receive("result", LIMITS["predict_seconds"] + LIMITS["response_grace_seconds"])
            for key, source in [("mlp_index", "index"), ("mlp_name", "name"), ("seed", "seed")]:
                if result[key] != entry[source]:
                    raise SupervisionError("identity_mismatch", "Worker result identity mismatch", result)
            return result
        except TimeoutError as exc:
            raise SupervisionError("time_exhausted", str(exc)) from exc

    def close(self):
        # A dead client cannot cancel synchronous server work. End BOTH processes
        # and their connection-lifetime handles before cold recovery.
        errors = []
        for process in [self.worker, self.server]:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                except ProcessLookupError:
                    process.wait(timeout=10)
                except Exception as exc:
                    errors.append(f"cleanup {process.pid}: {exc}")
        self.selector.close()
        for log in self.logs:
            log.close()
        self.tmp.cleanup()
        return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--inputs", required=True, help="Existing immutable RPC inputs.json")
    parser.add_argument("--dataset", default="benchmark/data/mini_first8")
    parser.add_argument("--n-mlps", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, estimator = Path(args.output).resolve(), Path(args.estimator).resolve()
    if output.exists() or list(output.parent.glob(output.stem + ".session*")) or args.n_mlps <= 0:
        parser.error("Positive n-mlps and a fresh output are required")
    inputs = Path(args.inputs).resolve()
    entries = json.loads(inputs.read_text())[:args.n_mlps]
    if len(entries) != args.n_mlps:
        parser.error("Insufficient input entries")
    for entry in entries:
        if hashlib.sha256(Path(entry["weights"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Input weights hash mismatch")
    provenance = json.loads(Path(args.dataset, "provenance.json").read_text())
    manifest = {"mode": "supervised local development RPC; official deployment unknown",
                "n_mlps": args.n_mlps, "dataset": args.dataset,
                "dataset_commit": provenance["revision"], "limits": LIMITS,
                "estimator_sha256": hashlib.sha256(estimator.read_bytes()).hexdigest(),
                "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
                "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "worker_sha256": hashlib.sha256((ROOT / "benchmark/rpc_strict_worker.py").read_bytes()).hexdigest(),
                "versions": {k: importlib.metadata.version(k) for k in ["flopscope", "whestbench"]},
                "command": sys.argv, "setup_ok": True,
                "residual_policy": "Official-style post-predict gate; not interrupted in-flight",
                "server_max_array_bytes": 1073741824,
                "continuous_lifecycle": True, "completed": False,
                "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "network_ids": [{k: e[k] for k in ["index", "name", "seed"]} for e in entries]}
    env = os.environ.copy()
    env.update(FLOPSCOPE_MAX_ARRAY_BYTES="1073741824", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for key in list(env):
        if key.startswith(tuple(f"V{i}_" for i in range(17, 30))):
            del env[key]
    rows, sessions, session, attempts = [], [], None, 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps({"manifest": manifest, "sessions": sessions, "per_mlp": rows}) + "\n")
    try:
        for entry in entries:
            row = {"mlp_index": entry["index"], "mlp_name": entry["name"],
                   "seed": entry["seed"], "status": "error", "predictions": None,
                   "budget_summary": None}
            try:
                if session is None:
                    ordinal, attempts = attempts, attempts + 1
                    session = Session(estimator, output, env, ordinal, manifest)
                    sessions.append({"ordinal": ordinal, "setup": session.setup,
                                     "ready_manifest": session.ready["manifest"],
                                     "client_pid": session.worker.pid, "server_pid": session.server.pid})
                row.update(session.predict(entry))
                row.pop("event", None)
                row["session_ordinal"] = session.ordinal
                summary = row.get("budget_summary") or {}
                if row.get("status") == "ok":
                    measured = [summary.get(k) for k in ["flops_used", "wall_time_s", "residual_wall_time_s"]]
                    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in measured):
                        row.update(status="invalid_summary", failure="Missing or invalid budget counters")
                    elif summary["flops_used"] > LIMITS["flop_budget"]:
                        row.update(status="budget_exhausted", failure="FLOPs exceeded budget")
                    elif summary["wall_time_s"] > 120:
                        row.update(status="time_exhausted", failure="predict exceeded 120s")
                    elif summary.get("residual_wall_time_s", float("inf")) > 0.4:
                        row.update(status="residual_wall_time_exhausted", failure="residual exceeded 0.4s")
                if row.get("status") != "ok":
                    row["supervisor_cleanup_errors"] = session.close()
                    session = None
                    manifest["continuous_lifecycle"] = False
            except Exception as exc:
                row.update(status=getattr(exc, "phase", "supervisor_failure"),
                           failure={"type": type(exc).__name__, "message": str(exc),
                                    "details": getattr(exc, "details", None)})
                manifest["continuous_lifecycle"] = False
                if session is not None:
                    row["supervisor_cleanup_errors"] = session.close()
                    session = None
                row["session_ordinal"] = attempts - 1
                if row["status"].startswith("setup"):
                    sessions.append({"ordinal": attempts - 1, "setup_failed": True,
                                     "failure": row["failure"]})
            rows.append(row)
            manifest["setup_ok"] = any(not s.get("setup_failed") for s in sessions)
            manifest["all_setup_attempts_ok"] = not any(s.get("setup_failed") for s in sessions)
            manifest["completed"] = len(rows) == len(entries)
            output.write_text(json.dumps({"manifest": manifest, "sessions": sessions, "per_mlp": rows}) + "\n")
            print(f"Strict RPC network {entry['index']}: {row['status']}", flush=True)
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()

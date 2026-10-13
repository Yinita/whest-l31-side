"""Trusted JSONL RPC worker; its parent enforces setup/predict response deadlines.

Normal protocol: ready -> start_setup -> setup_done -> predict -> predict_ready
-> go -> result, repeated predict/go on the same estimator and connection.
Only stdout carries protocol events. Participant/library output goes to stderr.
The RPC client's real BudgetContext enforces FLOPs and measures timing; this
worker neither changes metering nor claims RPC has native time-limit parameters.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import resource
import sys
import time
import traceback
import types
from pathlib import Path

BUDGET = 2**41
WIDTH = 1024
DEPTH = 16
PROTOCOL_STDOUT = sys.stdout


def emit(event, **fields):
    PROTOCOL_STDOUT.write(json.dumps({"event": event, **fields}, allow_nan=False) + "\n")
    PROTOCOL_STDOUT.flush()


def receive():
    line = sys.stdin.readline()
    if not line:
        return None
    request = json.loads(line)
    if not isinstance(request, dict):
        raise ValueError("Protocol request must be a JSON object")
    return request


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def error_record(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def run_budget(flops, action):
    """Preserve the primary error if closing or obtaining a summary also fails."""
    ctx = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
    value = summary = primary = cleanup = summary_error = None
    entered = False
    try:
        ctx.__enter__()
        entered = True
        value = action()
    except Exception as exc:
        primary = error_record(exc)
    finally:
        if entered:
            try:
                ctx.__exit__(None, None, None)
            except Exception as exc:
                cleanup = error_record(exc)
        # After a failed close, live summary retrieval could hang or replace the
        # original failure. The parent must discard both client and server then.
        if cleanup is None:
            try:
                summary = ctx.summary_dict()
            except Exception as exc:
                summary_error = error_record(exc)
    return value, summary, primary, cleanup, summary_error


def validate_values(values):
    if not isinstance(values, list) or len(values) != DEPTH:
        raise ValueError("Invalid prediction depth")
    for row in values:
        if not isinstance(row, list) or len(row) != WIDTH:
            raise ValueError("Invalid prediction width")
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in row):
            raise ValueError("Non-finite or non-numeric prediction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--estimator", required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    import flopscope as flops
    import flopscope.numpy as fnp

    sdk_root = Path(args.sdk_root).resolve()
    candidate = Path(args.estimator).resolve()
    package = types.ModuleType("whestbench")
    package.__path__ = [str(sdk_root)]
    sys.modules["whestbench"] = package
    domain = load_module("whestbench.domain", sdk_root / "domain.py")
    sdk = load_module("whestbench.sdk", sdk_root / "sdk.py")
    package.MLP = domain.MLP
    package.BaseEstimator = sdk.BaseEstimator
    package.SetupContext = sdk.SetupContext
    manifest = {
        "flopscope_client_version": importlib.metadata.version("flopscope-client"),
        "sdk_sha256": hashlib.sha256((sdk_root / "sdk.py").read_bytes()).hexdigest(),
        "domain_sha256": hashlib.sha256((sdk_root / "domain.py").read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "flop_budget": BUDGET,
        "memory_limit_bytes": 8 * 1024**3,
        "width": WIDTH,
        "depth": DEPTH,
        "seed": 0,
        "mode": "trusted locally supervised RPC; not the official hardened grader",
    }
    emit("ready", manifest=manifest)
    request = receive()
    if request is None:
        return
    if request.get("command") != "start_setup":
        raise ValueError("Expected start_setup after ready")
    setup_start = time.perf_counter()
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()

    def setup_action():
        module = load_module("strict_rpc_estimator", candidate)
        instance = module.Estimator()
        instance.setup(
            sdk.SetupContext(
                width=WIDTH, depth=DEPTH, flop_budget=BUDGET, api_version="0.1", seed=0,
                submission_dir=str(candidate.parent),
            )
        )
        return instance

    estimator, setup_summary, failure, cleanup_error, summary_error = run_budget(flops, setup_action)
    setup_elapsed = time.perf_counter() - setup_start
    setup_fields = {
        "status": "error" if failure or cleanup_error or summary_error else "ok",
        "setup_elapsed_s": setup_elapsed,
        "estimator_sha256": candidate_hash,
        "budget_summary": setup_summary,
        "failure": failure,
        "cleanup_error": cleanup_error,
        "summary_error": summary_error,
        "client_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if setup_fields["status"] != "ok":
        emit("setup_error", **setup_fields)
        return
    emit("setup_done", **setup_fields)
    while True:
        request = receive()
        if request is None or request.get("command") == "close":
            return
        if request.get("command") != "predict":
            raise ValueError("Expected predict or close")
        entry = request["entry"]
        identity = {
            "mlp_index": entry["index"],
            "mlp_name": entry["name"],
            "seed": entry["seed"],
        }

        def input_action():
            weights = fnp.load(str(entry.get("weights", entry.get("path"))))["weights"]
            return domain.MLP(
                width=WIDTH, depth=DEPTH, weights=[weights[j] for j in range(DEPTH)],
                seed=entry["seed"], name=entry["name"],
            )

        mlp, input_summary, failure, cleanup_error, summary_error = run_budget(flops, input_action)
        if failure or cleanup_error or summary_error:
            emit(
                "result", **identity, status="error", phase="input", predictions=None,
                budget_summary=None, input_budget_summary=input_summary, failure=failure,
                cleanup_error=cleanup_error, summary_error=summary_error, reusable=False,
            )
            return
        emit("predict_ready", **identity, input_budget_summary=input_summary)
        request = receive()
        if request is None:
            return
        if request.get("command") != "go":
            raise ValueError("Expected go after predict_ready")
        predict_start = time.perf_counter()

        def predict_action():
            predictions = estimator.predict(mlp, BUDGET)
            if tuple(predictions.shape) != (DEPTH, WIDTH):
                raise ValueError("Invalid prediction shape")
            values = predictions.tolist()  # Materialise in the real metering window.
            del predictions
            return values

        values, summary, failure, cleanup_error, summary_error = run_budget(flops, predict_action)
        predict_elapsed = time.perf_counter() - predict_start
        if failure is None and cleanup_error is None and summary_error is None:
            try:
                validate_values(values)
            except Exception as exc:
                failure = error_record(exc)
        failed = bool(failure or cleanup_error or summary_error)
        emit(
            "result", **identity, status="error" if failed else "ok", phase="predict",
            predictions=None if failed else values, budget_summary=summary,
            predict_elapsed_s=predict_elapsed, failure=failure, cleanup_error=cleanup_error,
            summary_error=summary_error, reusable=not failed,
            client_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        if failed:
            return


if __name__ == "__main__":
    with contextlib.redirect_stdout(sys.stderr):
        try:
            main()
        except Exception as exc:
            emit("worker_error", status="error", failure=error_record(exc), reusable=False)
            raise

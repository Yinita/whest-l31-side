"""Target-free canonical candidate states; instrumentation is not resource acceptance."""
# ruff: noqa: E402
import os
import re

for key in tuple(os.environ):
    if re.match(r"V\d+_", key):
        os.environ.pop(key)
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.estimator).resolve(), Path(args.output)
    report = output.with_suffix(".json")
    assert not output.exists() and not report.exists()
    weights_path = Path("benchmark/data/rpc_inputs/mlp_0.npz")
    weights_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    assert weights_sha == "7fb2e3f519b4a90f2c3de9e64d45ad89b7f20ffd254d887e4ba4fa5c1fdf0754"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("candidate_state_capture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        estimator.setup(SetupContext(width=1024, depth=16, flop_budget=2**41,
                                     api_version="0.1", seed=3909487726))
    trigger = next(i for i, line in enumerate(source.read_text().splitlines(), 1)
                   if "if riders and not NO_CORR:" in line)
    names = ("mu", "var", "sigma", "phi", "Phi", "pk1v", "D3", "g4row",
             "pk2v", "pk3v", "pk4v")
    captured = []

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == str(source) and frame.f_lineno == trigger:
            state = frame.f_locals
            values = [np.asarray(state[k]).copy() if state[k] is not None else np.zeros(1024)
                      for k in names]
            inverse = state["inv_perm"]
            if inverse is not None:
                values = [value[np.asarray(inverse)] for value in values]
            captured.append(np.stack(values, axis=1))
        return tracer

    start = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
        mlp = SimpleNamespace(width=1024, weights=[fnp.asarray(w) for w in np.load(weights_path)["weights"]])
        sys.settrace(tracer)
        try:
            predictions = np.asarray(estimator.predict(mlp, 2**41)).copy()
        finally:
            sys.settrace(None)
    assert len(captured) == 16 and predictions.shape == (16, 1024)
    assert np.isfinite(predictions).all()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    np.savez(output, pre_features=np.stack(captured), predictions=predictions)
    result = {"complete": True, "source": str(source), "source_sha256": source_sha,
              "weights_sha256": weights_sha, "output": str(output),
              "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "feature_order": list(names), "canonical_permutation": True,
              "flops": context.summary_dict()["flops_used"],
              "instrumented_elapsed_s": time.monotonic()-start,
              "versions": {k: importlib.metadata.version(k) for k in ("numpy", "flopscope", "whestbench")},
              "command": sys.argv, "pid": os.getpid(), "setup_seed": 3909487726,
              "configuration_env": {key: value for key, value in os.environ.items()
                                    if re.match(r"V\d+_", key)},
              "network_id": 0, "dataset_commit": "aa99830fdc09fad15407b10e8e3459d3e18bba0a",
              "scope": "r02 all V<number> environment cleared. No reference access. This call's own features/predictions must stay paired; trace changes floating order. No resource acceptance or score claim."}
    report.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

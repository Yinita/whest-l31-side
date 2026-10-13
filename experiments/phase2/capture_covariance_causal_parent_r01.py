"""Target-free canonical H18 preactivation trace for offline causal diagnosis."""
# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext


def main():
    source = Path("estimators/sprint_public_v37b_h18_r01/estimator.py").resolve()
    weights_path = Path("benchmark/data/rpc_inputs/mlp_0.npz")
    expected_sha = "7fb2e3f519b4a90f2c3de9e64d45ad89b7f20ffd254d887e4ba4fa5c1fdf0754"
    assert hashlib.sha256(weights_path.read_bytes()).hexdigest() == expected_sha
    output = Path("results/phase2/covariance_causal_parent_trace_r01.npz")
    report = output.with_suffix(".json")
    assert not output.exists() and not report.exists()
    spec = importlib.util.spec_from_file_location("causal_h18_r01", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        estimator.setup(SetupContext(width=1024, depth=16, flop_budget=2**41,
                                     api_version="0.1", seed=3909487726))
    trigger = next(i for i, line in enumerate(source.read_text().splitlines(), 1)
                   if "if riders and not NO_CORR:" in line)
    captured = []

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == str(source) and frame.f_lineno == trigger:
            state = frame.f_locals
            values = [np.asarray(state[k]).copy() for k in ("mu", "var", "sigma", "phi", "Phi", "pk1v")]
            inverse = state["inv_perm"]
            if inverse is not None:
                inverse = np.asarray(inverse)
                values = [x[inverse] for x in values]
            captured.append(np.stack(values, axis=1))
        return tracer

    with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
        mlp = SimpleNamespace(width=1024, weights=[fnp.asarray(w) for w in np.load(weights_path)["weights"]])
        sys.settrace(tracer)
        try:
            predictions = np.asarray(estimator.predict(mlp, 2**41)).copy()
        finally:
            sys.settrace(None)
    assert len(captured) == 16 and predictions.shape == (16, 1024)
    np.savez(output, pre_features=np.stack(captured), predictions=predictions)
    report.write_text(json.dumps({"complete": True, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "weights_sha256": expected_sha, "output": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "feature_order": ["mu", "var", "sigma", "phi", "Phi", "pk1v"],
        "canonical_permutation": True, "flops": context.summary_dict()["flops_used"],
        "scope": "Target-free instrumented canonical state; trace overhead excludes resource acceptance"}, indent=2)+"\n")
    print(report.read_text(), flush=True)


if __name__ == "__main__":
    main()

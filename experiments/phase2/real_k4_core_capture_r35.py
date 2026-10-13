"""Target-free actual layer14 K31 input/output slice capture, frozen V29."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import flopscope as flops
import numpy as np
from whestbench import SetupContext, load_dataset, mlp_at


def main():
    output = Path("results/phase2/ablation_v30_067_r35_real_k4_core_capture.json")
    cache = Path("benchmark/data/v29_real_k4_core_v30_067_r35")
    assert not output.exists() and not cache.exists()
    cache.mkdir()
    source = Path("estimators/k3_v29/estimator.py").resolve()
    expected = "86d9ca9b28e6fe2b6c74750a0b6ae4bba4c14f742ddc3f5f56bc7fb4ec9d8e27"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location("real_k4_core_frozen_v29", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    trigger = next(i for i, line in enumerate(source.read_text().splitlines(), 1)
                   if "if riders and not NO_CORR:" in line)
    captures = []

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == str(source) and frame.f_lineno == trigger:
            f = frame.f_locals
            if f["li"] == 14 and f["mode"] == 1:
                names = ("mu", "sigma", "alpha", "D3", "D21", "g4row", "wk4m", "wk431",
                         "C_off", "pk1v", "pk2v", "pk11", "pk21", "dG", "lam_prev", "METRIC_C")
                np.savez(cache/"state.npz", **{k: np.asarray(f[k] if k in f else frame.f_globals[k]).copy() for k in names})
                captures.append({"layer": 14, "keys": list(names)})
        return tracer

    data = load_dataset("benchmark/data/mini_first8", split="mini").select_columns(
        ["mlp_id", "mlp_name", "mlp_seed", "weights"])
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        estimator.setup(SetupContext(width=1024, depth=16, flop_budget=2**41,
                                     api_version="0.1", seed=0))
    sys.settrace(tracer)
    try:
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            estimator.predict(mlp_at(data, 0), 2**41)
    finally:
        sys.settrace(None)
    assert len(captures) == 1
    capture = cache/"state.npz"
    report = {"version": "v30.067.r35", "complete": True, "captures": captures,
              "source_sha256": expected, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "capture": str(capture), "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
              "dataset_commit": "aa99830fdc09fad15407b10e8e3459d3e18bba0a", "network_id": "0",
              "legal_predict_cost": ctx.summary_dict(), "no_adjusted_score": True,
              "scope": "Actual regenerated V29 core diagonal/lambda/identity metric parameters and slices; proxies remain approximations",
              "not_resource_acceptance": True}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(captures)


if __name__ == "__main__":
    main()

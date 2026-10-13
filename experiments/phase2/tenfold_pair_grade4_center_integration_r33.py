"""Verify actual candidate centering and source dtype against independent algebra."""
# ruff: noqa: E402
import os
import re

for key in tuple(os.environ):
    if re.match(r"V\d+_", key):
        os.environ.pop(key)
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
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


def load(path):
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    output = Path("results/phase2/tenfold_pair_grade4_center_integration_r33.json")
    assert not output.exists()
    candidate_path = Path("estimators/tenfold_pair_grade4_consistent_r29/estimator.py").resolve()
    parent_path = Path("estimators/sprint_public_v37b_h18_r01/estimator.py").resolve()
    module, parent = load(candidate_path), load(parent_path)
    # comments do not fire line events; use executable original common anchor.
    before = next(i for i, line in enumerate(candidate_path.read_text().splitlines(), 1)
                  if "if riders and not NO_CORR:" in line)
    after = next(i for i, line in enumerate(candidate_path.read_text().splitlines(), 1)
                 if "mu = K1v if delta is None else K1v + delta" in line)
    rng = np.random.default_rng(32001)
    rows = []
    for n in (32, 64):
        pks, captures = {}, []
        def trace(frame, event, arg):
            if event == "line" and frame.f_code.co_filename == str(candidate_path):
                f = frame.f_locals
                if frame.f_lineno == before and not f["last"]:
                    pks[f["li"]] = {k: np.asarray(f[k]).copy().astype(np.float64)
                                    for k in ("pk1v", "pk2v", "pk3v", "pk4v", "pk11", "pk21", "pk22")}
                elif frame.f_lineno == after and f["grade4"] is not None:
                    p = pks[f["li"]]
                    a, b, c, d = [p[f"pk{i}v"] for i in range(1, 5)]
                    expected = {"K1v": a, "K2v": b-a*a, "K3v": c-3*a*b+2*a**3,
                                "K4v": d-4*a*c-3*b*b+12*a*a*b-6*a**4,
                                "K11": p["pk11"], "K21": p["pk21"]-2*a[:, None]*p["pk11"],
                                "K22": p["pk22"]+4*a[:, None]*a[None, :]*p["pk11"]
                                -2*a[None, :]*p["pk21"]-2*a[:, None]*p["pk21"].T-2*p["pk11"]**2}
                    errors = {}
                    for key, value in expected.items():
                        if value.ndim == 2:
                            np.fill_diagonal(value, 0)
                        errors[key] = float(np.max(abs(np.asarray(f[key])-value.astype(np.float32))))
                        assert np.asarray(f[key]).dtype == np.float32
                        assert errors[key] < 1e-6
                    captures.append({"layer": f["li"], "native_float32_histories": True, "conversion_errors": errors})
            return trace
        weights = [rng.normal(0, np.sqrt(2/n), (n, n)).astype(np.float32) for _ in range(4)]
        predictions = []
        for md in (parent, module):
            est = md.Estimator()
            with flops.BudgetContext(flop_budget=2**41, quiet=True):
                est.setup(SetupContext(width=n, depth=4, flop_budget=2**41, api_version="0.1", seed=3909487726))
                mlp = SimpleNamespace(width=n, weights=[fnp.asarray(w) for w in weights])
                if md is module:
                    sys.settrace(trace)
                try:
                    predictions.append(np.asarray(est.predict(mlp, 2**41)).copy())
                finally:
                    sys.settrace(None)
        first = float(np.max(abs(predictions[1][0]-predictions[0][0])))
        assert first == 0 and len(captures) == 2
        rows.append({"width": n, "first_mode0_parent_bitwise": first == 0,
                     "conversion_layers": captures})
    report = {"complete": True, "rows": rows, "source_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Actual wholecall candidate traced before/after center, direct independent ordinarycumulant algebra, float32 storage, firstmode0 parentbitwise. No targets/score/resource acceptance."}
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(output), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

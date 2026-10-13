"""Capture all represented blocks of youngest layer14 source; no target access.

Tracing/conversion is offline diagnosis only and never enters a submitted estimator.
"""

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
    output = Path("results/phase2/ablation_v30_067_r14_full_source_capture.json")
    cache = Path("benchmark/data/v29_real_source_v30_067_r14")
    assert not output.exists() and not cache.exists()
    cache.mkdir()
    source = Path("estimators/k3_v29/estimator.py").resolve()
    expected = "86d9ca9b28e6fe2b6c74750a0b6ae4bba4c14f742ddc3f5f56bc7fb4ec9d8e27"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location("full_b1_v29", source)
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
                index = f["A_st"].shape[0]-1
                a, p = [np.asarray(f[key][index]).copy() for key in ("A_st", "P_st")]
                w2 = np.asarray(f["w2b_list"][index]).copy()
                rres, rfb = f["r"], f["rfb"]
                z = np.asarray(f["Z_st"][index]).copy()
                feedback = np.asarray(f["Zf_st"][index]).copy()
                np.savez(cache/"network_0.npz", a=a, p=p, z=z,
                         ell=np.asarray(f["L_st"][index]).copy(), w2=w2,
                         s=np.asarray(f["s_list"][index]).copy(),
                         e=np.asarray(f["e_list"][index]).copy(),
                         c1=np.asarray(f["c1_list"][index]).copy(),
                         c2=np.asarray(f["c2_list"][index]).copy(), y=z[:, rres+1],
                         f1=feedback[:, :rfb], f2=feedback[:, rfb:],
                         r1=np.asarray(f["R1T_st"][index]).copy(),
                         r2=np.asarray(f["R2T_st"][index]).copy(),
                         sigma=np.asarray(f["sigma"]).copy(), alpha=np.asarray(f["alpha"]).copy(),
                         pk1v=np.asarray(f["pk1v"]).copy())
                captures.append({"layer": 14, "source_index": index, "source_columns": a.shape[1]})
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
    capture = cache/"network_0.npz"
    report = {"version": "v30.067.r14", "complete": True, "captures": captures,
              "source_sha256": expected, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "capture": str(capture), "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
              "dataset_commit": "aa99830fdc09fad15407b10e8e3459d3e18bba0a", "network_id": "0",
              "legal_predict_cost": ctx.summary_dict(),
              "scope": "Target-free youngest layer14 complete represented source B1+feedback/M/feed, all1024columns; other source slots excluded",
              "not_resource_acceptance": True, "no_adjusted_score": True}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(captures)


if __name__ == "__main__":
    main()

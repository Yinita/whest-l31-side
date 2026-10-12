"""Trace actual V29 layer14 source column factors, target-free offline diagnostic."""

# ruff: noqa: E402
import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import flopscope as flops
import numpy as np
from whestbench import SetupContext, load_dataset, mlp_at


def main():
    report = Path("results/phase2/ablation_v30_059_r04_real_source_columns.json")
    assert not report.exists()
    cache = Path("benchmark/data/v29_real_source_v30_059_r04")
    cache.mkdir(exist_ok=False)
    source = Path("estimators/k3_v29/estimator.py").resolve()
    spec = importlib.util.spec_from_file_location("real_source_v29", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    trigger = next(
        i
        for i, line in enumerate(source.read_text().splitlines(), 1)
        if "if riders and not NO_CORR:" in line
    )
    captures = []

    def tracer(frame, event, arg):
        if (
            event == "line"
            and frame.f_code.co_filename == str(source)
            and frame.f_lineno == trigger
        ):
            f = frame.f_locals
            if f["li"] == 14 and f["mode"] == 1:
                index = f["A_st"].shape[0] - 1
                a = np.asarray(f["A_st"][index]).copy()
                p = np.asarray(f["P_st"][index]).copy()
                w2 = np.asarray(f["w2b_list"][index]).copy()
                # First16 fixed newborn-source columns, selection does not access target.
                selected = np.arange(16)
                # B1=3Sym(A,P,A diag(w2)); omit feedback/feed/M explicitly.
                np.savez(
                    cache / "network_0.npz",
                    a=a[:, selected] * 3,
                    p=p[:, selected],
                    c=a[:, selected] * w2[selected],
                    covariance=np.asarray(f["C_off"]).copy(),
                    w1=np.asarray(f["Phi"]).copy(),
                    w2=np.asarray(f["W_all"][:, estimator._i21]).copy(),
                    sigma=np.asarray(f["sigma"]).copy(),
                    alpha=np.asarray(f["alpha"]).copy(),
                )
                captures.append(
                    dict(
                        layer=f["li"],
                        source_index=index,
                        selected_columns=selected.tolist(),
                        a_norm=float(np.linalg.norm(a[:, selected])),
                        p_norm=float(np.linalg.norm(p[:, selected])),
                    )
                )
        return tracer

    data = load_dataset("benchmark/data/mini_first8", split="mini").select_columns(
        ["mlp_id", "mlp_name", "mlp_seed", "weights"]
    )
    with flops.BudgetContext(flop_budget=2**41):
        estimator.setup(
            SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0)
        )
    sys.settrace(tracer)
    try:
        with flops.BudgetContext(flop_budget=2**41):
            estimator.predict(mlp_at(data, 0), 2**41)
    finally:
        sys.settrace(None)
    assert len(captures) == 1
    report.write_text(
        json.dumps(
            dict(
                captures=captures,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                cache=str(cache),
                scope="Target-free actuallayer14prestate fixed16 young-source B1columns; excludes feedback/feed/M/allothercolumns/sources; not full K111 and no candidate score",
            ),
            indent=2,
        )
        + "\n"
    )
    print(captures)


if __name__ == "__main__":
    main()

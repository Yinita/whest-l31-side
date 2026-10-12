"""Offline real V29 symmetry capture, no candidate or metering claim."""

import importlib.util
import json
import sys
from pathlib import Path

import flopscope as flops
import numpy as np
from whestbench import SetupContext, load_dataset, mlp_at


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "estimators/k3_v29/estimator.py"
    output = root / "results/phase2/ablation_v30_023_symmetry_trace.json"
    assert not output.exists()
    spec = importlib.util.spec_from_file_location("symmetry_v29", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_product = module.Estimator._sym_product
    product_rows = []

    def intercept(self, X, Y, n, out, lev):
        value = np.asarray(X, dtype=np.float64) @ np.asarray(Y, dtype=np.float64)
        product_rows.append(
            dict(
                n=n,
                max_raw_product_asymmetry=float(abs(value - value.T).max()),
                relative_raw_product_asymmetry=float(
                    np.linalg.norm(value - value.T) / max(np.linalg.norm(value), 1e-30)
                ),
            )
        )
        return original_product(self, X, Y, n, out, lev)

    module.Estimator._sym_product = intercept
    estimator = module.Estimator()
    rows = []
    trigger = next(
        i
        for i, line in enumerate(source.read_text().splitlines(), 1)
        if "if riders and not NO_CORR:" in line
    )

    def measure(a):
        value = np.asarray(a, dtype=np.float64)
        return dict(
            max_asymmetry=float(abs(value - value.T).max()),
            relative_norm_asymmetry=float(
                np.linalg.norm(value - value.T) / max(np.linalg.norm(value), 1e-30)
            ),
        )

    def trace(frame, event, arg):
        if (
            event == "line"
            and frame.f_code.co_filename == str(source)
            and frame.f_lineno == trigger
        ):
            f = frame.f_locals
            record = dict(layer=f["li"], covariance=measure(f["C"]))
            for name in ["C_pre", "Sg", "S2", "Sj"]:
                if f.get(name) is not None:
                    record[name] = measure(f[name])
            rows.append(record)
        return trace

    data = load_dataset("benchmark/data/mini_first8", split="mini")
    mlp = mlp_at(data, 0)
    with flops.BudgetContext(flop_budget=2**41):
        estimator.setup(
            SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0)
        )
    sys.settrace(trace)
    try:
        with flops.BudgetContext(flop_budget=2**41) as ctx:
            prediction = estimator.predict(mlp, 2**41)
    finally:
        sys.settrace(None)
    output.write_text(
        json.dumps(
            dict(
                scope="Actual network0 legal V29 state captured offline; no timing score, no extra candidate operations",
                rows=rows,
                product_rows=product_rows,
                flops=ctx.summary_dict()["flops_used"],
                prediction_finite=bool(np.isfinite(np.asarray(prediction)).all()),
                limitation="Raw X@Y asymmetry before symmetric copy requires separate product interception; this records post-assembly and state cores",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

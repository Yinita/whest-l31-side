# ruff: noqa: E402
"""Offline trace captures original V29 features without adding flopscope ops.

Trace copying and NumPy derived features are diagnostic-only; timing/resource
cannot be reported as an admissible candidate. Original reference never enters
predict or feature construction. Serialized arrays stay in ignored input cache.
"""

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

FEATURES = [
    "ones",
    "mu",
    "var",
    "sigma",
    "alpha",
    "abs_alpha",
    "phi",
    "Phi",
    "pk1",
    "D3",
    "D21norm",
    "K4_sigma",
    "sigma_phi",
]


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "estimators/k3_v29/estimator.py"
    report = root / "results/phase2/ablation_v30_096_r03_samecycle_features_manifest.json"
    assert not report.exists()
    cache = root / "benchmark/data/v29_features_v30_096_r03"
    cache.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location("feature_v29", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    dataset = load_dataset("benchmark/data/mini_first8", split="mini")
    trigger = next(
        i
        for i, line in enumerate(source.read_text().splitlines(), 1)
        if "if riders and not NO_CORR:" in line
    )
    frozen = json.loads((root / "results/phase2/v29_mini8_local.json").read_text())["results"][
        "per_mlp"
    ]
    controls = []
    for _ in range(2):
        control = module.Estimator()
        with flops.BudgetContext(flop_budget=2**41):
            control.setup(SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0))
        controls.append(control)
    records = []
    with flops.BudgetContext(flop_budget=2**41):
        estimator.setup(
            SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0)
        )
    for index in range(2):
        captured = []

        def tracer(frame, event, arg):
            if (
                event == "line"
                and frame.f_code.co_filename == str(source)
                and frame.f_lineno == trigger
            ):
                f = frame.f_locals

                def array(name):
                    return np.array(f[name], dtype=np.float32, copy=True)

                ones = array("ones_n")
                mu = array("mu")
                var = array("var")
                sigma = array("sigma")
                alpha = array("alpha")
                phi = array("phi")
                cdf = array("Phi")
                pk = array("pk1v")
                if f["mode"] == 1:
                    d3 = array("D3")
                    d21 = (
                        np.abs(np.asarray(f["D21"])).sum(axis=1, dtype=np.float32)
                        if f["D21"] is not None
                        else ones * 0
                    )
                    k4 = ones * float(f["K4_sigma"])
                else:
                    d3 = ones * 0
                    d21 = ones * 0
                    k4 = ones * 0
                captured.append(
                    np.stack(
                        [
                            ones,
                            mu,
                            var,
                            sigma,
                            alpha,
                            abs(alpha),
                            phi,
                            cdf,
                            pk,
                            d3,
                            d21,
                            k4,
                            sigma * phi,
                        ],
                        axis=1,
                    )
                )
            return tracer

        mlp = mlp_at(dataset, index)
        sys.settrace(tracer)
        try:
            with flops.BudgetContext(flop_budget=2**41) as ctx:
                prediction = estimator.predict(mlp, 2**41)
        finally:
            sys.settrace(None)
        pred = np.asarray(prediction, dtype=np.float32).copy()
        features = np.stack(captured)
        assert features.shape == (16, 1024, 13)
        target = np.asarray(dataset[index]["all_layer_means"], dtype=np.float32)
        output = cache / f"network_{index}.npz"
        np.savez(output, features=features, predictions=pred, reference=target)
        summary = ctx.summary_dict()
        repeated = []
        for control in controls:
            with flops.BudgetContext(flop_budget=2**41) as untraced_ctx:
                repeated.append(np.asarray(control.predict(mlp, 2**41), dtype=np.float32).copy())
            assert untraced_ctx.summary_dict()["flops_used"] == summary["flops_used"]
        trace_delta = float(np.max(abs(pred - repeated[0])))
        repeat_delta = float(np.max(abs(repeated[0] - repeated[1])))
        raw = float(np.mean((pred[-1] - target[-1]) ** 2))
        assert summary["flops_used"] == frozen[index]["flops_used"]
        records.append(
            dict(
                network_id=index,
                name=mlp.name,
                seed=mlp.seed,
                path=str(output.relative_to(root)),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                flops=summary["flops_used"],
                raw=raw,
                trace_control_maxabs=trace_delta,
                repeat_control_maxabs=repeat_delta,
                within_repeat_envelope=trace_delta <= repeat_delta,
                frozen_raw=frozen[index]["final_layer_mse"],
                raw_relative_difference=raw / frozen[index]["final_layer_mse"] - 1,
            )
        )
        report.with_suffix(".progress.json").write_text(
            json.dumps(dict(completed=index + 1, records=records), indent=2) + "\n"
        )
        print("cached", index, flush=True)
    report.write_text(
        json.dumps(
            dict(
                scope="Offline same-source trace, development0-7 only; no online candidate and no official time/resource pass",
                feature_order=FEATURES,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                records=records,
                limitations=[
                    "Trace changes residual timing; no timing acceptance claim",
                    "Features D21norm computed offline; no meter bypass in submission",
                    "Reference arrays serialized after prediction; never used to compute features",
                ],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""O27 reserved public network state trace, explicitly avoids reference columns."""

# ruff: noqa: E402 -- freeze threads before imports
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
    protocol = json.loads(
        Path("results/phase2/ablation_v30_089_frozen_shift_features.json").read_text()
    )
    report = Path("results/phase2/ablation_v30_089_reserved_state_only.json")
    assert not report.exists()
    source = Path("estimators/k3_v29/estimator.py").resolve()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol["source_sha256"]
    spec = importlib.util.spec_from_file_location("reserved_v29", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    # Delete targets at dataset interface before any per-network access.
    dataset = load_dataset("benchmark/data/mini_shard01", split="mini").select_columns(
        ["mlp_id", "mlp_name", "mlp_seed", "weights"]
    )
    trigger = next(
        i
        for i, line in enumerate(source.read_text().splitlines(), 1)
        if "if riders and not NO_CORR:" in line
    )
    with flops.BudgetContext(flop_budget=2**41):
        estimator.setup(
            SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0)
        )
    rows = []
    for index in range(8):
        captures = []

        def tracer(frame, event, arg):
            if (
                event == "line"
                and frame.f_code.co_filename == str(source)
                and frame.f_lineno == trigger
            ):
                f = frame.f_locals
                sigma = np.asarray(f["sigma"])
                alpha = np.asarray(f["alpha"])
                d3 = np.asarray(f["D3"]) if f["mode"] == 1 else np.zeros(1024)
                k4 = np.full(1024, float(f["K4_sigma"])) if f["mode"] == 1 else np.zeros(1024)
                arrays = [
                    np.asarray(f["var"]),
                    alpha,
                    d3 / sigma**3,
                    k4 / sigma**4,
                    np.asarray(f["Phi"]),
                ]
                captures.append(
                    {
                        name: np.quantile(x, protocol["quantile_levels"]).tolist()
                        for name, x in zip(protocol["names"], arrays)
                    }
                )
            return tracer

        sys.settrace(tracer)
        try:
            with flops.BudgetContext(flop_budget=2**41):
                estimator.predict(mlp_at(dataset, index), 2**41)
        finally:
            sys.settrace(None)
        assert len(captures) == 16
        rows.append(dict(network_id=str(dataset[index]["mlp_id"]), layers=captures))
        print("Reserved state", index, flush=True)
        report.with_suffix(".progress.json").write_text(
            json.dumps(dict(rows=rows, complete=False), indent=2) + "\n"
        )
    report.write_text(
        json.dumps(
            dict(
                rows=rows,
                labels_read=False,
                features=protocol["names"],
                scope="Public reserved state diagnostic; inspecting state consumes this feature-distribution holdout, no score or independent acceptance",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

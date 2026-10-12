"""L07 real per-neuron g4row trace; diagnostic only, no uncharged candidate."""

import hashlib
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
    report = root / "results/phase2/ablation_v30_102_g4_features.json"
    assert not report.exists()
    cache = root / "benchmark/data/v29_g4_v30_102"
    cache.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location("g4_v29", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    trigger = next(
        i
        for i, line in enumerate(source.read_text().splitlines(), 1)
        if "if riders and not NO_CORR:" in line
    )
    dataset = load_dataset("benchmark/data/mini_first8", split="mini")
    with flops.BudgetContext(flop_budget=2**41):
        estimator.setup(
            SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0)
        )
    rows = []
    for index in range(8):
        captures = []
        base_captures = []

        def tracer(frame, event, arg):
            if (
                event == "line"
                and frame.f_code.co_filename == str(source)
                and frame.f_lineno == trigger
            ):
                f = frame.f_locals
                g = np.asarray(f["g4row"]).copy() if f["mode"] == 1 else np.zeros(1024)
                sigma = np.asarray(f["sigma"]).copy()
                phi = np.asarray(f["phi"]).copy()
                captures.append(np.stack([g / sigma**4, g / sigma**4 * phi], axis=1))
                ones = np.asarray(f["ones_n"]).copy()
                mu = np.asarray(f["mu"]).copy()
                var = np.asarray(f["var"]).copy()
                alpha = np.asarray(f["alpha"]).copy()
                cdf = np.asarray(f["Phi"]).copy()
                pk = np.asarray(f["pk1v"]).copy()
                d3 = np.asarray(f["D3"]).copy() if f["mode"] == 1 else ones * 0
                d21 = (
                    np.abs(np.asarray(f["D21"])).sum(axis=1)
                    if f["mode"] == 1 and f["D21"] is not None
                    else ones * 0
                )
                k4 = ones * float(f["K4_sigma"]) if f["mode"] == 1 else ones * 0
                base_captures.append(
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

        sys.settrace(tracer)
        try:
            with flops.BudgetContext(flop_budget=2**41) as ctx:
                pred = estimator.predict(mlp_at(dataset, index), 2**41)
        finally:
            sys.settrace(None)
        features = np.stack(captures)
        assert features.shape == (16, 1024, 2)
        old = dict(np.load(root / f"benchmark/data/v29_features_v30_096/network_{index}.npz"))
        prediction = np.asarray(pred).copy()
        path = cache / f"network_{index}.npz"
        np.savez(
            path, features=features, base_features=np.stack(base_captures), predictions=prediction
        )
        rows.append(
            dict(
                network_id=index,
                flops=ctx.summary_dict()["flops_used"],
                path=str(path.relative_to(root)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                max_prediction_delta_vs_oldcache=float(
                    np.max(abs(prediction - old["predictions"]))
                ),
            )
        )
        print("g4", index, flush=True)
    report.write_text(
        json.dumps(
            dict(
                rows=rows,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                scope="Development offline traced real g4row; reference not used; cache from different run must not silently merge without prediction parity gate; no time/resource acceptance",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

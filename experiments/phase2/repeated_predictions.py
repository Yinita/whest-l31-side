"""Three independent estimator instances on one fixed network; offline drift."""

import fcntl
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import numpy as np
from whestbench import SetupContext, load_dataset, mlp_at


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "results/phase2/ablation_v30_025_prediction_drift.json"
    assert not path.exists()
    lock = (root / 'benchmark/data/ablation_execution.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dataset = load_dataset("benchmark/data/mini_first8", split="mini")
    mlp = mlp_at(dataset, 0)
    predictions = []
    rows = []
    for repeat in range(3):
        spec = importlib.util.spec_from_file_location(
            f"v29_repeat_{repeat}", root / "estimators/k3_v29/estimator.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        estimator = module.Estimator()
        with flops.BudgetContext(flop_budget=2**41):
            estimator.setup(
                SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0)
            )
        with flops.BudgetContext(flop_budget=2**41) as context:
            prediction = estimator.predict(mlp, 2**41)
        values = np.asarray(prediction.tolist(), dtype=np.float32)
        predictions.append(values)
        scratch = []

        def visit(value, seen):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, dict):
                for item in value.values():
                    visit(item, seen)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item, seen)
            elif hasattr(value, "shape") and hasattr(value, "size"):
                scratch.append(dict(shape=list(value.shape), logical_bytes=int(value.size) * 4))

        seen = set()
        visit(estimator._pool.bufs, seen)
        visit(estimator._smm.pool, seen)
        pool_report = dict(
            repeat=repeat,
            arrays=scratch,
            logical_bytes_sum=sum(item["logical_bytes"] for item in scratch),
            limitation="Views may alias, logical bytes are not unique resident memory; excludes BLAS temporary allocations",
        )
        path.with_name(f"ablation_v30_026_full_pool_r{repeat}.json").write_text(
            json.dumps(pool_report, indent=2) + "\n"
        )
        rows.append(dict(repeat=repeat, budget_summary=context.summary_dict()))
        path.with_suffix(".progress.json").write_text(
            json.dumps(dict(completed=repeat + 1, rows=rows), indent=2) + "\n"
        )
    comparisons = []
    for i in range(3):
        for j in range(i):
            delta = predictions[i].astype(np.float64) - predictions[j]
            comparisons.append(
                dict(
                    left=j,
                    right=i,
                    per_layer_max_abs=np.max(abs(delta), axis=1).tolist(),
                    per_layer_mean_squared_difference=np.mean(delta**2, axis=1).tolist(),
                )
            )
    np.savez(path.with_suffix(".npz"), predictions=np.stack(predictions))
    path.write_text(
        json.dumps(
            dict(
                scope="Offline three independent V29 instances, original network0; direct predictions and charged FLOPs; not grader score",
                rows=rows,
                comparisons=comparisons,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

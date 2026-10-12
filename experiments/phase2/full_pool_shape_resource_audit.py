"""Offline persistent V29 pool shape lifetime audit under original address-space cap."""

# ruff: noqa: E402
import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import importlib.util
import json
import resource
import time
from pathlib import Path

import flopscope as flops
from whestbench import MLP, SetupContext, load_dataset, mlp_at


def memory():
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmSize:", "VmRSS:", "VmPeak:", "VmHWM:")):
            name, value, _ = line.split()
            fields[name.rstrip(":")] = int(value)
    return fields


def main():
    path = Path("results/phase2/ablation_v30_026_r03_shape_resource.json")
    assert not path.exists()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    source = Path("estimators/k3_v29/estimator.py").resolve()
    spec = importlib.util.spec_from_file_location("pool_v29", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    with flops.BudgetContext(flop_budget=2**41):
        estimator.setup(
            SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0)
        )
    data = load_dataset("benchmark/data/mini_first8", split="mini").select_columns(
        ["mlp_id", "mlp_name", "mlp_seed", "weights"]
    )
    full = mlp_at(data, 0)
    rows = []
    for width, depth in [(64, 4), (1024, 16), (64, 4), (1024, 16)]:
        started = time.monotonic()
        before = memory()
        failure = None
        try:
            with flops.BudgetContext(flop_budget=2**41) as ctx:
                network = MLP(
                    width=width,
                    depth=depth,
                    weights=[w[:width, :width] for w in full.weights[:depth]],
                )
                estimator.predict(network, 2**41)
            charged = ctx.summary_dict()["flops_used"]
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            charged = None
        pool = getattr(estimator, "_pool", None)
        arrays = (
            []
            if pool is None
            else [
                dict(key=repr(k), shape=list(v.shape), logical_bytes=v.size * 4)
                for k, slot in pool.bufs.items()
                for v in (slot if isinstance(slot, tuple) else (slot,))
            ]
        )
        rows.append(
            dict(
                width=width,
                depth=depth,
                elapsed_s=time.monotonic() - started,
                before_memory_kib=before,
                after_memory_kib=memory(),
                failure=failure,
                flops=charged,
                pool_arrays=arrays,
                logical_bytes=sum(a["logical_bytes"] for a in arrays),
            )
        )
        path.with_suffix(".progress.json").write_text(
            json.dumps({"rows": rows, "complete": False}, indent=2) + "\n"
        )
        print(width, depth, failure, flush=True)
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                rlimit_as_bytes=8 * 1024**3,
                scope="Offline NumPy backend diagnostic with actual predict; slicing weights changes shape only. No accuracy score or RPC equivalence claim",
                logical_bytes_caveat="Views can alias; logical sum does not equal resident memory",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

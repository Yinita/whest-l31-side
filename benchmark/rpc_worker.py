"""Local development RPC probe: unmodified estimator, official SDK and client."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import resource
import sys
import types
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-mlps", type=int, default=8)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    import flopscope as flops
    import flopscope.numpy as fnp

    # Load the actual installed SDK/domain source without the server-only
    # whestbench.__init__ imports. These canonical classes are not reimplemented.
    sdk_root = Path(args.sdk_root)
    package = types.ModuleType("whestbench")
    package.__path__ = [str(sdk_root)]
    sys.modules["whestbench"] = package
    domain = load("whestbench.domain", sdk_root / "domain.py")
    sdk = load("whestbench.sdk", sdk_root / "sdk.py")
    package.MLP = domain.MLP
    package.BaseEstimator = sdk.BaseEstimator
    package.SetupContext = sdk.SetupContext
    module = load("benchmark_estimator", Path(args.estimator))
    estimator = module.Estimator()
    with flops.BudgetContext(flop_budget=2**41):
        estimator.setup(
            sdk.SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="0.1", seed=0)
        )
    results = []
    manifest = {
        "flopscope_client_version": importlib.metadata.version("flopscope-client"),
        "estimator_sha256": hashlib.sha256(Path(args.estimator).read_bytes()).hexdigest(),
        "sdk_sha256": hashlib.sha256((sdk_root / "sdk.py").read_bytes()).hexdigest(),
        "domain_sha256": hashlib.sha256((sdk_root / "domain.py").read_bytes()).hexdigest(),
        "mode": "local tokenless development RPC, not the official hardened grader",
    }
    output = Path(args.output)
    for entry in json.loads(Path(args.inputs).read_text())[: args.n_mlps]:
        with flops.BudgetContext(flop_budget=2**41):
            weights = fnp.load(entry["weights"])["weights"]
            mlp = domain.MLP(
                width=1024,
                depth=16,
                weights=[weights[j] for j in range(16)],
                seed=entry["seed"],
                name=entry["name"],
            )
        with flops.BudgetContext(flop_budget=2**41) as ctx:
            predictions = estimator.predict(mlp, 2**41)
        summary = ctx.summary_dict()
        with flops.BudgetContext(flop_budget=2**41):
            values = predictions.tolist()
        results.append(
            {
                "mlp_index": entry["index"],
                "mlp_name": entry["name"],
                "seed": entry["seed"],
                "predictions": values,
                "budget_summary": summary,
                "client_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        output.write_text(json.dumps({"manifest": manifest, "per_mlp": results}) + "\n")
        print(f"RPC network {entry['index']} completed", flush=True)


if __name__ == "__main__":
    main()

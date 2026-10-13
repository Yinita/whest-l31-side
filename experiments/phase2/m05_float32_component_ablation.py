"""Single-change dtype cost/accuracy ablation of frozen full M05 components."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import m05_float32_method as f32
import m05_small_split_ut as f64
import numpy as np


def mechanism_contract():
    source = Path("experiments/phase2/m05_small_split_ut.py").read_text()
    derived = Path("experiments/phase2/m05_float32_method.py").read_text()
    originals = {x.name: x for x in ast.parse(source.replace("fnp.float64", "fnp.float32")).body if isinstance(x, ast.FunctionDef)}
    copies = {x.name: x for x in ast.parse(derived).body if isinstance(x, ast.FunctionDef)}
    for name, node in copies.items():
        assert ast.dump(node, include_attributes=False) == ast.dump(originals[name], include_attributes=False)
    return {"same_function_AST_after_dtype_replacement": True, "functions": sorted(copies)}


def main():
    output = Path("results/phase2/m05_float32_component_ablation_r10.json")
    assert not output.exists()
    rows = []
    prior = json.loads(Path("results/phase2/m05_highwidth_component_cost_r07.json").read_text())
    for n in (3, 9, 128, 1024):
        seed = 61013200 + n
        rng = np.random.default_rng(seed)
        weights = rng.normal(size=(16, n, n)) * np.sqrt(2/n)
        mean = rng.normal(size=n) * .1
        values = {}
        for dtype, method in (("float64", f64), ("float32", f32)):
            npdtype = getattr(np, dtype)
            fdtype = getattr(fnp, dtype)
            with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=120, quiet=True) as ctx:
                result_mean, result_cov, criterion, _, _ = method.component(
                    fnp.asarray(mean.astype(npdtype)), fnp.eye(n, dtype=fdtype),
                    [fnp.asarray(w.astype(npdtype)) for w in weights], "paper_sum")
            m, c = np.asarray(result_mean), np.asarray(result_cov)
            assert str(m.dtype) == dtype and str(c.dtype) == dtype
            assert np.isfinite(m).all() and np.isfinite(c).all()
            values[dtype] = {"mean": m, "cov": c, "criterion": criterion, "cost": ctx.summary_dict(by_namespace=True)}
        before, after = values["float64"], values["float32"]
        if n in (128, 1024):
            reference = next(x for x in prior["rows"] if x["width"] == n)
            assert hashlib.sha256(weights.tobytes()).hexdigest() == reference["weights_sha256"]
            assert before["cost"]["flops_used"] == reference["legal_cost"]["flops_used"]
        mean_delta = after["mean"].astype(np.float64)-before["mean"]
        cov_delta = after["cov"].astype(np.float64)-before["cov"]
        row = {"width": n, "depth": 16, "seed": seed,
               "weights_before_cast_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
               "mean_max_abs_delta": float(np.abs(mean_delta).max()),
               "mean_delta_mse": float(np.mean(mean_delta**2)),
               "cov_max_abs_delta": float(np.abs(cov_delta).max()),
               "cov_relative_frobenius_delta": float(np.linalg.norm(cov_delta)/max(np.linalg.norm(before["cov"]), 1e-30)),
               "criterion_float64": before["criterion"], "criterion_float32": after["criterion"],
               "before_legal_cost": before["cost"], "after_legal_cost": after["cost"],
               "relative_flops": after["cost"]["flops_used"]/before["cost"]["flops_used"]-1,
               "component_evaluation_count_before_overhead": int(2**41//after["cost"]["flops_used"])}
        rows.append(row)
        print(n, row["relative_flops"], row["mean_delta_mse"], row["component_evaluation_count_before_overhead"], flush=True)
    report = {"version": "paper.M05.r10.dtype", "complete": True, "single_change": "float64 -> float32",
              "mechanism_contract": mechanism_contract(), "rows": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "method_before_sha256": hashlib.sha256(Path("experiments/phase2/m05_small_split_ut.py").read_bytes()).hexdigest(),
              "method_after_sha256": hashlib.sha256(Path("experiments/phase2/m05_float32_method.py").read_bytes()).hexdigest(),
              "versions": {p: importlib.metadata.version(p) for p in ("flopscope", "whestbench", "numpy")},
              "scope": "Dtype drift and measured billing only; neither dtype prediction is an accuracy reference; no competition score or RPC acceptance"}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()

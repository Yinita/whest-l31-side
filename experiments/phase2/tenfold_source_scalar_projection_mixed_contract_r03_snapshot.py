"""Bounded mixed-dtype parity of source numerator versus frozen r01 cap.

No full1024 benchmark, reference data, shared candidate edits, or state edits.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from tenfold_source_scalar_projection import source_projection


def main():
    output = Path("results/phase2/tenfold_source_scalar_projection_mixed_r03.json")
    assert not output.exists()
    candidate = Path("estimators/tenfold_fullcore_li14_r01/estimator.py")
    spec = importlib.util.spec_from_file_location("tenfold_fullcore_frozen_contract", candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(673803)
    rows = []
    for n in (8, 32):
        a, p = [rng.normal(size=(n, n)).astype(np.float32) for _ in range(2)]
        z, ell = [rng.normal(size=(n, 3)).astype(np.float32) for _ in range(2)]
        w2, s, e, c1, c2, y = [rng.normal(size=n).astype(np.float32) for _ in range(6)]
        f1, f2, r1, r2 = [rng.normal(size=(n, 2)).astype(np.float32) for _ in range(4)]
        u, v = [rng.normal(size=n) for _ in range(2)]
        raw = rng.normal(size=(n, n))
        covariance = (raw+raw.T)/2
        np.fill_diagonal(covariance, 0)
        vals = (a, p, z, ell, w2, s, e, c1, c2, y, f1, f2, r1, r2)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as before_ctx:
            operands = list(map(fnp.asarray, vals))
            cap = module.capped_structured_source(*operands, fnp.asarray(u), fnp.asarray(v))
            before = fnp.sum(cap*fnp.asarray(covariance))
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as after_ctx:
            h = fnp.asarray(covariance)*fnp.asarray(v)[:, None]*fnp.asarray(v)[None, :]
            after = source_projection(h, *map(fnp.asarray, vals), fnp.asarray(u))
        absolute = abs(float(after)-float(before))
        relative = absolute/max(abs(float(before)), 1e-30)
        assert relative < 3e-5, relative
        rows.append({"width": n, "source_dtype": "float32", "weight_dtype": "float64",
                     "source_columns": n, "before": float(before), "after": float(after),
                     "absolute_difference": absolute, "relative_difference": relative,
                     "before_flops": before_ctx.summary_dict()["flops_used"],
                     "after_flops": after_ctx.summary_dict()["flops_used"]})
    report = {"version": "tenfold.scalarprojection.r03", "complete": True, "rows": rows,
              "candidate": str(candidate), "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256(Path("experiments/phase2/tenfold_source_scalar_projection.py").read_bytes()).hexdigest(),
              "scope": "Same mixed-dtype source operands and weights, exact contraction regrouping within float32 rounding",
              "no_adjusted_score": True, "limitations": ["Bounded n8/32 only; actual1024 resource/parity remains pending"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

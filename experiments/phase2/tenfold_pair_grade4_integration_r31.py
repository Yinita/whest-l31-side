"""Charged production helper parity plus whole small-network candidate contract."""
# ruff: noqa: E402
import os
import re

for key in tuple(os.environ):
    if re.match(r"V\d+_", key):
        os.environ.pop(key)
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext

from experiments.phase2.tenfold_pair_grade4_contract_r26 import charged_grade4, table, wicks


def main():
    output = Path("results/phase2/tenfold_pair_grade4_integration_r31.json")
    assert not output.exists()
    source = Path("estimators/tenfold_pair_grade4_consistent_r29/estimator.py")
    spec = importlib.util.spec_from_file_location("pair_grade4_r29", source)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    terms, _ = table()
    terms = {k: v for k, v in terms.items() if k != (3, 1)}
    assert terms == candidate.GRADE4_TERMS
    rng = np.random.default_rng(29001)
    rows = []
    for n in (4, 8, 32):
        mu = rng.uniform(-1, 1, n)
        sigma = rng.uniform(.8, 1.3, n)
        values = {(3,): rng.normal(0, .01, n), (4,): rng.normal(0, .015, n)}
        for k in ((1, 1), (2, 1), (3, 1), (2, 2)):
            values[k] = rng.normal(0, .001, (n, n))
            np.fill_diagonal(values[k], 0)
        values[1, 2] = values[2, 1].T
        values[1, 3] = values[3, 1].T
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            inp = {k: fnp.asarray(v) for k, v in values.items()}
            actual = candidate.pair_grade4_increment(fnp.asarray(mu), fnp.asarray(sigma), inp)
            expected = charged_grade4(fnp.asarray(mu), fnp.asarray(sigma), inp, terms)
            ww = wicks(fnp.asarray(mu), fnp.asarray(sigma), {(6, 1), (7, 1)})
            expected[1,] = expected[1,]-inp[3,]**2*ww[6, 1]/72-inp[3,]*inp[4,]*ww[7, 1]/144
            error = max(float(np.max(np.abs(np.asarray(actual[k])-np.asarray(expected[k])))) for k in expected)
            trim = candidate.pair_grade4_increment(fnp.asarray(mu), fnp.asarray(sigma),
                                                   {(3,): inp[3,], (4,): inp[4,]}, trim=True)
            trim_error = float(np.max(abs(np.asarray(trim[1,])-np.asarray(actual[1,]))))
        assert error < 1e-12 and trim_error == 0
        rows.append({"n": n, "helper_abs_error": error, "trim_abs_error": trim_error,
                     "component_flops": ctx.summary_dict()["flops_used"]})
    for n, depth in ((32, 4), (64, 4)):
        estimator = candidate.Estimator()
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            estimator.setup(SetupContext(width=n, depth=depth, flop_budget=2**41, api_version="0.1", seed=3909487726))
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            weights = [fnp.asarray(rng.normal(0, np.sqrt(2/n), (n, n)).astype(np.float32)) for _ in range(depth)]
            prediction = np.asarray(estimator.predict(SimpleNamespace(width=n, weights=weights), 2**41)).copy()
        assert prediction.shape == (depth, n) and np.isfinite(prediction).all()
        assert prediction.dtype == np.float32
        rows.append({"wholecall_n": n, "depth": depth, "finite": True, "dtype": str(prediction.dtype),
                     "flops": ctx.summary_dict()["flops_used"], "prediction_maxabs": float(np.max(abs(prediction)))})
    result = {"complete": True, "rows": rows, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_contract_sha256": hashlib.sha256(Path("experiments/phase2/tenfold_pair_grade4_contract_r26.py").read_bytes()).hexdigest(),
              "scope": "Actual production helper equals independent fixed officialtable implementation incl p1dedup and finaltrim; whole charged n32/64depth4 finite/dtype/integration. No targets; smallshape not production1024 or RPCresource acceptance."}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

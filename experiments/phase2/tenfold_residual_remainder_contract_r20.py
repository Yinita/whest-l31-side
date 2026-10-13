"""Actual birth-residual remainder and transported CP(M,P,P) contracts."""
# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext

SOURCE = Path("estimators/tenfold_residual_remainder_r20/estimator.py")
OUTPUT = Path("results/phase2/tenfold_residual_remainder_contract_r20.json")


def sym(a, b, c):
    return sum(np.einsum("ir,jr,kr->ijk", *x)
               for x in itertools.permutations((a, b, c))) / 6


def main():
    assert not OUTPUT.exists()
    spec = importlib.util.spec_from_file_location("residual_r20", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(201310)
    rows = []
    for n in (4, 8, 32):
        rank = min(3, n)
        residual = rng.normal(size=(n, n))
        q, _ = np.linalg.qr(rng.normal(size=(n, rank)))
        bm = q.T @ residual
        delta = 3 * (residual - q @ bm).T
        p = np.eye(n)
        dense = sym(delta, p, p)
        initial21 = np.array([[dense[i, i, j] if i != j else 0
                              for j in range(n)] for i in range(n)])
        expected21 = residual - q @ bm
        np.fill_diagonal(expected21, 0)
        assert np.max(np.abs(initial21 - expected21)) < 1e-13
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            stack = fnp.asarray(np.stack([delta, p])[None])
            smm = module._Strassen(fnp.float64, 4)
            pool = module._Pool(fnp.float64)
            errs = []
            for age in range(3):
                w = rng.normal(size=(n, n)) / np.sqrt(n)
                if age:
                    w = w * rng.uniform(0.1, 0.9, size=n)[None, :]
                dst = fnp.empty(stack.shape, dtype=fnp.float64)
                smm.mm(fnp.asarray(w)[None, None], stack, dst, 0)
                stack = dst
                got3, got21 = module.residual_remainder_slices(stack, pool, smm, n, 0)
                dense = np.einsum("ia,jb,kc,abc->ijk", w, w, w, dense, optimize=True)
                expected3 = np.array([dense[i, i, i] for i in range(n)])
                expected21 = np.array([[dense[i, i, j] if i != j else 0
                                        for j in range(n)] for i in range(n)])
                errs.append(max(float(np.max(np.abs(np.asarray(got3) - expected3))),
                                float(np.max(np.abs(np.asarray(got21) - expected21)))))
            assert max(errs) < 1e-12
        rows.append({"width": n, "birth_residual_parity": True,
                     "transport_abs_errors": errs, "legal_flops": ctx.summary_dict()["flops_used"]})
    from types import SimpleNamespace

    n = 32
    weights = [rng.normal(size=(n, n)).astype(np.float32) / np.sqrt(n) for _ in range(4)]
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        estimator = module.Estimator()
        estimator.setup(SetupContext(width=n, depth=4, flop_budget=2**41,
                                     api_version="0.1", seed=0))
        mlp = SimpleNamespace(width=n, weights=[fnp.asarray(x) for x in weights])
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        prediction = np.asarray(estimator.predict(mlp, 2**41))
    assert prediction.shape == (4, n) and np.isfinite(prediction).all()
    report = {"version": "tenfold.residual.remainder.contract.r20", "complete": True,
              "rows": rows, "whole_predict_width": n, "whole_predict_depth": 4,
              "whole_predict_finite": True, "whole_predict_flops": ctx.summary_dict()["flops_used"],
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Independent repeated birth CP tensor and three age transports, actual helper and small predict; no targets/no score/no suite acceptance"}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

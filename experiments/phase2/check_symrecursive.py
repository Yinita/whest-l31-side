"""Offline numerical and charged-FLOP checks for the C_pre triangle candidate.

This diagnostic uses NumPy synthetic inputs and never runs a competition network.
The production candidate contains only the existing flopscope operator surface.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured(module, x, y, levels):
    estimator = module.Estimator()
    estimator._pool = module._Pool(fnp.float32)
    estimator._smm = module._Strassen(fnp.float32, 2)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
        xx = fnp.array(x, dtype=fnp.float32)
        yy = fnp.array(y, dtype=fnp.float32)
        out = fnp.empty((len(x), len(x)), dtype=fnp.float32)
        estimator._sym_product(xx, yy, len(x), out, levels)
    return np.asarray(out).copy(), int(context.flops_used)


def main():
    output = Path('results/phase2/symrecursive_validation_20261012.json')
    if output.exists():
        raise FileExistsError(output)
    original_path = Path('estimators/k3_v29/estimator.py')
    candidate_path = Path('estimators/k3_v29_symrecursive/estimator.py')
    original = load(original_path, 'baseline_sym_check')
    candidate = load(candidate_path, 'candidate_sym_check')
    rng = np.random.default_rng(78413)
    rows = []
    for n in (5, 64, 128, 256, 1024):
        for case in range(2 if n < 1024 else 1):
            w = rng.normal(0, (2/n)**0.5, size=(n, n)).astype(np.float32)
            core = rng.normal(0, n**-0.5, size=(n, n)).astype(np.float32)
            c = (core @ core.T).astype(np.float32)
            x = w @ c
            y = w.T.copy()
            exact = x.astype(np.float64) @ y.astype(np.float64)
            # Float32 W C already differs from real W C, so X Y need not be
            # exactly symmetric. This is recorded, never silently asserted away.
            ideal = w.astype(np.float64) @ c.astype(np.float64) @ w.astype(np.float64).T
            levels = 4
            old, old_flops = measured(original, x, y, levels)
            new, new_flops = measured(candidate, x, y, levels)
            scale = np.max(np.abs(ideal))
            row = dict(width=n, case=case, levels=levels,
                       baseline_flops=old_flops, candidate_flops=new_flops,
                       flops_ratio=new_flops/old_flops,
                       baseline_relative_max_error=float(np.max(np.abs(old-ideal))/scale),
                       candidate_relative_max_error=float(np.max(np.abs(new-ideal))/scale),
                       candidate_vs_baseline_relative_max=float(np.max(np.abs(new-old))/scale),
                       candidate_vs_baseline_relative_frobenius=float(np.linalg.norm(new-old)/np.linalg.norm(old)),
                       rounded_XY_relative_asymmetry=float(np.max(np.abs(exact-exact.T))/scale),
                       input_C_max_asymmetry=float(np.max(np.abs(c-c.T))),
                       candidate_max_asymmetry=float(np.max(np.abs(new-new.T))),
                       finite=bool(np.isfinite(new).all()))
            assert row['finite']
            assert row['candidate_relative_max_error'] < 1e-4
            if n >= 128:
                assert new_flops < old_flops
            rows.append(row)
    report = dict(candidate=str(candidate_path),
                  candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                  frozen_baseline_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
                  formula='Cpre=W C W.T; recurse only diagonal output blocks, compute upper off-diagonal blocks once, copy lower transpose',
                  measurement='flopscope BudgetContext counts actual permitted arithmetic/copies; no symmetry tags added by candidate',
                  assumptions='Input C is explicitly symmetrized by unchanged V29 assembly. X is its already rounded float32 W C; X Y need not be exactly symmetric, so equivalence is real-arithmetic only.',
                  limitations=['Synthetic covariance-only diagnostic, not benchmark accuracy or resource success',
                               'Deeper symmetry copying changes float32 rounding inside original diagonal output blocks',
                               'More small copyto calls may increase residual time; fixed 0.4-second contract must be tested',
                               'No direct operator supports triangular output, so diagonal leaves are computed densely'],
                  checks=rows)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(rows[-1], indent=2))


if __name__ == '__main__':
    main()

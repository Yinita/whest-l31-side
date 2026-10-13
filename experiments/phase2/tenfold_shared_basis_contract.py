"""Small n8/32 gate for isolated r03 shared-HQ source projection.

Preserves all birth columns. Original A/P are formed with parent's float32
Strassen family; candidate uses float64 H/Q/F regrouping. No reference data.
"""

import hashlib
import importlib.util
import itertools
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


def main():
    output = Path("results/phase2/tenfold_shared_basis_contract_r01.json")
    assert not output.exists()
    parent_path = Path("estimators/tenfold_corelambda_li14_r02/estimator.py")
    candidate_path = Path("estimators/tenfold_corelambda_li14_r03/estimator.py")
    parent = load(parent_path, "tenfold_basis_parent")
    candidate = load(candidate_path, "tenfold_basis_candidate")
    rng = np.random.default_rng(673901)
    rows = []
    for n, prefix, strassen_level in itertools.product((8, 32), (0, 1), (0, 1)):
        rank1, rank2 = n//2, n//4
        start, stop = prefix, n-2
        qc, qu = [rng.normal(size=(n, r)).astype(np.float32) for r in (rank1, rank2)]
        # Stale areas deliberately nonzero; represented rows are only [start:stop].
        fa, fp = [rng.normal(size=(2, rank1, n)).astype(np.float32) for _ in range(2)]
        fa2, fp2 = [rng.normal(size=(2, rank2, n)).astype(np.float32) for _ in range(2)]
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            smm = parent._Strassen(fnp.float32, 8)
            smm.smin = 1
            formed = []
            for basis, factors in ((qu, fa2), (qu, fp2), (qc, fa), (qc, fp)):
                masked = basis.copy()
                masked[:start], masked[stop:] = 0, 0
                qf, ff = fnp.asarray(masked), fnp.asarray(factors)
                out = fnp.zeros((2, 1, n, n), dtype=fnp.float32)
                smm.mm(qf[None, None], ff[:, None], out, strassen_level)
                formed.append(np.asarray(out[:, 0]).copy())
        a_st = np.concatenate((formed[0], formed[2]), axis=0)
        p_st = np.concatenate((formed[1], formed[3]), axis=0)
        # Add a young dense source to ensure the uncompressed branch remains.
        a_st = np.concatenate((a_st, rng.normal(size=(1, n, n)).astype(np.float32)))
        p_st = np.concatenate((p_st, rng.normal(size=(1, n, n)).astype(np.float32)))
        k, thin, feedback = len(a_st), 3, 2
        z, ell = [rng.normal(size=(k, n, thin+2)).astype(np.float32) for _ in range(2)]
        vectors = [rng.normal(size=(k, n)).astype(np.float32) for _ in range(5)]
        f = rng.normal(size=(k, n, 2*feedback)).astype(np.float32)
        r1, r2 = [rng.normal(size=(k, n, feedback)).astype(np.float32) for _ in range(2)]
        u, v = [rng.normal(size=n) for _ in range(2)]
        raw = rng.normal(size=(n, n))
        cov = (raw+raw.T)/2
        np.fill_diagonal(cov, 0)
        h = cov*v[:, None]*v[None, :]
        numerator_values, contexts = [], []
        for use_shared in (False, True):
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
                hf, uf = fnp.asarray(h), fnp.asarray(u)
                if use_shared:
                    qcf, quf = fnp.asarray(qc).astype(fnp.float64), fnp.asarray(qu).astype(fnp.float64)
                    for basis in (qcf, quf):
                        fnp.copyto(basis[:start], 0.)
                        fnp.copyto(basis[stop:], 0.)
                    hqc, hqu = hf@qcf, hf@quf
                total = fnp.zeros((), dtype=fnp.float64)
                for j in range(k):
                    vals = [x.astype(np.float64) for x in (a_st[j], p_st[j], z[j], ell[j],
                            *(x[j] for x in vectors), z[j, :, thin+1],
                            f[j, :, :feedback], f[j, :, feedback:], r1[j], r2[j])]
                    kwargs = {}
                    if use_shared and j < 4:
                        basis_product, af, pf = (hqu, fa2[j], fp2[j]) if j < 2 else (hqc, fa[j-2], fp[j-2])
                        kwargs = {"ha": basis_product@fnp.asarray(af).astype(fnp.float64),
                                  "hp": basis_product@fnp.asarray(pf).astype(fnp.float64)}
                    total = total+candidate.source_projection(hf, *map(fnp.asarray, vals), uf, **kwargs)
            numerator_values.append(float(total))
            contexts.append(context.summary_dict()["flops_used"])
        absolute = abs(numerator_values[1]-numerator_values[0])
        relative = absolute/max(abs(numerator_values[0]), 1e-30)
        assert absolute <= 2e-5*max(abs(numerator_values[0]), 1)+1e-8, (n, prefix, strassen_level, relative)
        rows.append({"width": n, "source_columns_each": n, "source_slots": k,
                     "tier2_slots": 2, "tier1_slots": 2, "young_slots": 1,
                     "row_range": [start, stop], "parent_formation_strassen_levels": strassen_level,
                     "before_numerator": numerator_values[0], "after_numerator": numerator_values[1],
                     "absolute_difference": absolute, "relative_difference": relative,
                     "before_flops": contexts[0], "after_flops": contexts[1]})
    report = {"version": "tenfold.sharedhq.r01", "complete": True, "rows": rows,
              "parent": str(parent_path), "parent_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
              "candidate": str(candidate_path), "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Bounded same all-source scalar numerator with actual parent float32 Strassen formed legs and masked sharedHQ",
              "single_change": "H@(formedA/P) replaced by shared(H@maskedQc/QU)@FA/FP; source legs/all columns retained",
              "precision_gate": "absdelta<=2e-5*max(absbefore,1)+1e-8; no bit-identical claim",
              "no_adjusted_score": True, "limitations": ["No actual1024 cost/parity or RPC resource acceptance; root runs those serially"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Exact algebraic fusion of six source-cap products into three products."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from full_source_cp_mapping_contract import full_source_factors
from real_source_full_k211 import KEYS
from real_source_k211_contract import capped_sym_source, direct_entry


def capped_three_products(a, b, c, u, v):
    sa = fnp.sum(u[:, None]*a, axis=0)
    sb = fnp.sum(u[:, None]*b, axis=0)
    sc = fnp.sum(u[:, None]*c, axis=0)
    uf, vf = u[:, None], v[:, None]
    x = (b*(sa-2*uf*a)*vf)@(c*vf).T
    x = x+(c*(sb-2*uf*b)*vf)@(a*vf).T
    x = x+(a*(sc-2*uf*c)*vf)@(b*vf).T
    out = (x+x.T)/6
    fnp.fill_diagonal(out, 0)
    return out


def main():
    output = Path("results/phase2/ablation_v30_067_r17_three_products.json")
    assert not output.exists()
    provenance = Path("results/phase2/ablation_v30_067_r14_full_source_capture.json")
    metadata = json.loads(provenance.read_text())
    capture = Path(metadata["capture"])
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == metadata["capture_sha256"]
    data = np.load(capture)
    rows = []
    for n in (8, 32, 1024):
        vs = [data[k].copy() if k in ("w2", "s", "e", "c1", "c2", "ell", "r1", "r2")
              else data[k][:n].copy() for k in KEYS]
        mu, sigma, alpha = [data[k][:n].copy() for k in ("pk1v", "sigma", "alpha")]
        results, costs = [], []
        for fn in (capped_sym_source, capped_three_products):
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                cp = full_source_factors(*map(fnp.asarray, vs))
                fm, sd, al = map(fnp.asarray, (mu, sigma, alpha))
                gate = flops.stats.norm.cdf(al).astype(fnp.float32)
                density = flops.stats.norm.pdf(al).astype(fnp.float32)
                u = 2*sd*(density+al*gate)-2*fm*gate
                cap = fn(*cp, u, gate)
            results.append(np.asarray(cap).astype(np.float64))
            costs.append(ctx.summary_dict())
        left, right = results
        delta = float(np.max(np.abs(left-right)))
        relative = float(np.linalg.norm(left-right)/max(np.linalg.norm(left), 1e-30))
        assert relative < 2e-6
        assert np.max(np.abs(np.diag(right))) == 0
        assert np.max(np.abs(right-right.T)) == 0
        # Independent scalar oracle over frozen mapped float32 operands.
        af, bf, cf = [np.asarray(x).astype(np.float64) for x in cp]
        uf, vf = np.asarray(u).astype(np.float64), np.asarray(gate).astype(np.float64)
        entries = [(i, j) for i in range(n) for j in range(n)] if n <= 32 else [
            (i*61, (i*67+5)%n) for i in range(16)]
        expected = np.array([direct_entry(af, bf, cf, uf, vf, i, j) for i, j in entries])
        actual = np.array([right[i, j] for i, j in entries])
        oracle_error = float(np.max(np.abs(actual-expected)))
        assert oracle_error < 5e-5*max(float(np.max(np.abs(expected))), 1e-12)+1e-8
        rows.append({"width": n, "cp_rank": 3072, "before_products": 6, "after_products": 3,
                     "pair_max_abs_delta": delta, "pair_relative_error": relative,
                     "independent_oracle_max_abs_error": oracle_error,
                     "before_flops": costs[0]["flops_used"], "after_flops": costs[1]["flops_used"],
                     "relative_flops": costs[1]["flops_used"]/costs[0]["flops_used"]-1,
                     "before_legal_cost": costs[0], "after_legal_cost": costs[1],
                     "scope": "same frozen fullsource and mean, completecomponent; no dslice diagnostic cost included"})
    report = {"version": "v30.067.r17", "complete": True, "rows": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "capture_sha256": metadata["capture_sha256"],
              "before": "Three trace products plus three repeated-index subtraction products",
              "after": "Fuse repeated-index operands before same three trace products; symmetrize once",
              "single_change": "exact compute regrouping only; same data/factors/mean/Wick/float32",
              "no_adjusted_score": True,
              "limitations": ["One real source, one K211 orbit; no complete candidate/score/strictRPC",
                              "No allsource integrated precision or residual acceptance"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print([{k: r[k] for k in ("width", "before_flops", "after_flops", "pair_relative_error", "relative_flops")}
           for r in rows])


if __name__ == "__main__":
    main()

"""Exact K211 contraction using repeated P legs and rank-one feed structure.

Only the contraction order changes. Full source columns, centering, and dtype
stay frozen. No reference labels are loaded. This is a component cost gate.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import itertools
import json
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from full_source_cp_mapping_contract import full_source_factors
from real_source_full_k211 import KEYS
from real_source_k211_contract import direct_entry
from source_cap_three_products import capped_three_products


def capped_structured_source(a, p, z, ell, w2, s, e, c1, c2, y,
                             f1, f2, r1, r2, u, v):
    """Six width-by-width products instead of nine; no symmetry assumption."""
    x1, y1 = 3*a+f1@r1.T, a*w2[None, :]+f2@r2.T
    total = capped_three_products(x1, p, y1, u, v)
    m = p*s[None, :]+3*a*e[None, :]+z@ell.T
    uf, vf = u[:, None], v[:, None]
    sm, sp = fnp.sum(uf*m, axis=0), fnp.sum(uf*p, axis=0)
    pv = p*vf
    xm = (p*(sm-4*uf*m)*vf)@pv.T
    xp = (p*(sp-uf*p)*vf)@(m*vf).T
    total = total+(xm+xm.T+2*(xp+xp.T))/6

    # Feed CP's third leg is y times a row of ones. Collapse that dimension
    # exactly, retaining both directions of Q; X P^T need not be symmetric.
    x = a*c1[None, :]+p*c2[None, :]
    sx = fnp.sum(uf*x, axis=0)
    q = x@p.T
    symmetric_q = q+q.T
    h = fnp.sum(x*p, axis=1)
    t = p@sx+x@sp-2*u*h
    uy = u*y
    feed = t[:, None]*y[None, :]+y[:, None]*t[None, :]
    feed = feed+fnp.sum(uy)*symmetric_q
    feed = feed-2*uy[:, None]*symmetric_q-2*symmetric_q*uy[None, :]
    total = total+(feed*v[:, None]*v[None, :])/6
    total = (total+total.T)/2
    fnp.fill_diagonal(total, 0)
    return total


def independent_synthetic_contract():
    rng = np.random.default_rng(6720)
    rows = []
    for n in (4, 7, 12):
        a, p = [rng.normal(size=(n, n)).astype(np.float32) for _ in range(2)]
        z, ell = [rng.normal(size=(n, 3)).astype(np.float32) for _ in range(2)]
        w, s, e, c1, c2, y, u, v = [rng.normal(size=n).astype(np.float32) for _ in range(8)]
        f1, f2, r1, r2 = [rng.normal(size=(n, 2)).astype(np.float32) for _ in range(4)]
        vals = (a, p, z, ell, w, s, e, c1, c2, y, f1, f2, r1, r2)
        af, pf, zf, lf, wf, sf, ef, cf, df, yf, ff, gf, rf, tf = [
            t.astype(np.float64) for t in vals]
        blocks = [(3*af+ff@rf.T, pf, af*wf+gf@tf.T),
                  (pf*sf+3*af*ef+zf@lf.T, pf, pf),
                  (af*cf+pf*df, pf, np.broadcast_to(yf[:, None], (n, n)))]
        tensor = np.zeros((n, n, n))
        for x, b, c in blocks:
            raw = np.einsum("ir,jr,kr->ijk", x, b, c)
            tensor += sum(raw.transpose(perm) for perm in itertools.permutations(range(3)))/6
        expected = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                if j != k:
                    expected[j, k] = sum(float(u[i])*float(v[j])*float(v[k])*tensor[i, j, k]
                                         for i in range(n) if i != j and i != k)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            got = capped_structured_source(*map(fnp.asarray, vals), fnp.asarray(u), fnp.asarray(v))
        out = np.asarray(got).astype(np.float64)
        relative = float(np.linalg.norm(out-expected)/np.linalg.norm(expected))
        assert relative < 3e-6
        assert np.max(np.abs(out-out.T)) == 0 and np.max(np.abs(np.diag(out))) == 0
        rows.append({"width": n, "relative_error_vs_independent_dense_sym": relative,
                     "max_abs_error": float(np.max(np.abs(out-expected))),
                     "legal_cost": ctx.summary_dict()})
    return rows


def main():
    output = Path("results/phase2/ablation_v30_067_r21_structured_source.json")
    assert not output.exists()
    synthetic = independent_synthetic_contract()
    meta_path = Path("results/phase2/ablation_v30_067_r18_all_source_capture.json")
    meta = json.loads(meta_path.read_text())
    state_path = Path(meta["capture"])
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == meta["capture_sha256"]
    with np.load(state_path) as state:
        mu, sigma, alpha = [state[k].copy() for k in ("pk1v", "sigma", "alpha")]
    sources = []
    for path, sha in meta["source_files_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
        with np.load(path) as data:
            sources.append([data[k].copy() for k in KEYS])
    results, costs, records = [], [], []
    # Separate charged before/after runs. Offline comparison occurs after both
    # contexts close, so timing excludes dense/scalar diagnostic oracles.
    for method in ("before", "after"):
        start = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            fm, sd, al = map(fnp.asarray, (mu, sigma, alpha))
            gate = flops.stats.norm.cdf(al).astype(fnp.float32)
            density = flops.stats.norm.pdf(al).astype(fnp.float32)
            u = 2*sd*(density+al*gate)-2*fm*gate
            total = fnp.zeros((1024, 1024), dtype=fnp.float32)
            for index, vals in enumerate(sources):
                prior = ctx.summary_dict()["flops_used"]
                legal_vals = list(map(fnp.asarray, vals))
                if method == "before":
                    cp = full_source_factors(*legal_vals)
                    cap = capped_three_products(*cp, u, gate)
                else:
                    cap = capped_structured_source(*legal_vals, u, gate)
                total = total+cap
                records.append({"method": method, "source_index": index,
                                "flops_used": ctx.summary_dict()["flops_used"]-prior})
        costs.append({"method": method, "elapsed_wall_s": time.monotonic()-start,
                      "legal_cost": ctx.summary_dict()})
        results.append(np.asarray(total).astype(np.float64))
        print(method, costs[-1], flush=True)
    before, after = results
    relative = float(np.linalg.norm(after-before)/max(np.linalg.norm(before), 1e-30))
    assert relative < 3e-6
    assert np.max(np.abs(after-after.T)) == 0 and np.max(np.abs(np.diag(after))) == 0
    # Fixed indices, selected without references, against all original source
    # columns using an independent scalar six-permutation contraction.
    entries = [(i*61, (i*67+5)%1024) for i in range(16)]
    uf, vf = np.asarray(u).astype(np.float64), np.asarray(gate).astype(np.float64)
    oracle = np.zeros(len(entries))
    for vals in sources:
        a, p, z, ell, w, s, e, c1, c2, y, f1, f2, r1, r2 = [t.astype(np.float64) for t in vals]
        cp = (np.concatenate((3*a+f1@r1.T, p*s+3*a*e+z@ell.T, a*c1+p*c2), axis=1),
              np.concatenate((p, p, p), axis=1),
              np.concatenate((a*w+f2@r2.T, p, np.broadcast_to(y[:, None], p.shape)), axis=1))
        oracle += [direct_entry(*cp, uf, vf, j, k) for j, k in entries]
    got = np.array([after[j, k] for j, k in entries])
    scalar_error = float(np.max(np.abs(got-oracle)))
    assert scalar_error < 5e-5*max(float(np.max(np.abs(oracle))), 1e-12)+1e-8
    report = {"version": "v30.067.r21", "complete": True,
              "single_change": "Exact repeated-P and rank-one-feed contraction; no source truncation",
              "before": "Three products per rank3072 CP, nine width1024 products per source",
              "after": "B1 three products, repeated-P M two products, rank-one feed one product",
              "synthetic_contract": synthetic, "source_slots": len(sources), "source_columns_each": 1024,
              "before_after_costs": costs, "per_source_costs": records,
              "relative_output_delta": relative,
              "max_abs_output_delta": float(np.max(np.abs(before-after))),
              "independent_fixed16_max_abs_error": scalar_error,
              "relative_flops": costs[1]["legal_cost"]["flops_used"]/costs[0]["legal_cost"]["flops_used"]-1,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "capture_metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
              "dependency_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
                  "experiments/phase2/full_source_cp_mapping_contract.py",
                  "experiments/phase2/source_cap_three_products.py",
                  "experiments/phase2/real_source_k211_contract.py"]},
              "dataset_commit": meta["dataset_commit"], "network_id": meta["network_id"], "layer": 14,
              "versions": {k: __import__("importlib.metadata").metadata.version(k)
                           for k in ("numpy", "flopscope", "whestbench")},
              "no_adjusted_score": True, "not_resource_acceptance": True,
              "limitations": ["Only one K111-to-K211 orbit; other diagrams and proxy removal pending",
                              "Component timing excludes oracles but is not full predict/strictRPC acceptance"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({k: report[k] for k in ("relative_output_delta", "independent_fixed16_max_abs_error", "relative_flops")})


if __name__ == "__main__":
    main()

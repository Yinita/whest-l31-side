"""Full represented youngest source CP vs independent V29 dslice equations.

No reference labels or source selection are used. This tests the source mapping
and its one K211 orbit, not the remaining full nonlinear production diagrams.
"""

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
from real_source_k211_contract import capped_sym_source, direct_entry

KEYS = ("a", "p", "z", "ell", "w2", "s", "e", "c1", "c2", "y", "f1", "f2", "r1", "r2")


def independent_v29_slices(values):
    a, p, z, ell, w, s, e, c1, c2, y, f1, f2, r1, r2 = values
    xt, yt = f1@r1.T, f2@r2.T
    m = p*s+3*a*e+z@ell.T
    ap, pp = a*p, p*p
    d3 = 3*np.sum(a*a*p*w, axis=1)+np.sum(m*pp, axis=1)
    d21 = (2*ap*w)@a.T+(a*a*w)@p.T+pp@m.T/3+2*(m*p)@p.T/3
    lp = a*yt+xt*a*w/3+xt*yt/3
    la = xt*p*w/3+p*yt
    d3 += 3*np.sum(p*lp, axis=1)
    d21 += la@a.T+lp@p.T+(ap+xt*p/3)@yt.T+(ap*w+p*yt)@xt.T/3
    r = ap@c1+pp@c2
    x3 = a*c1+p*c2
    d3 += y*r
    d21 += np.outer(r, y)/3+(2*y[:, None]*x3)@p.T/3
    np.fill_diagonal(d21, 0)
    return d3, d21


def main():
    output = Path("results/phase2/ablation_v30_067_r15_real_full_source_k211.json")
    assert not output.exists()
    provenance = Path("results/phase2/ablation_v30_067_r14_full_source_capture.json")
    metadata = json.loads(provenance.read_text())
    capture = Path(metadata["capture"])
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == metadata["capture_sha256"]
    x = np.load(capture)
    rows = []
    for n in (8, 32, 1024):
        # All source columns are preserved. Only neuron rows are bounded for small contracts.
        values = [x[k].copy() if k in ("w2", "s", "e", "c1", "c2", "ell", "r1", "r2")
                  else x[k][:n].copy() for k in KEYS]
        mu, sigma, alpha = [x[k][:n].copy() for k in ("pk1v", "sigma", "alpha")]
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            cp = full_source_factors(*map(fnp.asarray, values))
            fm, sd, al = map(fnp.asarray, (mu, sigma, alpha))
            gate = flops.stats.norm.cdf(al).astype(fnp.float32)
            density = flops.stats.norm.pdf(al).astype(fnp.float32)
            weight = 2*sd*(density+al*gate)-2*fm*gate
            cap = capped_sym_source(*cp, weight, gate)
            a, b, c = cp
            d3 = fnp.sum(a*b*c, axis=1)
            d21 = ((a*b)@c.T+(a*c)@b.T+(b*c)@a.T)/3
            fnp.fill_diagonal(d21, 0)
        # Independent float64 V29 equations have separate B1, M, feedback, feed terms.
        exact3, exact21 = independent_v29_slices([v.astype(np.float64) for v in values])
        actual3, actual21 = np.asarray(d3).astype(np.float64), np.asarray(d21).astype(np.float64)
        e3 = float(np.linalg.norm(actual3-exact3)/max(np.linalg.norm(exact3), 1e-30))
        e21 = float(np.linalg.norm(actual21-exact21)/max(np.linalg.norm(exact21), 1e-30))
        # V29 feed equations use symmetry of X3 P^T, which is true at birth and
        # preserved by shared transport, up to numerical/compression error.
        av, pv = values[0].astype(np.float64), values[1].astype(np.float64)
        feed = (av*values[7]+pv*values[8])@pv.T
        feed_asymmetry = float(np.linalg.norm(feed-feed.T)/max(np.linalg.norm(feed), 1e-30))
        # Scalar capped-orbit oracle uses float64 mapped operands, not the charged outputs.
        a0, p0, z0, l0, w0, s0, e0, c10, c20, y0, f10, f20, r10, r20 = [v.astype(np.float64) for v in values]
        ca = np.concatenate((3*a0+f10@r10.T, p0*s0+3*a0*e0+z0@l0.T, a0*c10+p0*c20), axis=1)
        cb = np.concatenate((p0, p0, p0), axis=1)
        cc = np.concatenate((a0*w0+f20@r20.T, p0, np.broadcast_to(y0[:, None], p0.shape)), axis=1)
        got = np.asarray(cap).astype(np.float64)
        wf, gf = np.asarray(weight).astype(np.float64), np.asarray(gate).astype(np.float64)
        entries = [(i, j) for i in range(n) for j in range(n)] if n <= 32 else [
            (i*61, (i*67+5)%n) for i in range(16)]
        expected = np.array([direct_entry(ca, cb, cc, wf, gf, j, k) for j, k in entries])
        actual = np.array([got[j, k] for j, k in entries])
        error = float(np.max(np.abs(actual-expected)))
        scale = max(float(np.max(np.abs(expected))), 1e-12)
        assert error < 5e-5*scale+1e-8, (n, error, scale)
        assert np.max(np.abs(np.diag(got))) == 0
        assert np.max(np.abs(got-got.T)) < 2e-8
        rows.append({"width": n, "source_slots": 1, "source_columns": 1024,
                     "cp_rank": cp[0].shape[1], "D3_relative_error": e3, "D21_relative_error": e21,
                     "dslice_parity_passed": max(e3, e21) < 3e-6,
                     "feed_matrix_relative_asymmetry": feed_asymmetry,
                     "capped_max_abs_error": error, "capped_max_relative_error": error/scale,
                     "component_norm": float(np.linalg.norm(got)),
                     "projected_core_norm": float(np.linalg.norm(got)*6/(n+4)),
                     "verification": "full scalar capped oracle and full dslice equations" if n <= 32 else "16 fixed scalar capped entries, full D3/D21/diagonal/symmetry",
                     "largest_charged_array_elements": max(n*n, n*3072), "legal_cost": ctx.summary_dict()})
    report = {"version": "v30.067.r15", "complete": True, "rows": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "capture_sha256": metadata["capture_sha256"],
              "capture_provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
              "component_sources_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                  "experiments/phase2/full_source_cp_mapping_contract.py", "experiments/phase2/real_source_k211_contract.py")},
              "before": "Real B1 allcolumns; feedback/M/feed missing",
              "after": "Complete represented single young source with actual pk1v; independently measures V29 dslice parity",
              "scope": "Real fullsource mapping plus one K111-to-K211 orbit only; no full nonlinear production or candidate score",
              "limitations": ["Other source slots and compressed old-source reconstruction pending",
                              "Remaining K31/K211 production orbits, covariance centering, proxy subtraction pending",
                              "No strictRPC/fullcandidate resource pass or score"], "no_adjusted_score": True}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print([{k: row[k] for k in ("width", "D3_relative_error", "D21_relative_error", "capped_max_abs_error")}
           for row in rows])


if __name__ == "__main__":
    main()

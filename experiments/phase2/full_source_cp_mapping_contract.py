"""Small-object full lean source CP mapping, including feedback, M and feed.

Uses synthetic legal-state operands to establish the representation, not an
actual production-state score. Compare to independent six-permutation tensors.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from real_source_k211_contract import capped_sym_source


def full_source_factors(a, p, z, ell, w2, s, e, c1, c2, y, f1, f2, r1, r2):
    """Exact represented young source (no new approximation, no cubic input)."""
    xt, yt = f1@r1.T, f2@r2.T
    x1, y1 = 3*a+xt, a*w2[None, :]+yt
    m = p*s[None, :]+3*a*e[None, :]+z@ell.T
    x3 = a*c1[None, :]+p*c2[None, :]
    y3 = fnp.ones((1, p.shape[1]), dtype=fnp.float32)*y[:, None]
    return (fnp.concatenate((x1, m, x3), axis=1),
            fnp.concatenate((p, p, p), axis=1),
            fnp.concatenate((y1, p, y3), axis=1))


def main():
    output = Path("results/phase2/ablation_v30_067_r13_full_source_cp_mapping.json")
    assert not output.exists()
    rng = np.random.default_rng(6713)
    rows = []
    for n in (4, 7, 12):
        a, p = [rng.normal(size=(n, n)).astype(np.float32) for _ in range(2)]
        z, ell = [rng.normal(size=(n, 3)).astype(np.float32) for _ in range(2)]
        w2, s, e, c1, c2, y, u, v = [rng.normal(size=n).astype(np.float32) for _ in range(8)]
        f1, f2, r1, r2 = [rng.normal(size=(n, 2)).astype(np.float32) for _ in range(4)]
        operands = (a, p, z, ell, w2, s, e, c1, c2, y, f1, f2, r1, r2)
        # Independent float64 construction keeps the three different source blocks separate.
        af, pf, zf, lf, wf, sf, ef, c1f, c2f, yf, f1f, f2f, r1f, r2f = [
            x.astype(np.float64) for x in operands]
        m = pf*sf+3*af*ef+zf@lf.T
        x3 = af*c1f+pf*c2f
        blocks = [(3*af+f1f@r1f.T, pf, af*wf+f2f@r2f.T),
                  (m, pf, pf), (x3, pf, np.broadcast_to(yf[:, None], (n, n)))]
        dense = np.zeros((n, n, n))
        for x, b, c in blocks:
            raw = np.einsum("ir,jr,kr->ijk", x, b, c)
            dense += sum(raw.transpose(perm) for perm in itertools.permutations(range(3)))/6
        expected = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                if j != k:
                    expected[j, k] = sum(float(u[i])*float(v[j])*float(v[k])*dense[i, j, k]
                                         for i in range(n) if i != j and i != k)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            cp = full_source_factors(*map(fnp.asarray, operands))
            cap = capped_sym_source(*cp, fnp.asarray(u), fnp.asarray(v))
            # Full ordinary D3/D21 slices of this represented tensor.
            ca, cb, cc = cp
            d3 = fnp.sum(ca*cb*cc, axis=1)
            d21 = ((ca*cb)@cc.T+(ca*cc)@cb.T+(cb*cc)@ca.T)/3
            fnp.fill_diagonal(d21, 0)
        got = np.asarray(cap).astype(np.float64)
        relative = float(np.linalg.norm(got-expected)/np.linalg.norm(expected))
        pair = np.array([[dense[i, i, j] if i != j else 0 for j in range(n)] for i in range(n)])
        d3_error = float(np.max(np.abs(np.asarray(d3)-np.einsum("iii->i", dense))))
        d21_error = float(np.max(np.abs(np.asarray(d21)-pair)))
        assert relative < 2e-6
        assert np.allclose(np.asarray(d3), np.einsum("iii->i", dense), rtol=2e-6, atol=5e-6)
        assert np.allclose(np.asarray(d21), pair, rtol=2e-6, atol=5e-6)
        rows.append({"width": n, "equivalent_cp_rank": 3*n,
                     "capped_relative_error": relative, "D3_max_abs_error": d3_error,
                     "D21_max_abs_error": d21_error, "legal_cost": ctx.summary_dict()})
    output.write_text(json.dumps({"version": "v30.067.r13", "complete": True, "rows": rows,
                                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  "component_sha256": hashlib.sha256(Path("experiments/phase2/real_source_k211_contract.py").read_bytes()).hexdigest(),
                                  "before": "Real B1 only; feedback/M/feed omitted",
                                  "after": "Exact CP representation of B1+feedback, M block and K4 feed on synthetic small objects",
                                  "formula": "Sym(3A+Xt,P,A*w2+Yt)+Sym(P*s+3A*e+ZL^T,P,P)+Sym(A*c1+P*c2,P,y*ones^T)",
                                  "scope": "Complete represented single-source algebra, not full official production or actual-source score",
                                  "limitations": ["Actual fullstate capture and V29 per-source dslice parity pending",
                                                  "Old-source shared bases need reconstruction; all source slots not tested",
                                                  "Full K211 other orbits/K31/proxy removal remain unresolved"],
                                  "no_adjusted_score": True}, indent=2, allow_nan=False)+"\n")
    print([{k: r[k] for k in ("width", "capped_relative_error", "D3_max_abs_error", "D21_max_abs_error")}
           for r in rows])


if __name__ == "__main__":
    main()

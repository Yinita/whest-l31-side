"""Exact source capped-core projection without forming a capped matrix.

All candidate helpers use metered flopscope operations. The runnable contract
uses independent offline six-permutation tensors on bounded small objects.
No candidate, benchmark reference, shared state, or estimator is changed.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def sym_source_projection(a, b, c, ha, hb, hc, u):
    """<capped Sym(a,b,c), C> when H=diag(v) C diag(v), C symmetric zero-diag.

    H-times-leg operands are provided to share products between source blocks.
    The two repeated-index corrections have already combined under symmetric C.
    """
    uf = u[:, None]
    sa, sb, sc = [fnp.sum(uf*x, axis=0) for x in (a, b, c)]
    return (fnp.sum(b*(sa-2*uf*a)*hc)
            + fnp.sum(c*(sb-2*uf*b)*ha)
            + fnp.sum(a*(sc-2*uf*c)*hb))/3


def source_projection(h, a, p, z, ell, w2, s, e, c1, c2, y,
                      f1, f2, r1, r2, u, ha=None, hp=None):
    """Full represented B1+feedback/M/feed scalar numerator, all source columns.

    Optional ha/hp may use existing shared-basis factors: H(QF)=(HQ)F.
    This is an exact contraction-order change, not column/source truncation.
    """
    ha = h@a if ha is None else ha
    hp = h@p if hp is None else hp
    x1, y1 = 3*a+f1@r1.T, a*w2[None, :]+f2@r2.T
    hx1, hy1 = 3*ha+(h@f1)@r1.T, ha*w2[None, :]+(h@f2)@r2.T
    value = sym_source_projection(x1, p, y1, hx1, hp, hy1, u)
    m = p*s[None, :]+3*a*e[None, :]+z@ell.T
    hm = hp*s[None, :]+3*ha*e[None, :]+(h@z)@ell.T
    value = value+sym_source_projection(m, p, p, hm, hp, hp, u)
    x = a*c1[None, :]+p*c2[None, :]
    hx = ha*c1[None, :]+hp*c2[None, :]
    hy = h@y
    uf, yf = u[:, None], y[:, None]
    sx, sp = fnp.sum(uf*x, axis=0), fnp.sum(uf*p, axis=0)
    sy = fnp.sum(u*y)
    value = value+(fnp.sum(p*(sx-2*uf*x)*hy[:, None])
                 + fnp.sum(yf*(sp-2*uf*p)*hx)
                 + fnp.sum(x*(sy-2*uf*yf)*hp))/3
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        assert not args.output.exists(), "Preserve original evidence"
    rng = np.random.default_rng(673802)
    rows = []
    for n, dtype, shared in itertools.product((8, 32), (np.float32, np.float64), (False, True)):
        rank = min(5, n-1)
        basis = rng.normal(size=(n, rank)).astype(dtype)
        fa, fp = [rng.normal(size=(rank, n)).astype(dtype) for _ in range(2)]
        # Include a pruned tail and a saturated prefix. The shared Q supplied
        # to HQ must use exactly the row mask of represented A/P.
        basis[:1], basis[-2:] = 0, 0
        if shared:
            a, p = basis@fa, basis@fp
        else:
            a, p = [rng.normal(size=(n, n)).astype(dtype) for _ in range(2)]
            a[:1], p[:1], a[-2:], p[-2:] = 0, 0, 0, 0
        z, ell = [rng.normal(size=(n, 3)).astype(dtype) for _ in range(2)]
        w2, s, e, c1, c2, y, u, v = [rng.normal(size=n).astype(dtype) for _ in range(8)]
        f1, f2, r1, r2 = [rng.normal(size=(n, 2)).astype(dtype) for _ in range(4)]
        raw = rng.normal(size=(n, n)).astype(dtype)
        covariance = (raw+raw.T)/2
        np.fill_diagonal(covariance, 0)
        h = covariance*v[:, None]*v[None, :]
        # Independent high-precision block construction from frozen operands.
        af, pf, zf, lf, wf, sf, ef, cf, df, yf, ff, gf, rf, tf = [
            x.astype(np.float64) for x in (a, p, z, ell, w2, s, e, c1, c2, y, f1, f2, r1, r2)]
        blocks = [(3*af+ff@rf.T, pf, af*wf+gf@tf.T),
                  (pf*sf+3*af*ef+zf@lf.T, pf, pf),
                  (af*cf+pf*df, pf, np.broadcast_to(yf[:, None], p.shape))]
        tensor = np.zeros((n, n, n))
        for left, middle, right in blocks:
            unsym = np.einsum("ir,jr,kr->ijk", left, middle, right)
            tensor += sum(unsym.transpose(perm) for perm in itertools.permutations(range(3)))/6
        expected = sum(float(u[i])*float(v[j])*float(v[k])*tensor[i, j, k]*float(covariance[j, k])
                       for i, j, k in itertools.permutations(range(n), 3))
        operands = (h, a, p, z, ell, w2, s, e, c1, c2, y, f1, f2, r1, r2, u)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            legal = list(map(fnp.asarray, operands))
            if shared:
                hq = legal[0]@fnp.asarray(basis)
                actual = source_projection(*legal, ha=hq@fnp.asarray(fa), hp=hq@fnp.asarray(fp))
            else:
                actual = source_projection(*legal)
        error = abs(float(actual)-expected)
        tolerance = 3e-5 if dtype == np.float32 else 1e-10
        assert error < tolerance*max(abs(expected), 1)+1e-12, (n, error, expected)
        # A deliberately nonzero diagonal cannot be ignored in this helper:
        # the public interface explicitly requires zero-diagonal C.
        rows.append({"width": n, "dtype": np.dtype(dtype).name, "shared_basis": shared,
                     "source_columns": n, "row_mask": "prefix1 and tail2 zeros", "scalar_error": error,
                     "expected": expected, "actual": float(actual),
                     "flops": ctx.summary_dict()["flops_used"]})
    report = {"scope": "bounded independent source-projection contract; no score",
              "version": "tenfold.scalarprojection.r02", "complete": True, "rows": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "before": "Full source cap matrix then inner product with live post covariance",
              "after": "Exact scalar inner product with shared H@A/H@P, all source columns",
              "no_adjusted_score": True,
              "requirements": ["H uses symmetric zero-diagonal post covariance and both gate factors",
                               "shared HQ uses identical masked basis underlying current represented A/P",
                               "call before in-place old-source Wick scaling; u already combines pK211 and mean centering"],
              "limitations": ["Small synthetic float32/float64 only; actual1024 cost/parity pending",
                              "source orbit only; lower-support/tree/K4 and K31 have separate contracts"]}
    serialized = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output is not None:
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()

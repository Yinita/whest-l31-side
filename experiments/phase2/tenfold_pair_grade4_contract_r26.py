"""Complete fixed grade4 bivariate cumulant response, no targets or fitting."""
# ruff: noqa: E402
import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import leggauss

from experiments.phase2.audit_official_augmented_terms import coefficient, extract_generators


def table():
    with tempfile.TemporaryDirectory() as td:
        target = Path(td)/"src/mlp_kprop"
        target.mkdir(parents=True)
        for name in ("partitions.py", "kprop_harmonic.py"):
            shutil.copyfile(Path("benchmark/data/official_source_93d091a4")/name, target/name)
        ns, sources = extract_generators(Path(td))
        terms = ns["get_all_terms"](4, d_max=4, augment=True)
        rows = {part: [(v, float(coefficient(v, 1))) for p, v in terms
                       if p == part and sum(ns["vec_block_cost"](b) for b in v) == 4]
                for part in ((1,), (2,), (3,), (4,), (1, 1), (2, 1), (2, 2), (3, 1))}
    return rows, sources


def wicks(mu, sigma, pairs):
    alpha = mu/sigma
    phi = flops.stats.norm.pdf(alpha)
    cdf = flops.stats.norm.cdf(alpha)
    moments = {0: cdf, 1: mu*cdf+sigma*phi}
    for p in range(2, 5):
        moments[p] = mu*moments[p-1]+(p-1)*sigma*sigma*moments[p-2]
    hs = {0: fnp.ones(len(mu)), 1: alpha}
    for k in range(2, 8):
        hs[k] = alpha*hs[k-1]-(k-1)*hs[k-2]
    result = {}
    for k, p in pairs:
        if k <= p:
            result[(k, p)] = math.factorial(p)/math.factorial(p-k)*moments[p-k]
        else:
            q = k-p-1
            result[(k, p)] = math.factorial(p)*(-1)**q*hs[q]*phi/sigma**(k-p)
    return result


def charged_grade4(mu, sigma, inputs, terms):
    pairs = {(sum(v[axis] for v in vp), p) for part, entries in terms.items()
             for vp, _ in entries for axis, p in enumerate(part)}
    ws = wicks(mu, sigma, pairs)
    out = {}
    for part, entries in terms.items():
        shape = (len(mu),)*len(part)
        acc = fnp.zeros(shape)
        for vp, c in entries:
            value = fnp.ones(shape)*c
            for v in vp:
                nz = tuple(i for i, x in enumerate(v) if x)
                key = tuple(v[i] for i in nz)
                factor = inputs[key]
                if len(nz) == 1 and len(part) == 2:
                    factor = factor[:, None] if nz[0] == 0 else factor[None, :]
                value = value*factor
            for axis, p in enumerate(part):
                wick = ws[(sum(v[axis] for v in vp), p)]
                value = value*(wick[:, None] if axis == 0 else wick[None, :]) if len(part) == 2 else value*wick
            acc = acc+value
        if len(part) == 2:
            # official connected pK is used only on distinct coordinates.
            fnp.fill_diagonal(acc, 0)
        out[part] = acc
    return out


def independent_integral(mu, sigma, cumulants, powers):
    # Exhaustive exponential differential operator, independent of official
    # partitions: integrate full bivariate Hermite density and subtract the
    # disconnected product of the two graded marginal expectations.
    blocks = [(a, b) for a in range(5) for b in range(5-a)
              if 2 <= a+b <= 4 and not (a+b == 2 and (a == 0 or b == 0))]
    costs = {v: sum(v)-1-int(all(x % 2 == 0 for x in v)) for v in blocks}
    polynomials = {g: np.zeros((1, 1)) for g in range(5)}
    mono = []
    def visit(begin, chosen, cost):
        if chosen:
            mono.append((tuple(chosen), cost))
        for i in range(begin, len(blocks)):
            v = blocks[i]
            if cost+costs[v] <= 4:
                visit(i, chosen+[v], cost+costs[v])
    visit(0, [], 0)
    nodes, weights = leggauss(160)
    zz, dw = [], []
    for m, s in zip(mu, sigma, strict=True):
        lower = -m/s
        z = lower+(nodes+1)*(16-lower)/2
        zz.append(z)
        dw.append(weights*(16-lower)/2*np.exp(-z*z/2)/math.sqrt(2*math.pi))
    H = [[hermeval(z, [0.0]*k+[1.0]) for k in range(9)] for z in zz]
    # Numerical expectation of each density-polynomial term (unfactorized
    # two-dimensional integration); no analytic Wick coefficient here.
    mixed = np.zeros(5)
    marginal = [np.zeros(5), np.zeros(5)]
    f = [(mu[i]+sigma[i]*zz[i])**powers[i] for i in (0, 1)]
    mixed[0] = float(np.sum(np.outer(dw[0]*f[0], dw[1]*f[1])))
    for i in (0, 1):
        marginal[i][0] = float(np.sum(dw[i]*f[i]))
    for vp, cost in mono:
        c = 1.0
        for v, count in Counter(vp).items():
            c *= cumulants[v]**count/(math.factorial(count)*math.prod(math.factorial(x) for x in v)**count)
        degrees = tuple(sum(v[i] for v in vp) for i in (0, 1))
        c /= sigma[0]**degrees[0]*sigma[1]**degrees[1]
        mixed[cost] += c*float(np.sum(np.outer(dw[0]*f[0]*H[0][degrees[0]], dw[1]*f[1]*H[1][degrees[1]])))
        if degrees[1] == 0:
            marginal[0][cost] += c*float(np.sum(dw[0]*f[0]*H[0][degrees[0]]))
        if degrees[0] == 0:
            marginal[1][cost] += c*float(np.sum(dw[1]*f[1]*H[1][degrees[1]]))
    del polynomials
    return float(mixed[4]-sum(marginal[0][j]*marginal[1][4-j] for j in range(5)))


def main():
    output = Path("results/phase2/tenfold_pair_grade4_contract_r26.json")
    assert not output.exists()
    terms, sources = table()
    rows = []
    for case, (mu, sigma) in enumerate([([-0.3, 0.7], [0.8, 1.3]), ([0.0, 0.0], [1.0, 1.0]), ([-2.0, 1.0], [0.7, 0.9])]):
        mu, sigma = np.asarray(mu), np.asarray(sigma)
        cumulants = {(1, 1): .04, (3, 0): .02, (0, 3): -.015, (2, 1): .004,
                     (1, 2): -.003, (4, 0): .012, (0, 4): -.007, (3, 1): .002,
                     (1, 3): -.001, (2, 2): .003}
        inputs = {(3,): np.asarray([cumulants[3, 0], cumulants[0, 3]]),
                  (4,): np.asarray([cumulants[4, 0], cumulants[0, 4]])}
        for key in ((1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (2, 2)):
            inputs[key] = np.asarray([[0.0, cumulants[key]], [cumulants[key[::-1]], 0.0]])
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            out = charged_grade4(fnp.asarray(mu), fnp.asarray(sigma), {k: fnp.asarray(v) for k, v in inputs.items()}, terms)
            actual = {part: float(value[0, 1]) for part, value in out.items() if len(part) == 2}
        for part, value in actual.items():
            expected = independent_integral(mu, sigma, cumulants, part)
            error = abs(value-expected)
            assert error < 2e-12, (case, part, value, expected, error)
            rows.append({"case": case, "powers": part, "actual": value, "integral": expected, "abs_error": error})
    report = {"complete": True, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "official_commit": "93d091a4c26c042bfffa28f2e76a81bc0aba94bb", "official_sources": sources,
              "counts_labelled_terms": {str(k): len(v) for k, v in terms.items()},
              "rows": rows, "maximum_error": max(r["abs_error"] for r in rows),
              "toy_last_flops": ctx.summary_dict()["flops_used"],
              "scope": "Only cost4 grade of fixed kmax4,dmax4,augmentTrue diagram table with current cumulant<=4 inputs. All marginal and connected pair coefficients; not complete fullk4 or 3/4support. Independent160² Gaussian-Hermite density integration with graded disconnected subtraction. No MC/reference/target access. No score or 1024 resource claim."}
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(output), "maximum_error": report["maximum_error"], "counts": report["counts_labelled_terms"]}))


if __name__ == "__main__":
    main()

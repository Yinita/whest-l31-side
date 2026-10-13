"""Exact three-product fusion of all 28 retained lower-support K211 diagrams.

Retains float64, original possibly nonsymmetric inputs, and distinct indices.
Synthetic and real contracts load no benchmark references. Full-width cost
measurement is optional so it can be scheduled alongside other experiments.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.real_k211_matrix_production import contract_diagram, matrix_wicks, table

# Exact raw orbit table frozen by r30 from official get_all_terms_iso.
TERMS = (
    (((0, 1, 1), (1, 0, 1)), 1.),
    (((0, 1, 1), (1, 1, 0)), 1.),
    (((1, 0, 1), (1, 1, 0)), 1.),
    (((0, 1, 1), (1, 0, 2)), .5),
    (((0, 1, 1), (1, 2, 0)), .5),
    (((0, 1, 2), (1, 0, 1)), .5),
    (((0, 2, 1), (1, 1, 0)), .5),
    (((0, 1, 2), (1, 1, 0)), .5),
    (((0, 2, 1), (1, 0, 1)), .5),
    (((0, 1, 1), (2, 0, 1)), .5),
    (((0, 1, 1), (2, 1, 0)), .5),
    (((1, 0, 1), (1, 2, 0)), .5),
    (((1, 0, 2), (1, 1, 0)), .5),
    (((1, 0, 1), (2, 1, 0)), .5),
    (((1, 1, 0), (2, 0, 1)), .5),
    (((0, 1, 1), (0, 1, 1), (1, 0, 1)), .5),
    (((0, 1, 1), (0, 1, 1), (1, 1, 0)), .5),
    (((0, 2, 2), (1, 0, 1)), .25),
    (((0, 2, 2), (1, 1, 0)), .25),
    (((0, 1, 1), (1, 0, 1), (1, 0, 1)), .5),
    (((0, 1, 1), (1, 1, 0), (1, 1, 0)), .5),
    (((0, 1, 1), (2, 0, 2)), .25),
    (((0, 1, 1), (2, 2, 0)), .25),
    (((0, 1, 1), (1, 0, 1), (1, 1, 0)), 1.),
    (((1, 0, 1), (1, 0, 1), (1, 1, 0)), .5),
    (((1, 0, 1), (1, 1, 0), (1, 1, 0)), .5),
    (((1, 0, 1), (2, 2, 0)), .25),
    (((1, 1, 0), (2, 0, 2)), .25),
)
DOUBLE_LEG = frozenset((2, 11, 12, 13, 14, 23, 24, 25, 26, 27))


def capped_product(left, right):
    """Bilinear sum_i L_ij R_ik with i distinct j,k; diagonals retained."""
    out = left.T@right
    out = out-fnp.diag(left)[:, None]*right-left.T*fnp.diag(right)[None, :]
    fnp.fill_diagonal(out, 0)
    return out


def single_leg_diagram(part, coefficient, factors, ws, n):
    """O(n²) evaluation when exactly one i-j or i-k leg is present."""
    contracted = None
    side = None
    edge = fnp.ones((n, n), dtype=fnp.float64)
    for v in part:
        ids = tuple(i for i, x in enumerate(v) if x)
        if ids == (1, 2):
            edge = edge*factors[v]
        elif ids in ((0, 1), (0, 2)):
            assert side is None or side == ids[1]
            side = ids[1]
            contracted = factors[v] if contracted is None else contracted*factors[v]
        else:
            raise ValueError(v)
    u, uj, uk = [ws[(sum(v[i] for v in part), power)] for i, power in enumerate((2, 1, 1))]
    sums = fnp.sum(u[:, None]*contracted, axis=0)-u*fnp.diag(contracted)
    if side == 1:
        result = sums[:, None]-contracted.T*u[None, :]
    else:
        result = sums[None, :]-contracted*u[:, None]
    result = result*edge*uj[:, None]*uk[None, :]*coefficient
    fnp.fill_diagonal(result, 0)
    return result


def fused_lower_support(c, d, g, ws):
    """All28 raw diagrams, jointly symmetrized; exactly three n×n products.

    Gram0 handles base plus triangle; Gram1 fuses two oriented D21 legs;
    Gram2 groups D21-transpose, doubled covariance and K4(22) legs sharing
    output Wicks. Symmetrization is only at the output, just like r30.
    """
    n = len(c)
    factors = table({"C_off": c, "D21": d, "wk4m": g})
    total = fnp.zeros((n, n), dtype=fnp.float64)
    for index, (part, coefficient) in enumerate(TERMS):
        if index not in DOUBLE_LEG:
            total = total+single_leg_diagram(part, coefficient, factors, ws, n)
    w11, w21, w22, w32 = [ws[p] for p in ((1, 1), (2, 1), (2, 2), (3, 2))]
    gram0 = capped_product(w22[:, None]*c, c)
    total = total+gram0*w11[:, None]*w11[None, :]
    total = total+gram0*c*w21[:, None]*w21[None, :]
    gram1 = capped_product(w32[:, None]*d, c)
    total = total+(gram1+gram1.T)*w11[:, None]*w11[None, :]/2
    leg2 = w22[:, None]*d.T/2+w32[:, None]*(c*c/2+g/4)
    gram2 = capped_product(leg2, c)
    total = total+gram2*w21[:, None]*w11[None, :]
    total = total+gram2.T*w11[:, None]*w21[None, :]
    total = (total+total.T)/2
    fnp.fill_diagonal(total, 0)
    return total


def baseline(c, d, g, ws):
    n = len(c)
    factors = table({"C_off": c, "D21": d, "wk4m": g})
    total = fnp.zeros((n, n), dtype=fnp.float64)
    for part, coefficient in TERMS:
        total = total+contract_diagram(part, coefficient, factors, ws, n)
    return (total+total.T)/2


def compare(data, ws_np, label, scalar=True):
    costs, outputs = [], []
    for name, method in (("before", baseline), ("after", fused_lower_support)):
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            c, d, g = [fnp.asarray(data[k]).astype(fnp.float64) for k in ("C_off", "D21", "wk4m")]
            ws = {p: fnp.asarray(x).astype(fnp.float64) for p, x in ws_np.items()}
            result = method(c, d, g, ws)
        outputs.append(np.asarray(result).copy())
        costs.append({"method": name, "legal_cost": context.summary_dict()})
    before, after = outputs
    relative = float(np.linalg.norm(after-before)/max(np.linalg.norm(before), 1e-30))
    error = float(np.max(np.abs(after-before)))
    assert relative < 2e-12, (label, relative)
    n = len(after)
    scalar_error = None
    if scalar:
        factors = table(data)
        entries = list(itertools.product(range(n), repeat=2)) if n <= 32 else [
            (i*61, (i*67+5)%n) for i in range(16)]
        def expected(j, k):
            value = 0.
            if j != k:
                for part, coefficient in TERMS:
                    for i in range(n):
                        if i == j or i == k:
                            continue
                        axes = (i, j, k)
                        term = coefficient
                        for v in part:
                            term *= float(factors[v][tuple(axes[a] for a, support in enumerate(v) if support)])
                        for axis, power in enumerate((2, 1, 1)):
                            term *= float(ws_np[(sum(v[axis] for v in part), power)][axes[axis]])
                        value += term
            return value
        oracle = [(expected(j, k)+expected(k, j))/2 for j, k in entries]
        scalar_error = float(np.max(np.abs(np.array([after[j, k] for j, k in entries])-oracle)))
        assert scalar_error < 2e-11*max(float(np.max(np.abs(oracle))), 1)+1e-12
    return {"label": label, "width": n, "relative_output_delta": relative,
            "max_abs_output_delta": error, "independent_scalar_max_abs_error": scalar_error,
            "before_after_costs": costs,
            "relative_flops": costs[1]["legal_cost"]["flops_used"]/costs[0]["legal_cost"]["flops_used"]-1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/phase2/tenfold_k211_fused_r01.json"))
    parser.add_argument("--full-width", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    spec = Path("results/phase2/ablation_v30_067_r30_real_k211_matrix.json")
    reference_terms = json.loads(spec.read_text())["retained_raw_terms"]
    assert [(tuple(map(tuple, p)), c) for p, c in reference_terms] == list(TERMS)
    rows = []
    rng = np.random.default_rng(674001)
    for n in (4, 8):
        # Deliberately nonsymmetric and nonzero diagonal tests ensure no hidden
        # identity/zero-diagonal assumptions enter distributive fusion.
        data = {k: rng.normal(size=(n, n)) for k in ("C_off", "D21", "wk4m")}
        pairs = {(sum(v[axis] for v in part), pw) for part, _ in TERMS
                 for axis, pw in enumerate((2, 1, 1))}
        ws = {p: rng.normal(size=n) for p in pairs}
        rows.append(compare(data, ws, f"synthetic_arbitrary_n{n}"))
    captures = []
    for version in (24, 27):
        meta_path = Path(f"results/phase2/ablation_v30_067_r{version}_real_k31_capture.json")
        meta = json.loads(meta_path.read_text())
        capture = Path(meta["capture"])
        assert hashlib.sha256(capture.read_bytes()).hexdigest() == meta["capture_sha256"]
        captures.append({"metadata": str(meta_path), "sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                         "capture_sha256": meta["capture_sha256"]})
        with np.load(capture) as src:
            full = {k: src[k].copy() for k in ("mu", "sigma", "C_off", "D21", "wk4m")}
        for n in ((8, 32, 1024) if args.full_width else (8, 32)):
            data = {k: x[:n, :n] if x.ndim == 2 else x[:n] for k, x in full.items()}
            with flops.BudgetContext(flop_budget=2**41, quiet=True):
                ws = matrix_wicks(fnp.asarray(data["mu"]).astype(fnp.float64),
                                  fnp.asarray(data["sigma"]).astype(fnp.float64))
            rows.append(compare(data, {p: np.asarray(x).copy() for p, x in ws.items()},
                                f"real_network{meta['network_id']}_n{n}", scalar=n <= 32))
    output = {"version": "tenfold.k211.fused.r01", "complete": True, "rows": rows,
              "single_change": "Exact contraction grouping: 18 single-leg sums and 10 double-leg diagrams fused into 3 products; same 28 terms, float64, distinct indices, output symmetrization",
              "before_matrix_products": 28, "after_matrix_products": 3,
              "raw_terms": TERMS, "captures": captures,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "specification_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
              "no_adjusted_score": True, "not_resource_acceptance": True,
              "limitations": ["Only lower-support raw K211 component; source/K4 centering, CovCov and harmonic projection remain separate", "Full-width measurement only present when --full-width supplied; cost projection is not a measured score"]}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print({"output": str(args.output), "rows": [{k: v for k, v in r.items() if k != "before_after_costs"} for r in rows]}, flush=True)


if __name__ == "__main__":
    main()

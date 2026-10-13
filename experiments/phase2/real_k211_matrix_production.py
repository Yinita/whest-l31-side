"""K211 two-index input diagrams via matrix contractions, no cubic operands.

Retains official factored-keeps-term filter. Three-index K111/K211 orbits are
explicitly excluded, not replaced with a proxy. No full core or score claim.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.audit_official_augmented_terms import coefficient, extract_generators
from experiments.phase2.real_k31_production import legal_wicks


def matrix_wicks(mu, sigma):
    w = legal_wicks(mu, sigma)
    alpha = mu/sigma
    phi = flops.stats.norm.pdf(alpha).astype(fnp.float64)
    gate = w[(1, 1)]
    w.update({(1, 2): 2*(sigma*phi+mu*gate), (2, 2): 2*gate,
              (3, 2): 2*phi/sigma, (4, 2): -2*alpha*phi/(sigma*sigma)})
    return w


def contract_diagram(part, coef, factors, wicks, n):
    """Contract i, leave j,k, then remove i=j/k and j=k.

    Group all ij and ik factors into two matrices, jk into a multiplier.
    Unary factors are carried in Wick vectors; no i,j,k tensor is allocated.
    """
    left = fnp.ones((n, n), dtype=fnp.float64)
    right = fnp.ones((n, n), dtype=fnp.float64)
    edge = fnp.ones((n, n), dtype=fnp.float64)
    unary = [fnp.ones(n, dtype=fnp.float64) for _ in range(3)]
    for v in part:
        ids = tuple(i for i, x in enumerate(v) if x)
        operand = factors[v]
        if ids == (0, 1):
            left = left*operand
        elif ids == (0, 2):
            right = right*operand
        elif ids == (1, 2):
            edge = edge*operand
        elif len(ids) == 1:
            unary[ids[0]] = unary[ids[0]]*operand
        else:
            raise ValueError((v, ids))
    for axis, power in enumerate((2, 1, 1)):
        unary[axis] = unary[axis]*wicks[(sum(v[axis] for v in part), power)]
    # Weight only one contracted leg. Remaining unary factors are applied once
    # after contraction, including the repeated-index subtraction terms.
    weighted_left = left*unary[0][:, None]
    value = weighted_left.T@right
    value = value-fnp.diag(weighted_left)[:, None]*right
    value = value-weighted_left.T*fnp.diag(right)[None, :]
    value = value*edge*unary[1][:, None]*unary[2][None, :]*coef
    fnp.fill_diagonal(value, 0)
    return value


def table(data):
    c, d, g = data["C_off"], data["D21"], data["wk4m"]
    return {(0, 1, 1): c, (1, 0, 1): c, (1, 1, 0): c,
            (0, 1, 2): d.T, (0, 2, 1): d,
            (1, 0, 2): d.T, (2, 0, 1): d,
            (1, 2, 0): d.T, (2, 1, 0): d,
            (0, 2, 2): g, (2, 0, 2): g, (2, 2, 0): g}


def main():
    output = Path("results/phase2/ablation_v30_067_r30_real_k211_matrix.json")
    assert not output.exists()
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)/"src/mlp_kprop"
        directory.mkdir(parents=True)
        for name in ("partitions.py", "kprop_harmonic.py"):
            shutil.copyfile(Path("benchmark/data/official_source_93d091a4")/name, directory/name)
        ns, sources = extract_generators(Path(td))
        classes = ns["get_all_terms_iso"](3, d_max=4, augment=True)[(2, 1, 1)]
        retained, excluded = [], []
        for part, count in classes.items():
            if not ns["factored_keeps_term"](3, (2, 1, 1), part):
                excluded.append({"part": part, "reason": "official non-hypertree filter"})
                continue
            if any(sum(x > 0 for x in v) == 3 for v in part):
                excluded.append({"part": part, "reason": "separate three-index source/K4 orbit pending"})
                continue
            orbit = {tuple(sorted(tuple(v[i] for i in perm) for v in part))
                     for perm in ((0, 1, 2), (0, 2, 1))}
            assert len(orbit) == count
            retained.extend((p, float(coefficient(p, 1))) for p in sorted(orbit))
    rows = []
    captures = []
    for version in (24, 27):
        mp = Path(f"results/phase2/ablation_v30_067_r{version}_real_k31_capture.json")
        meta = json.loads(mp.read_text())
        cp = Path(meta["capture"])
        assert hashlib.sha256(cp.read_bytes()).hexdigest() == meta["capture_sha256"]
        captures.append({"metadata": str(mp), "sha256": hashlib.sha256(mp.read_bytes()).hexdigest()})
        with np.load(cp) as src:
            all_data = {k: src[k].copy() for k in ("mu", "sigma", "C_off", "D21", "wk4m")}
        for n in (8, 32, 1024):
            data = {k: x[:n, :n].copy() if x.ndim == 2 else x[:n].copy() for k, x in all_data.items()}
            factors = table(data)
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                fs = {v: fnp.asarray(x).astype(fnp.float64) for v, x in factors.items()}
                ws = matrix_wicks(fnp.asarray(data["mu"]).astype(fnp.float64),
                                  fnp.asarray(data["sigma"]).astype(fnp.float64))
                total = fnp.zeros((n, n), dtype=fnp.float64)
                for part, coef in retained:
                    total = total+contract_diagram(part, coef, fs, ws, n)
            actual = np.asarray(total)
            entries = list(itertools.product(range(n), repeat=2)) if n <= 32 else [
                (i*61, (i*67+5)%n) for i in range(16)]
            # Scalar per-index independent contraction, after legal timing ends.
            offline_w = {p: np.asarray(w).copy() for p, w in ws.items()}
            expected = []
            for j, k in entries:
                value = 0.0
                if j != k:
                    for part, coef in retained:
                        ids = [i for i in range(n) if i != j and i != k]
                        term = np.ones(len(ids), dtype=np.float64)*coef
                        axes = (np.asarray(ids), j, k)
                        for v in part:
                            term *= factors[v][tuple(axes[a] for a, support in enumerate(v) if support)]
                        for axis, power in enumerate((2, 1, 1)):
                            term *= offline_w[(sum(v[axis] for v in part), power)][axes[axis]]
                        value += float(np.sum(term))
                expected.append(value)
            got = np.array([actual[j, k] for j, k in entries])
            error = float(np.max(np.abs(got-expected)))
            assert error < 1e-10*max(float(np.max(np.abs(expected))), 1e-12)+1e-12
            assert np.max(np.abs(np.diag(actual))) == 0
            asym = float(np.linalg.norm(actual-actual.T)/max(np.linalg.norm(actual), 1e-30))
            # Official factor_k3 symmetrizes each output int_partition slice
            # before conversion. Captured float32 input C/wk4m need not be
            # bitwise symmetric, even when the scalar diagram sum is correct.
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as sym_ctx:
                symmetric = (total+total.T)/2
            assert np.max(np.abs(np.asarray(symmetric)-np.asarray(symmetric).T)) == 0
            rows.append({"network_id": meta["network_id"], "width": n,
                         "raw_terms": len(retained), "fixed_entries": len(entries),
                         "scalar_max_abs_error": error, "preprojection_relative_asymmetry": asym,
                         "symmetrization_cost": sym_ctx.summary_dict(),
                         "capped_power_component_norm": float(np.linalg.norm(actual)),
                         "legal_cost": ctx.summary_dict()})
            print({k:v for k,v in rows[-1].items() if k!='legal_cost'}, flush=True)
    output.write_text(json.dumps({"version": "v30.067.r30", "complete": True, "rows": rows,
                                 "before": "Full polynomial bounded slabs on synthetic/offline inputs",
                                 "after": "Actual two public network states, all official retained lower-support terms, matrix contractions only",
                                 "retained_raw_terms": retained, "excluded_classes": excluded,
                                 "capture_metadata_sha256": captures, "official_sources": sources,
                                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "no_adjusted_score": True, "not_resource_acceptance": True,
                                 "limitations": ["Three-index K111/K4 orbits explicitly separate, not included in this matrix component",
                                                 "Actual wk4m is inherited proxy; centered source/CovCov and complete proxy dedup pending",
                                                 "Scalar oracle reuses legally constructed Wick table; independent Wick contracts exist separately"]},
                                indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()

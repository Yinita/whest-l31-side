"""Contract one frozen real V29 B1 source through the K111-to-K211 orbit.

This is a target-free component diagnostic. It excludes the other source columns,
feedback/feed/M legs, all other diagram orbits and non-Gaussian output mean terms.
The formula follows factor_k3.py at pinned upstream commit 93d091a4, lines 471-521.
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


def capped_sym_source(a, b, c, repeated_weight, single_weight):
    """sum_{i distinct j,k} u_i v_j v_k Sym(A_i B_j C_k), j != k.

    Sym is the average over six permutations, matching upstream FactoredTensor.
    No charged cubic tensor is formed. The diagonal is excluded explicitly.
    """
    v = single_weight[:, None]
    weighted_a = fnp.sum(repeated_weight[:, None]*a, axis=0)
    weighted_b = fnp.sum(repeated_weight[:, None]*b, axis=0)
    weighted_c = fnp.sum(repeated_weight[:, None]*c, axis=0)
    av, bv, cv = a*v, b*v, c*v
    direct = (bv*weighted_a)@cv.T
    direct = direct+(cv*weighted_b)@av.T
    direct = direct+(av*weighted_c)@bv.T
    direct = (direct+direct.T)/6
    uv = (repeated_weight*single_weight)[:, None]
    # At i=j, Sym(T_jjk)=(A_j B_j C_k+A_j C_j B_k+B_j C_j A_k)/3.
    repeated = (uv*a*b)@cv.T
    repeated = repeated+(uv*a*c)@bv.T
    repeated = repeated+(uv*b*c)@av.T
    result = direct-(repeated+repeated.T)/3
    fnp.fill_diagonal(result, 0)
    return result


def direct_entry(a, b, c, u, v, j, k):
    """Independent scalar distinct-index oracle; never charged or submitted."""
    if j == k:
        return 0.0
    ids = np.array([i for i in range(len(a)) if i != j and i != k])
    total = np.zeros(len(ids), dtype=np.float64)
    factors = (a, b, c)
    for x, y, z in itertools.permutations(range(3)):
        total += np.sum(factors[x][ids]*factors[y][j]*factors[z][k], axis=1)/6
    return float(np.sum(u[ids]*total)*v[j]*v[k])


def main():
    output = Path("results/phase2/ablation_v30_067_r10_real_source_k211.json")
    assert not output.exists(), "Preserve previous evidence"
    capture = Path("benchmark/data/v29_real_source_v30_059_r05/network_0.npz")
    provenance = Path("results/phase2/ablation_v30_059_r05_layer14_real_source_columns.json")
    upstream = Path("benchmark/data/official_source_93d091a4/factor_k3.py")
    assert hashlib.sha256(upstream.read_bytes()).hexdigest() == (
        "e408aa1d8544e7d77b95dad5a548560098e7265845b7f713846ff8b165007918")
    source = np.load(capture)
    rows = []
    for n in (8, 32, 1024):
        # Prespecified neuron subset; all 16 previously frozen source columns retained.
        a, b, c = [source[name][:n].copy() for name in ("a", "p", "c")]
        sigma, alpha = [source[name][:n].copy() for name in ("sigma", "alpha")]
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            fa, fb, fc, sd, al = map(fnp.asarray, (a, b, c, sigma, alpha))
            gate = flops.stats.norm.cdf(al).astype(fnp.float32)
            density = flops.stats.norm.pdf(al).astype(fnp.float32)
            # Gaussian moment proxy is explicit. It is not the full V29 pk1v.
            mean = sd*(density+al*gate)
            power_weight = 2*mean  # E[(ReLU^2)'(X)]
            raw = capped_sym_source(fa, fb, fc, power_weight, gate)
            mean111 = capped_sym_source(fa, fb, fc, mean*gate, gate)
            centered = raw-2*mean111
            collapsed = capped_sym_source(fa, fb, fc, power_weight-2*mean*gate, gate)
            # Official K211 trace contribution coefficient: 6*2/(2*n+8).
            core = centered*(6/(n+4))
        af, bf, cf = [x.astype(np.float64) for x in (a, b, c)]
        gf, mf = np.asarray(gate).astype(np.float64), np.asarray(mean).astype(np.float64)
        got, rg = np.asarray(centered).astype(np.float64), np.asarray(raw).astype(np.float64)
        entries = list(itertools.product(range(n), repeat=2)) if n <= 32 else [
            (i*61, (i*67+5)%n) for i in range(16)]
        expected = np.array([direct_entry(af, bf, cf, 2*mf-2*mf*gf, gf, j, k)
                             for j, k in entries])
        actual = np.array([got[j, k] for j, k in entries])
        error = float(np.max(np.abs(actual-expected)))
        scale = max(float(np.max(np.abs(expected))), 1e-12)
        assert error < 5e-5*scale+2e-9, (n, error, scale)
        assert np.max(np.abs(np.diag(got))) == 0
        symmetry = float(np.max(np.abs(got-got.T)))
        assert symmetry < 2e-9
        collapse_error = float(np.max(np.abs(got-np.asarray(collapsed))))
        rows.append({"width": n, "source_columns": 16, "unsymmetrized_rank": 16,
                     "equivalent_explicit_sym_cp_rank": 96,
                     "verification": "full scalar distinct-index oracle" if n <= 32 else "16 frozen scalar entries and full diagonal/symmetry",
                     "max_abs_error": error, "max_error_over_max_expected": error/scale,
                     "raw_to_centered_relative_change": float(np.linalg.norm(got-rg)/max(np.linalg.norm(rg), 1e-30)),
                     "centered_norm": float(np.linalg.norm(got)),
                     "projected_core_norm": float(np.linalg.norm(np.asarray(core))),
                     "collapsed_weight_max_abs_delta": collapse_error,
                     "symmetry_max_abs_delta": symmetry,
                     "largest_charged_array_elements": n*n,
                     "legal_cost": ctx.summary_dict()})
    report = {"version": "v30.067.r10", "complete": True, "rows": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
              "capture_provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
              "upstream_commit": "93d091a4c26c042bfffa28f2e76a81bc0aba94bb",
              "upstream_url": "https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py#L471-L521",
              "upstream_file_sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
              "before": "Synthetic unsymmetrized CP centering only",
              "after": "Frozen real layer14 B1 first16 columns, Sym factor mapping, Wick orbit, distinct centering and harmonic coefficient",
              "scope": "Target-free real-source subset component; no complete production replacement or candidate score",
              "no_adjusted_score": True,
              "limitations": ["Other columns, sources, feedback/feed/M and other K211 orbits excluded",
                              "Gaussian output mean proxy used; full non-Gaussian pk1v still required",
                              "No proxy removal/doublecounting proof or full K31/K211 integration",
                              "Math/local cost only, no strict RPC or candidate resource pass"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print([{k: row[k] for k in ("width", "max_abs_error", "centered_norm")} for row in rows])
    print("width1024 charged FLOPs", rows[-1]["legal_cost"]["flops_used"])


if __name__ == "__main__":
    main()

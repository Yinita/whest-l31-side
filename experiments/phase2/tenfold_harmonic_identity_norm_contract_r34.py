"""Post identity metric normalization independently from full tensor trace.

Proves postD4=g, pre actual metric differs, and expected scale2 is pre-only.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

OUTPUT = Path("results/phase2/tenfold_harmonic_identity_norm_contract_r34.json")


def pair(a, b):
    return (
        np.einsum("ij,kl->ijkl", a, b)
        + np.einsum("ik,jl->ijkl", a, b)
        + np.einsum("il,jk->ijkl", a, b)
        + np.einsum("ij,kl->ijkl", b, a)
        + np.einsum("ik,jl->ijkl", b, a)
        + np.einsum("il,jk->ijkl", b, a)
    ) / 6


def main():
    assert not OUTPUT.exists()
    rng = np.random.default_rng(34101)
    rows = []
    for n in (4, 8, 16):
        raw = rng.normal(size=(n, n))
        g = (raw + raw.T) / 2
        h = pair(g, np.eye(n))
        trace = np.einsum("iijk->jk", h)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            tf = fnp.asarray(trace)
            recovered = 6 / (n + 4) * tf - 3 / ((n + 2) * (n + 4)) * fnp.sum(
                fnp.diag(tf)
            ) * fnp.eye(n, dtype=fnp.float64)
        err = float(np.max(abs(np.asarray(recovered) - g)))
        assert err < 1e-12
        assert np.max(abs(np.einsum("iiii->i", h) - np.diag(g))) < 1e-12
        w = rng.normal(size=(n, n)) * np.sqrt(2 / n)
        transported = np.einsum("ia,jb,kc,ld,abcd->ijkl", w, w, w, w, h, optimize=True)
        direct = pair(w @ g @ w.T, w @ w.T)
        transporterror = float(np.max(abs(direct - transported)))
        assert transporterror < 1e-11
        exactpreD4 = np.diag(w @ g @ w.T) * np.sum(w * w, axis=1)
        assert np.max(abs(np.einsum("iiii->i", direct) - exactpreD4)) < 1e-11
        wrongpost = float(np.max(abs(np.einsum("iiii->i", h) - 2 * np.diag(g))))
        assert wrongpost > 1e-4
        rows.append(
            {
                "width": n,
                "post_identity_trace_core_recovery_error": err,
                "post_D4_equals_gdiag": True,
                "actual_pre_fulltransport_error": transporterror,
                "actual_pre_D4_equals_transportedcore_times_metricdiag": True,
                "wrong_post_metric2_D4_error": wrongpost,
                "legal_projection_flops": ctx.summary_dict()["flops_used"],
            }
        )
    official = Path("benchmark/data/official_source_93d091a4/kprop_harmonic.py")
    assert "assert K_d.has_identity_metric()" in official.read_text()
    r = {
        "version": "tenfold.harmonic.identitynorm.r34.contract",
        "complete": True,
        "rows": rows,
        "official_commit": "93d091a4c26c042bfffa28f2e76a81bc0aba94bb",
        "official_source_sha256": hashlib.sha256(official.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conclusion": "Post proxy is S(G,I), D4=gdiag; init-scale metric2 belongs to next prelinear transport only",
        "no_reference_or_target_access": True,
        "no_adjusted_score": True,
    }
    OUTPUT.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()

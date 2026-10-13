"""Exact affine projection preserving D4 and diagonal trace jointly.

Proxy S(diag(g),I), arbitrary signed old CP4 E; only vector diagonal correction.
Proof contract, no candidate, score, or oracle-fitting.
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

OUTPUT = Path("results/phase2/tenfold_joint_diagonal_trace_projection_contract.json")


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
    rng = np.random.default_rng(34032)
    rows = []
    for n in (4, 8, 16):
        q = rng.normal(size=(n, n)) * 0.2
        c = rng.normal(size=n)
        old = np.einsum("ir,jr,kr,lr,r->ijkl", q, q, q, q, c, optimize=True)
        target4 = rng.normal(size=n) * 0.1
        raw = rng.normal(size=(n, n)) * 0.01
        k22 = (raw + raw.T) / 2
        np.fill_diagonal(k22, 0)
        rowsum = k22.sum(axis=1)
        e4 = np.einsum("iiii->i", old)
        et = np.einsum("iijj->j", old)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            qf, cf, k4f, rowf = map(fnp.asarray, (q, c, target4, rowsum))
            q2 = qf * qf
            old4 = fnp.sum(q2 * q2 * cf, axis=1)
            oldtrace = q2 @ (cf * fnp.sum(q2, axis=0))
            cross = rowf - (oldtrace - old4)
            g = 6 / (n - 2) * cross - 3 / ((n - 2) * (n - 1)) * fnp.sum(cross)
            newborn = k4f - g - old4
        gf, birth = np.asarray(g), np.asarray(newborn)
        full = old + pair(np.diag(gf), np.eye(n))
        for i in range(n):
            full[i, i, i, i] += birth[i]
        actual4 = np.einsum("iiii->i", full)
        actualtrace = np.einsum("iijj->j", full)
        error4 = float(np.max(abs(actual4 - target4)))
        errort = float(np.max(abs(actualtrace - target4 - rowsum)))
        assert max(error4, errort) < 1e-12
        # Independent solve for unknown g and birth vectors from both constraints.
        a = np.zeros((2 * n, 2 * n))
        a[:n, :n] = np.eye(n)
        a[:n, n:] = np.eye(n)
        a[n:, :n] = (n + 4) / 6 * np.eye(n) + np.ones((n, n)) / 6
        a[n:, n:] = np.eye(n)
        solved = np.linalg.solve(a, np.concatenate([target4 - e4, target4 + rowsum - et]))
        independent = float(np.max(abs(np.concatenate([gf, birth]) - solved)))
        assert independent < 1e-12
        parent = (6 / (n + 4)) * (target4 + rowsum) - 3 / ((n + 2) * (n + 4)) * np.sum(
            target4 + rowsum
        )
        wrong = old + pair(np.diag(parent), np.eye(n))
        for i in range(n):
            wrong[i, i, i, i] += target4[i] - parent[i] - e4[i]
        wrongtrace = float(np.max(abs(np.einsum("iijj->j", wrong) - target4 - rowsum)))
        assert wrongtrace > 1e-6
        rows.append(
            {
                "width": n,
                "D4_error": error4,
                "diagonaltrace_error": errort,
                "independent2n_linear_system_error": independent,
                "unadjusted_parentG_plus_D4_residual_trace_error": wrongtrace,
                "legal_projection_flops": ctx.summary_dict()["flops_used"],
            }
        )
    r = {
        "version": "tenfold.jointD4diagonalTrace.affineProjection.contract",
        "complete": True,
        "rows": rows,
        "formula": {
            "cross": "K22row - (trace(Eold)-D4(Eold))",
            "gdiag": "6/(n-2)*cross - 3/((n-2)*(n-1))*sum(cross)",
            "cnew": "K4v-gdiag-D4(Eold)",
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "ExactD4 and diagonaltrace affineprojection ofrepresentedoldE; offdiagcore requiresseparateconstraints",
        "no_reference_or_target_access": True,
        "no_adjusted_score": True,
        "limitations": [
            "Maintains K22row sums, not individualK22matrix orGaussian22exacttensor",
            "Fulloffdiagtrace and nonlinearfeed mapping notimplemented",
            "No productioncandidate orresource/scoreclaim",
        ],
    }
    OUTPUT.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()

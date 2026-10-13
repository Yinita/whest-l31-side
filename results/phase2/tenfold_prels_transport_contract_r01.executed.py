"""Independent bounded full-G/pre-covariance LS versus adjoint scalar path."""

import hashlib
import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from tenfold_shared_basis_contract import load


def main():
    output = Path("results/phase2/tenfold_prels_transport_contract_r01.json")
    assert not output.exists()
    path = Path("estimators/tenfold_precorelambda_alllayers_r07/estimator.py")
    candidate = load(path, "tenfold_prels_candidate")
    rng = np.random.default_rng(674701)
    rows = []
    for n in (8, 32):
        a, p = [rng.normal(size=(n, n))*.1 for _ in range(2)]
        z, ell = [rng.normal(size=(n, 3))*.1 for _ in range(2)]
        w2, s, e, c1, c2, y, u, v = [rng.normal(size=n)*.1 for _ in range(8)]
        f1, f2, r1, r2 = [rng.normal(size=(n, 2))*.1 for _ in range(4)]
        blocks = [(3*a+f1@r1.T, p, a*w2+f2@r2.T),
                  (p*s+3*a*e+z@ell.T, p, p),
                  (a*c1+p*c2, p, np.broadcast_to(y[:, None], (n, n)))]
        tensor = np.zeros((n, n, n))
        for left, middle, right in blocks:
            raw = np.einsum("ir,jr,kr->ijk", left, middle, right)
            tensor += sum(raw.transpose(perm) for perm in itertools.permutations(range(3)))/6
        source_cap = np.zeros((n, n))
        for i, j, k in itertools.permutations(range(n), 3):
            source_cap[j, k] += u[i]*v[j]*v[k]*tensor[i, j, k]
        raw = rng.normal(size=(n, n))*.001
        other_core_off = (raw+raw.T)/2
        np.fill_diagonal(other_core_off, 0)
        diagonal = rng.normal(size=n)*.001
        g_post = np.diag(diagonal)+other_core_off+source_cap*(6/(n+4))
        raw = rng.normal(size=(n, n))
        cov_post = raw@raw.T+np.eye(n)
        next_weight = rng.normal(size=(n, n)).astype(np.float32)*.2
        input_perm = rng.permutation(n)
        output_perm = rng.permutation(n)
        # Network storage is input-by-output; current state is input_perm.
        w = next_weight[input_perm].T.astype(np.float64)
        c_pre = w@cov_post@w.T
        np.fill_diagonal(c_pre, 0)
        g_pre = w@g_post@w.T
        expected = np.sum(g_pre*c_pre)/np.sum(c_pre*c_pre)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            stored = fnp.asarray(next_weight)
            nw = fnp.take(stored, fnp.asarray(input_perm), axis=0).T.astype(fnp.float64)
            cp = (nw@fnp.asarray(cov_post))@nw.T
            fnp.fill_diagonal(cp, 0.)
            target = (nw.T@cp)@nw
            target = (target+target.T)/2
            target_diag = fnp.diag(target).copy()
            fnp.fill_diagonal(target, 0.)
            source = candidate.source_projection(target*v[:, None]*v[None, :],
                *map(fnp.asarray, (a,p,z,ell,w2,s,e,c1,c2,y,f1,f2,r1,r2)), fnp.asarray(u))
            numerator = fnp.sum(fnp.asarray(other_core_off)*target)+source*(6/(n+4))
            numerator = numerator+fnp.sum(fnp.asarray(diagonal)*target_diag)
            actual = numerator/fnp.sum(cp*cp)
        error = abs(float(actual)-expected)
        assert error < 1e-10*max(abs(expected),1)+1e-12
        permuted_w = w[output_perm]
        pg, pc = permuted_w@g_post@permuted_w.T, permuted_w@cov_post@permuted_w.T
        np.fill_diagonal(pc, 0)
        permuted = np.sum(pg*pc)/np.sum(pc*pc)
        assert abs(permuted-expected) < 1e-12
        rows.append({"width": n, "lambda_fullG_direct": expected, "lambda_adjoint_scalar": float(actual),
                     "absolute_error": error, "future_output_perm_error": abs(permuted-expected),
                     "network_input_by_output_and_current_perm_checked": True,
                     "legal_flops": context.summary_dict()["flops_used"]})
    report = {"version": "tenfold.prels.transport.r01", "complete": True, "rows": rows,
              "candidate": str(path), "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "formula": "D=off(W*Cpost*W.T); H=W.T*D*W; lambda=(dot(gpostdiag,diag(H))+dot(Gpostoff,off(H)))/dot(D,D)",
              "scope": "Full dense G transport versus independent source tensor/adjoint scalar preLS, bounded8/32",
              "no_adjusted_score": True, "limitations": ["No actual1024/RPC proof; postG production has separate contracts",
                             "Candidate stores preLS scalar regeneration; does not preserve dense Gpre diagonal"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

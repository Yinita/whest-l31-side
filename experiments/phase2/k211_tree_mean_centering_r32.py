"""Official hypertree pK111 mean-centering component with an identity CP leg.

Two real public states; no reference. Does not include transported sources or
the separate K4 feed-to-K111 orbit, and does not form a complete K211 core.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.real_k31_production import legal_wicks


def identity_leg_cap(a, c, mean):
    """sum_i distinct j,k mean_i * Sym(A_i, 3I_j, C_k)."""
    sa = fnp.sum(mean[:, None]*a, axis=0)
    sc = fnp.sum(mean[:, None]*c, axis=0)
    middle = (c*mean[None, :])@a.T
    raw = c.T*sa[:, None]+middle+a*sc[None, :]
    repeated = (mean*fnp.diag(a))[:, None]*c.T
    repeated = repeated+mean[:, None]*a*c
    repeated = repeated+(mean*fnp.diag(c))[:, None]*a.T
    out = (raw+raw.T)/2-(repeated+repeated.T)
    fnp.fill_diagonal(out, 0)
    return out


def main():
    output = Path("results/phase2/ablation_v30_067_r32_tree_mean_centering.json")
    assert not output.exists()
    source = Path("benchmark/data/official_source_93d091a4/factor_k3.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "e408aa1d8544e7d77b95dad5a548560098e7265845b7f713846ff8b165007918"
    rows = []
    for version in (24, 27):
        mp = Path(f"results/phase2/ablation_v30_067_r{version}_real_k31_capture.json")
        meta = json.loads(mp.read_text())
        capture = Path(meta["capture"])
        assert hashlib.sha256(capture.read_bytes()).hexdigest() == meta["capture_sha256"]
        with np.load(capture) as src:
            full = {k: src[k].copy() for k in ("mu", "sigma", "C_off", "D21", "wk4m", "pk1v")}
        for n in (8, 32, 1024):
            data = {k: x[:n, :n] if x.ndim == 2 else x[:n] for k, x in full.items()}
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                c, d, g, mean = [fnp.asarray(data[k]).astype(fnp.float64)
                                 for k in ("C_off", "D21", "wk4m", "pk1v")]
                ws = legal_wicks(fnp.asarray(data["mu"]).astype(fnp.float64),
                                 fnp.asarray(data["sigma"]).astype(fnp.float64))
                w = lambda k: ws[(k, 1)]
                legs = [(1, 1, c, 1), (2, 1, d/2, 2), (1, 2, d.T/2, 2),
                        (2, 2, g/4, 2), (2, 2, c*c/2, 2)]
                total = fnp.zeros((n, n), dtype=fnp.float64)
                oracle_factors = []
                for al, bl, sl, cl in legs:
                    rights = [(ar, br, sr) for ar, br, sr, cr in legs if cl+cr <= 3]
                    if not rights:
                        continue
                    a = w(al)[:, None]*sl
                    cleg = sum(w(bl+ar)[:, None]*w(br)[None, :]*sr for ar, br, sr in rights).T
                    total = total+identity_leg_cap(a, cleg, mean)
                    oracle_factors.append((np.asarray(a).copy(), np.asarray(cleg).copy()))
                contribution = -2*total
            mf = data["pk1v"].astype(np.float64)
            entries = list(itertools.product(range(n), repeat=2)) if n <= 32 else [
                (i*61, (i*67+5)%n) for i in range(16)]
            expected = []
            for j, k in entries:
                val = 0.0
                if j != k:
                    ids = [i for i in range(n) if i != j and i != k]
                    # Exactly two of six permutations survive I in repeated
                    # contraction; accumulate explicit three-neuron sum.
                    for a, cf in oracle_factors:
                        for i in ids:
                            val += mf[i]*(a[i, j]*cf[k, j]+cf[i, j]*a[k, j]
                                          +a[i, k]*cf[j, k]+cf[i, k]*a[j, k]
                                          +a[j, i]*cf[k, i]+cf[j, i]*a[k, i])/2
                expected.append(-2*val)
            got = np.array([np.asarray(contribution)[j, k] for j, k in entries])
            error = float(np.max(np.abs(got-expected)))
            assert error < 1e-10*max(float(np.max(np.abs(expected))), 1e-12)+1e-12
            assert np.max(np.abs(np.asarray(contribution)-np.asarray(contribution).T)) == 0
            assert np.max(np.abs(np.diag(np.asarray(contribution)))) == 0
            rows.append({"network_id": meta["network_id"], "width": n,
                         "birth_factor_groups": len(oracle_factors),
                         "independent_scalar_max_abs_error": error,
                         "tree_mean_centering_component_norm": float(np.linalg.norm(np.asarray(contribution))),
                         "legal_cost": ctx.summary_dict()})
            print({k:v for k,v in rows[-1].items() if k!='legal_cost'}, flush=True)
    output.write_text(json.dumps({"version": "v30.067.r32", "complete": True, "rows": rows,
                                 "before": "Lower-support K211 minus CovCov, missing tree pK111 mean-centering",
                                 "after": "Complete official cost<=3 hypertree legs contracted with outputmean using exact 3I middle leg",
                                 "official_source": "https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py#L392-L417",
                                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "no_adjusted_score": True, "not_resource_acceptance": True,
                                 "limitations": ["K4feed and inputsource pK111 mean-centering are separate, not included",
                                                 "Captured fourth-order input legs inherit proxies; no complete core/score",
                                                 "Independent scalar uses same legal factors; diagram coefficients copied from pinned primary source"]},
                                indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()

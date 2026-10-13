"""Bounded full-helper lambda output parity r02/r03; no benchmark or labels."""

import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from tenfold_shared_basis_contract import load


def main():
    output = Path("results/phase2/tenfold_shared_lambda_contract_r03.json")
    assert not output.exists()
    paths = [Path(f"estimators/tenfold_corelambda_li14_r0{i}/estimator.py") for i in (2, 3)]
    modules = [load(p, f"shared_lambda_r0{i}") for i, p in zip((2, 3), paths)]
    rng = np.random.default_rng(673902)
    rows = []
    for n in (8, 32):
        rank1, rank2, k, ka, kb, thin, feedback = n//2, n//4, 5, 4, 2, 3, 2
        qc, qu = [rng.normal(size=(n, r)).astype(np.float32)*.1 for r in (rank1, rank2)]
        fa, fp = [rng.normal(size=(2, rank1, n)).astype(np.float32)*.1 for _ in range(2)]
        fa2, fp2 = [rng.normal(size=(2, rank2, n)).astype(np.float32)*.1 for _ in range(2)]
        masked = [x.copy() for x in (qc, qu)]
        for x in masked:
            x[-2:] = 0
        a = np.concatenate((masked[1]@fa2, masked[0]@fa,
                            rng.normal(size=(1, n, n)).astype(np.float32)*.01))
        p = np.concatenate((masked[1]@fp2, masked[0]@fp,
                            rng.normal(size=(1, n, n)).astype(np.float32)*.01))
        z, ell = [rng.normal(size=(k, n, thin+2)).astype(np.float32)*.01 for _ in range(2)]
        vecs = [rng.normal(size=(k, n)).astype(np.float32)*.01 for _ in range(5)]
        f = rng.normal(size=(k, n, 2*feedback)).astype(np.float32)*.01
        r1, r2 = [rng.normal(size=(k, n, feedback)).astype(np.float32)*.01 for _ in range(2)]
        def matrix():
            x = rng.normal(size=(n, n)).astype(np.float32)*.001
            np.fill_diagonal(x, 0)
            return x
        c, d, g, g31, pk21, cov = [matrix() for _ in range(6)]
        c, g, cov = [(x+x.T)/2 for x in (c, g, cov)]
        mu, sigma, d3, g4, mean, moment2 = [rng.uniform(.2, 1, n).astype(np.float32) for _ in range(6)]
        values, costs = [], []
        for i, module in enumerate(modules):
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                args = list(map(fnp.asarray, (mu, sigma, c, d3, d, g4, g, g31,
                                             mean, moment2, pk21, cov)))
                args += [.01, 2.]
                args += list(map(fnp.asarray, (a, p, z, ell)))
                args += [[fnp.asarray(row) for row in x] for x in vecs]
                args += list(map(fnp.asarray, (f, r1, r2)))+[thin, feedback]
                if i == 1:
                    args += list(map(fnp.asarray, (qc, qu, fa, fp, fa2, fp2)))+[ka, kb, 0, n-2]
                value = module.tenfold_fullcore(*args)
            values.append(float(value))
            costs.append(ctx.summary_dict()["flops_used"])
        absolute = abs(values[1]-values[0])
        assert absolute < 2e-5*max(abs(values[0]), 1)+1e-8
        rows.append({"width": n, "source_slots": k, "birth_columns": n,
                     "lambda_dtype": str(value.dtype), "before_lambda": values[0],
                     "after_lambda": values[1], "absolute_delta": absolute,
                     "before_flops": costs[0], "after_flops": costs[1]})
    report = {"version": "tenfold.sharedlambda.r03", "complete": True, "rows": rows,
              "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Bounded complete helper output parity including source tier indexing and float32 lambda",
              "no_adjusted_score": True, "limitations": ["Synthetic source state; no actual1024/RPC pass"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Independent tiny tensor contract for exact full-history heavy paths.

No whole 1024 benchmark or reference/target access. Parent owns strict runs.
"""
# ruff: noqa: E402
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import importlib.util
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
from experiments.phase2.audit_official_augmented_terms import extract_generators
from experiments.phase2.real_k31_production import legal_wicks

CANDIDATE = Path("estimators/tenfold_exact_hypertree_history_r14/estimator.py")
OUTPUT = Path("results/phase2/tenfold_exact_hypertree_history_contract_r14.json")

def sym(a, b, c):
    return sum(np.einsum("it,jt,kt->ijk", *x)
               for x in itertools.permutations((a, b, c))) / 6

def independent_paths(c, d, g, ws, official):
    legs = [(1, 1, c, 1), (2, 1, d / 2, 2),
            (1, 2, d.T / 2, 2), (2, 2, g / 4, 2), (2, 2, c * c / 2, 2)]
    tensor = np.zeros((len(c),) * 3)
    middle = 3 * np.eye(len(c))
    # Exactly the official loops; omit the base light-light path only.
    for al, bl, sl, cl in legs:
        for ar, br, sr, cr in legs:
            if cl + cr != 3:
                continue
            assert official["vec_block_cost"]((1, 1)) == 1
            left = ws[al][:, None] * sl
            right = (ws[bl + ar][:, None] * ws[br][None, :] * sr).T
            tensor += sym(left, middle, right)
    return tensor

def dslices(t):
    n = len(t)
    return np.array([t[i, i, i] for i in range(n)]), np.array(
        [[t[i, i, j] if i != j else 0 for j in range(n)] for i in range(n)])

def main():
    assert not OUTPUT.exists()
    spec = importlib.util.spec_from_file_location("exactheavy14", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "src/mlp_kprop"
        dest.mkdir(parents=True)
        for name in ("partitions.py", "kprop_harmonic.py"):
            shutil.copyfile(Path("benchmark/data/official_source_93d091a4") / name, dest / name)
        official, source_hashes = extract_generators(Path(td))
    rng = np.random.default_rng(673914)
    rows = []
    for n in (4, 7, 12, 32):
        c, d, g = [rng.normal(size=(n, n)) for _ in range(3)]
        c, g = (c + c.T) / 2, (g + g.T) / 2
        for value in (c, d, g):
            np.fill_diagonal(value, 0)
        ws = {i: rng.normal(size=n) for i in (1, 2, 3)}
        direct = independent_paths(c, d, g, ws, official)
        a = ws[1][:, None] * c
        h = (0.5 * ws[1][:, None] * d.T * ws[3][None, :]
             + 0.5 * ws[2][:, None] * d * ws[2][None, :]
             + 0.25 * ws[2][:, None] * g * ws[3][None, :]
             + 0.5 * ws[2][:, None] * c * c * ws[3][None, :])
        compact = 6 * sym(a, np.eye(n), h)
        compression_error = float(np.max(np.abs(compact - direct)))
        assert compression_error < 1e-10
        distinct = np.array([len(set(ids)) == 3 for ids in
                             itertools.product(range(n), repeat=3)]).reshape((n,) * 3)
        repeated_removed = direct * distinct
        m = -6 * (a * h).T
        complete = compact + sym(m, np.eye(n), np.eye(n))
        repeat_error = float(np.max(np.abs(complete - repeated_removed)))
        assert repeat_error < 1e-10
        # Actual candidate helpers, metered permitted operators.
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            pool = module._Pool(fnp.float64)
            smm = module._Strassen(fnp.float64, 4)
            stack = fnp.empty((1, 4, n, n), dtype=fnp.float64)
            module.exact_heavy_birth(*map(fnp.asarray, (c, d, g, ws[1], ws[2], ws[3])), stack[0])
            birth3, birth21 = module.exact_heavy_slices(stack, pool, smm, n, 0)
            assert np.max(np.abs(np.asarray(birth3))) < 1e-10
            assert np.max(np.abs(np.asarray(birth21))) < 1e-10
            dense = complete
            max_d3, max_d21 = 0.0, 0.0
            for age in range(3):
                w = rng.normal(size=(n, n)) / np.sqrt(n)
                gain = np.ones(n) if age == 0 else rng.uniform(0.1, 1.0, size=n)
                wd = w * gain[None, :]
                stack = fnp.matmul(fnp.asarray(wd)[None, None], stack)
                dense = np.einsum("ia,jb,kc,abc->ijk", wd, wd, wd, dense)
                got3, got21 = module.exact_heavy_slices(stack, pool, smm, n, 0)
                want3, want21 = dslices(dense)
                max_d3 = max(max_d3, float(np.max(np.abs(np.asarray(got3) - want3))))
                max_d21 = max(max_d21, float(np.max(np.abs(np.asarray(got21) - want21))))
                assert max_d3 < 1e-9 and max_d21 < 1e-9
        rows.append({"width": n, "diagram_compaction_max_error": compression_error,
                     "birth_distinct_max_error": repeat_error,
                     "three_transport_d3_max_error": max_d3,
                     "three_transport_d21_max_error": max_d21,
                     "legal_flops": ctx.summary_dict()["flops_used"]})
    # Real V29 pre-ReLU state with all original objects, restricted small width.
    meta = json.loads(Path("results/phase2/ablation_v30_067_r24_real_k31_capture.json").read_text())
    capture = Path(meta["capture"])
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == meta["capture_sha256"]
    real_rows = []
    with np.load(capture) as src:
        for n in (8, 32):
            c, d, g = [src[k][:n, :n].astype(np.float64) for k in ("C_off", "D21", "wk4m")]
            c, g = (c + c.T) / 2, (g + g.T) / 2
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                wlegal = legal_wicks(fnp.asarray(src["mu"][:n]).astype(fnp.float64),
                                    fnp.asarray(src["sigma"][:n]).astype(fnp.float64))
                ws = {k: np.asarray(wlegal[(k, 1)]).copy() for k in (1, 2, 3)}
                stack = fnp.empty((1, 4, n, n), dtype=fnp.float64)
                module.exact_heavy_birth(*map(fnp.asarray, (c, d, g, ws[1], ws[2], ws[3])), stack[0])
            wanted = independent_paths(c, d, g, ws, official)
            mask = np.array([len(set(ids)) == 3 for ids in itertools.product(range(n), repeat=3)]).reshape((n,) * 3)
            a, p, h, z = [np.asarray(stack[0, i]).copy() for i in range(4)]
            tensor = 6 * sym(a, p, h) + sym(z, p, p)
            error = float(np.max(np.abs(tensor - wanted * mask)))
            assert error < 1e-12
            real_rows.append({"network_id": meta["network_id"], "capture_sha256": meta["capture_sha256"],
                              "width": n, "real_distinct_max_error": error,
                              "legal_birth_flops": ctx.summary_dict()["flops_used"]})
    report = {"version": "tenfold.exacthypertree.history.r14", "complete": True,
              "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "official_generator_sources": source_hashes, "rows": rows, "real_rows": real_rows,
              "scope": "All cost3 heavy+light official tree paths, exact birth-distinct subtraction, three full-row transport ages; isolated small legal component",
              "no_reference_or_target_access": True, "no_adjusted_score": True,
              "limitations": ["No full-size whole-network cost/resource/RPC result", "Inherited parent covariance/K4 proxies and M residual rank16 remain", "Triangle and K111*edge non-hypertree orbits remain omitted"]}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()

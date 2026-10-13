"""Tiny whole-predict and FP32 legal-Strassen integration contracts, no target."""
# ruff: noqa: E402
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext
from local_engine import build_mlp

CANDIDATE = Path("estimators/tenfold_exact_hypertree_history_r14/estimator.py")
OUTPUT = Path("results/phase2/tenfold_exact_hypertree_history_integration_r14.json")

def sym(a, b, c):
    return sum(np.einsum("it,jt,kt->ijk", *x) for x in itertools.permutations((a,b,c)))/6

def main():
    assert not OUTPUT.exists()
    spec = importlib.util.spec_from_file_location("heavy14integration", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for n in (32, 64):
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            mlp = build_mlp(n, 4, 97314+n)
            estimator = module.Estimator()
            estimator.setup(SetupContext(width=n, depth=4, flop_budget=2**41, api_version="0.1", seed=0))
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            prediction = estimator.predict(mlp, 2**41)
        assert np.asarray(prediction).shape == (4,n)
        assert np.isfinite(np.asarray(prediction)).all()
        rows.append({"width": n, "depth": 4, "whole_predict_flops": ctx.summary_dict()["flops_used"], "all_finite": True})
    rng = np.random.default_rng(1431)
    n=64
    c,d,g=[rng.normal(size=(n,n)).astype(np.float32) for _ in range(3)]
    c,g=(c+c.T)/2,(g+g.T)/2
    for v in (c,d,g):
        np.fill_diagonal(v,0)
    ws=[rng.normal(size=n).astype(np.float32) for _ in range(3)]
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        stack=fnp.empty((1,4,n,n),dtype=fnp.float32)
        module.exact_heavy_birth(*map(fnp.asarray,(c,d,g,*ws)),stack[0])
        parts=[np.asarray(stack[0,i]).astype(np.float64) for i in range(4)]
        dense=6*sym(parts[0],parts[1],parts[2])+sym(parts[3],parts[1],parts[1])
        w=(rng.normal(size=(n,n))/np.sqrt(n)).astype(np.float32)
        smm=module._Strassen(fnp.float32,4)
        out=fnp.empty(stack.shape,dtype=fnp.float32)
        smm.mm(fnp.asarray(w)[None,None],fnp.reshape(stack,(4,1,n,n)),
               fnp.reshape(out,(4,1,n,n)),smm.level(n,n,n,1))
        got3,got21=module.exact_heavy_slices(out,module._Pool(fnp.float32),smm,n,1)
    expected=np.einsum("ia,jb,kc,abc->ijk",w.astype(np.float64),w.astype(np.float64),w.astype(np.float64),dense,optimize=True)
    d3=np.array([expected[i,i,i] for i in range(n)])
    d21=np.array([[expected[i,i,j] if i!=j else 0 for j in range(n)] for i in range(n)])
    err3=float(np.linalg.norm(np.asarray(got3)-d3)/np.linalg.norm(d3))
    err21=float(np.linalg.norm(np.asarray(got21)-d21)/np.linalg.norm(d21))
    assert err3<2e-5 and err21<2e-5
    report={"version":"tenfold.exacthypertree.history.r14.integration", "complete":True,
            "candidate_sha256":hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "whole_predict_rows":rows,"fp32_strassen_width":n,
            "fp32_strassen_d3_relative_error":err3,"fp32_strassen_d21_relative_error":err21,
            "fp32_component_legal_flops":ctx.summary_dict()["flops_used"],
            "no_reference_or_target_access":True,"no_adjusted_score":True,
            "limitations":["Whole predictions are smoke shape, not1024suite; inherited suite riders/pruning not tested here", "No strict1024 cost/resource/RPC score acceptance"]}
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))
if __name__=="__main__":
    main()

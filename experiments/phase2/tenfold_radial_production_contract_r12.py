"""Actual parent represented birth -> r12 sphere fulltensor/slice contract."""

# ruff: noqa: E402
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext


def sym(t):
    return sum(t.transpose(p) for p in itertools.permutations(range(3)))/6


def represented(state):
    a, r, ell, s, e, w2 = [state[k].astype(float) for k in ("a", "r", "ell", "s", "e", "w2")]
    identity = np.eye(len(s))
    m = np.diag(s)+3*a*e[None,:]+r@ell.T
    return (sym(np.einsum("ir,jr,kr->ijk", 3*a, a*w2[None,:], identity))
            +sym(np.einsum("ir,jr,kr->ijk", m, identity, identity)))


def run(path, weights):
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n, depth = weights.shape[1], len(weights)
    est = mod.Estimator()
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        est.setup(SetupContext(width=n, depth=depth, flop_budget=2**41, api_version="0.1", seed=0))
    lines = path.read_text().splitlines()
    row_line = next(i for i,line in enumerate(lines, 1) if "rows.append(fnp.take(mu," in line or "radial_row = mu*radial_a1" in line)
    pre_line = next(i for i,line in enumerate(lines, 1) if line.strip() == "sigma = fnp.sqrt(var)")
    captures = {}

    def trace(frame, event, arg):
        if event != "line" or frame.f_code.co_filename != str(path):
            return trace
        st = frame.f_locals
        if frame.f_lineno == row_line and st["li"] == 0:
            keys = {"a": st["a_b"], "r": st["Rr_full"], "ell": st["lb"][st["k_b"]],
                    "s": st["S3c"], "e": st["e_b"], "w2": st["w2b_list"][-1],
                    "mu": st["mu"], "cov": st["C"], "K4": st["K4v"], "K22": st["K22"]}
            keys.update({key:st[key] for key in ("pk1v","pk2v","pk3v","pk4v","pk11","pk21","pk22")})
            if "radial_B" in st:
                keys.update(B=st["radial_B"], v=st["radial_v"])
                captures["pair"], captures["cube"] = st["radial_pair"], st["radial_cube"]
            captures["birth"] = {k:np.asarray(v).copy() for k,v in keys.items()}
        if frame.f_lineno == pre_line and st["li"] == 1:
            captures["pre"] = {key:np.asarray(st[key]).copy() for key in ("D3","D21","W")}
        return trace

    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        mlp = SimpleNamespace(width=n, depth=depth, weights=[fnp.asarray(w) for w in weights])
        sys.settrace(trace)
        try:
            pred = np.asarray(est.predict(mlp, 2**41)).copy()
        finally:
            sys.settrace(None)
    assert captures.keys() >= {"birth", "pre"}
    assert np.isfinite(pred).all()
    return captures, ctx.summary_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    assert not out.exists()
    root = Path(__file__).resolve().parents[2]
    paths = [root/"estimators/sprint_public_v37b_h18_r01/estimator.py",
             root/"estimators/tenfold_radial_sphere_r12/estimator.py"]
    rows = []
    for n in (8,32):
        weights = (np.random.default_rng(261013920+n).normal(size=(3,n,n))*math.sqrt(2/n)).astype(np.float32)
        parent, parent_cost = run(paths[0], weights)
        new, new_cost = run(paths[1], weights)
        gb, sb = parent["birth"], new["birth"]
        tg, ts = represented(gb), represented(sb)
        a1 = math.exp(math.lgamma((n+1)/2)-math.lgamma(n/2)+0.5*(math.log(2)-math.log(n)))
        a3, a4 = a1*(1+1/n), 1+2/n
        m, c = gb["mu"].astype(float), gb["cov"].astype(float)
        pair, cube = 1/a3-1/a1, 1/a3-3/a1+2/a1**3
        extra = (pair*(np.einsum("ij,k->ijk",c,m)+np.einsum("ik,j->ijk",c,m)+np.einsum("jk,i->ijk",c,m))
                 +cube*np.einsum("i,j,k->ijk",m,m,m))
        expected = tg/a3+extra
        actual_extra = (new["pair"]*(np.einsum("ij,k->ijk",sb["B"],sb["v"])
                       +np.einsum("ik,j->ijk",sb["B"],sb["v"])+np.einsum("jk,i->ijk",sb["B"],sb["v"]))
                       +new["cube"]*np.einsum("i,j,k->ijk",sb["v"],sb["v"],sb["v"]))
        got = ts+actual_extra
        tensor_delta = float(np.max(np.abs(got-expected)))
        assert tensor_delta < 5e-7
        ms, r2, r3, r4 = gb["pk1v"]/a1, gb["pk2v"], gb["pk3v"]/a3, gb["pk4v"]/a4
        cs = c+(1-1/a1**2)*np.outer(m,m)
        ps11 = gb["pk11"]+(1-1/a1**2)*np.outer(m,m)
        np.fill_diagonal(ps11,0)
        ps21 = (gb["pk21"]+np.outer(r2,m))/a3-np.outer(r2,ms)
        ps22 = (gb["pk22"]+np.outer(r2,r2))/a4-np.outer(r2,r2)
        np.fill_diagonal(ps21,0)
        np.fill_diagonal(ps22,0)
        k4 = r4-4*ms*r3-3*r2*r2+12*ms*ms*r2-6*ms**4
        k22 = (ps22-2*ms[None,:]*ps21-2*ms[:,None]*ps21.T
               -2*ps11**2+4*np.outer(ms,ms)*ps11)
        np.fill_diagonal(k22,0)
        deltas = {"mu":float(np.max(np.abs(sb["mu"]-ms))),
                  "cov":float(np.max(np.abs(sb["cov"]-cs))),
                  "K4":float(np.max(np.abs(sb["K4"]-k4))),
                  "K22":float(np.max(np.abs(sb["K22"]-k22)))}
        assert max(deltas.values()) < 1e-5
        w = new["pre"]["W"].astype(float)
        transported = np.einsum("ia,jb,kc,abc->ijk",w,w,w,expected,optimize=True)
        d3, d21 = np.einsum("iii->i",transported), np.einsum("iij->ij",transported).copy()
        np.fill_diagonal(d21,0)
        slice_delta = {"D3":float(np.max(np.abs(new["pre"]["D3"]-d3))),
                       "D21":float(np.max(np.abs(new["pre"]["D21"]-d21)))}
        assert max(slice_delta.values()) < 3e-6, (n, slice_delta, float(np.max(abs(d3))), float(np.max(abs(d21))))
        rows.append({"width":n,"depth":3,"full_tensor_birth_maxdelta":tensor_delta,
                     "post_state_maxdeltas":deltas,"actual_secondpre_slices_maxdeltas":slice_delta,
                     "parent_flops":parent_cost["flops_used"],"candidate_flops":new_cost["flops_used"],
                     "scope":"Actual parent Gaussian birth, actual candidate scaled birth plus full radial channels; synthetic weights, represented K3 not true tensor"})
    report = {"complete":True,"version":"tenfold-radial-production-contract-r12","rows":rows,
              "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope":"Actual represented fulltensor and D3/D21 plus raw-moment sphere state conversion; trace instrumentation excludes resource acceptance",
              "limitations":["Parent lowrank repeated approximation and Gaussian-trained LAM/REF_R remain",
                             "Small synthetic weights; no actual1024 score or independent reference",
                             "Unpruned special channel, original pruned/confined parent sources remain approximate"],
              "no_adjusted_score":True}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()

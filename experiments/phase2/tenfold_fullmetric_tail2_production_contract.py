"""R16 actual estimator external feed birth-zero and final transport contract."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext


def cp(group):
    x=np.einsum("ir,jr,kr->ijk",*group)
    return sum(x.transpose(p) for p in itertools.permutations(range(3)))/6


def run(path,weights):
    spec=importlib.util.spec_from_file_location(path.parent.name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    n,depth=weights.shape[1],len(weights)
    est=mod.Estimator()
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        est.setup(SetupContext(width=n,depth=depth,flop_budget=2**41,api_version="0.1",seed=0))
    lines=path.read_text().splitlines()
    pre=next(i for i,l in enumerate(lines,1) if l.strip()=="sigma = fnp.sqrt(var)")
    birth=next(i for i,l in enumerate(lines,1) if l.strip()=="rep21 = _zero_diag(rep21)")
    captures={}
    def trace(frame,event,arg):
        if event!="line" or frame.f_code.co_filename!=str(path):return trace
        s=frame.f_locals
        if frame.f_lineno==pre and s["li"] in (depth-2,depth-1):
            fields={k:np.asarray(s[k]).copy() for k in ("D3","W","g4row")}
            if s["li"]==depth-2:
                fields.update({k:np.asarray(s[k]).copy() for k in ("wk431","wk4m","fullmetric_A","fullmetric_B")})
            else:
                fields["actual_external_D3"]=np.asarray(s["extra_d3"]).copy()
            captures[f'pre{s["li"]}']=fields
        if frame.f_lineno==birth and s["li"]==depth-2:
            captures["groups"]=[tuple(np.asarray(x).copy() for x in g) for g in s["fullmetric_groups"]]
            captures["gain"]={k:np.asarray(s[k]).copy() for k in ("w1","w2","correction_m")}
        return trace
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        mlp=SimpleNamespace(width=n,depth=depth,weights=[fnp.asarray(w) for w in weights])
        sys.settrace(trace)
        try:pred=np.asarray(est.predict(mlp,2**41)).copy()
        finally:sys.settrace(None)
    assert np.isfinite(pred).all()
    return captures,ctx.summary_dict()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    args=p.parse_args();assert not args.output.exists()
    root=Path(__file__).resolve().parents[2]
    path=root/"estimators/tenfold_fullmetric_tail2_r16/estimator.py"
    rows=[]
    for n in (8,32):
        weights=(np.random.default_rng(261013960+n).normal(size=(3,n,n))*math.sqrt(2/n)).astype(np.float32)
        c,cost=run(path,weights);a,b=c["pre1"]["fullmetric_A"].astype(float),c["pre1"]["fullmetric_B"].astype(float)
        x=np.einsum("ij,kl->ijkl",a,b)
        h=sum(x.transpose(p) for p in itertools.permutations(range(4)))/24
        d31,d22=np.einsum("iiij->ij",h).copy(),np.einsum("iijj->ij",h).copy()
        np.fill_diagonal(d31,0);np.fill_diagonal(d22,0)
        use_err={"D4":float(np.max(abs(c["pre1"]["g4row"]-np.einsum("iiii->i",h)))),
                 "D31":float(np.max(abs(c["pre1"]["wk431"]-d31))),
                 "D22":float(np.max(abs(c["pre1"]["wk4m"]-d22)))}
        feed=sum(cp(tuple(x.astype(float) for x in group)) for group in c["groups"])
        born_err={"D3":float(np.max(abs(np.einsum("iii->i",feed)))),
                  "D21":float(np.max(abs(np.einsum("iij->ij",feed))))}
        w=c["pre2"]["W"].astype(float)
        moved=np.einsum("ia,jb,kc,abc->ijk",w,w,w,feed,optimize=True)
        # Independently re-evaluate external groups after raw W; the actual
        # final D3 includes parent sources, so capture the addition separately.
        actual=c["pre2"]["actual_external_D3"]
        transport_error=float(np.max(abs(actual-np.einsum("iii->i",moved))))
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as fusedctx:
            fa,fb,fw,fm,fw1,fw2=[fnp.asarray(v) for v in (
                c["pre1"]["fullmetric_A"],c["pre1"]["fullmetric_B"],c["pre2"]["W"],
                c["gain"]["correction_m"],c["gain"]["w1"],c["gain"]["w2"])]
            v=fw*fw1[None,:]
            xa,xb,xm=v@fa,v@fb,fw@fm
            fused=(fw@(fw2*fnp.diag(fa)))*fnp.sum(v*xb,axis=1)*0.25
            fused=fused+fnp.sum(xa*(fw*fw2[None,:])*xb,axis=1)
            fused=fused+(fw@(fw2*fnp.diag(fb)))*fnp.sum(v*xa,axis=1)*0.25
            fused=fused-fnp.sum(xm*(fw*fw),axis=1)
        fused_error=float(np.max(abs(np.asarray(fused)-np.einsum("iii->i",moved))))
        assert max([*use_err.values(),*born_err.values(),transport_error])<3e-6
        assert fused_error<3e-6
        rows.append({"width":n,"depth":3,"USE_fullmetric_maxdeltas":use_err,
                     "external_feed_birth_repeated_maxabs":born_err,
                     "rawW_transport_D3_maxdelta":transport_error,"legal_component_estimator_cost":cost,
                     "fused_threeGEMM_D3_maxdelta":fused_error,"fused_legal_component_cost":fusedctx.summary_dict(),
                     "fullmetric_external_feed_tensor_norm":float(np.linalg.norm(feed))})
    report={"version":"tenfold-fullmetric-tail2-production-contract-r03","complete":True,"rows":rows,
            "candidate_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope":"Actual generic path estimator n8/32; captured USE objects and exact-zero repeated external fullfeed; no score/reference/real1024",
            "limitations":["Parent lowrank residual, K4 birth/projection proxy frozen",
                           "Trace instrumentation not resource acceptance",
                           "Only penultimate metric replacement and one external feed transport"]}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"candidate_sha256":report["candidate_sha256"],"rows":[{k:v for k,v in r.items() if k!="legal_component_estimator_cost"} for r in rows]},indent=2))


if __name__=="__main__":main()

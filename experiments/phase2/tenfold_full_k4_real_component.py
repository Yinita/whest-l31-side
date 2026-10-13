"""Real 1024 state component: actual metric and fixed nonoracle QR compression.

Uses frozen internal cumulant capture and public weights, never references.
Not an estimator, actual transported full K4 history, or score measurement.
"""
import os

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.tenfold_full_k4_component_contract import (
    slices, weighted_distinct_211, weighted_distinct_121, direct_feed_slices_and_center)
from experiments.phase2.real_k211_matrix_production_r29 import matrix_wicks


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_and_feed(a,b,ws,mean):
    w1,w2,w12,w22=[ws[x] for x in ((1,1),(2,1),(1,2),(2,2))]
    cap=weighted_distinct_211(a,b,w22)
    near=weighted_distinct_121(a,b,w12)
    raw=(cap*w1[:,None]*w1[None,:]+near*w2[:,None]*w1[None,:]
         +near.T*w1[:,None]*w2[None,:])/2
    feed=direct_feed_slices_and_center(a,b,w1,w2,mean)
    return [*slices(a,b),raw,*feed]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args();assert not args.output.exists()
    meta_path=Path("results/phase2/ablation_v30_067_r24_real_k31_capture.json")
    meta=json.loads(meta_path.read_text());capture=Path(meta["capture"])
    assert digest(capture)==meta["capture_sha256"]
    with np.load(capture) as f:
        state={k:f[k].copy().astype(np.float64) for k in ("mu","sigma","wk4m","wk431","pk1v")}
    entries=json.loads(Path("benchmark/data/rpc_inputs/inputs.json").read_text())
    entry=entries[0];weight_path=Path(entry["weights"])
    with np.load(weight_path) as f:
        weight=f["weights"][14].astype(np.float64)
    n=len(state["mu"]);metric_c=2.
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        c=fnp.asarray(state["wk4m"])
        dg_sum=3*fnp.sum(c)/(metric_c*(n-1))
        dg=(6*fnp.sum(c,axis=1)/metric_c-dg_sum)/(n-2)
        a=fnp.asarray(state["wk431"])*(2/metric_c)
        fnp.fill_diagonal(a,dg)
        a=(a+a.T)/2
        fw=fnp.asarray(weight)
        b=fw.T@fw
        ws=matrix_wicks(fnp.asarray(state["mu"]),fnp.asarray(state["sigma"]))
        mean=fnp.asarray(state["pk1v"])
    construction_cost=ctx.summary_dict()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        full=raw_and_feed(a,b,ws,mean)
    full_cost=ctx.summary_dict()
    exact=[np.asarray(x).copy() for x in full]
    # Offline independent literal sum of H4 entries; not benchmark targets.
    an,bn=np.asarray(a),np.asarray(b)
    def h(i,j,k,l):
        return (an[i,j]*bn[k,l]+an[i,k]*bn[j,l]+an[i,l]*bn[j,k]
                +bn[i,j]*an[k,l]+bn[i,k]*an[j,l]+bn[i,l]*an[j,k])/6
    wn={k:np.asarray(v) for k,v in ws.items()};m=state["pk1v"]
    raw_err,feed_err=[],[]
    for t in range(12):
        j,k=(t*61+3)%n,(t*67+9)%n
        raw,center=0.,0.
        for i in range(n):
            if i in (j,k):continue
            raw+=(h(i,i,j,k)*wn[(2,2)][i]*wn[(1,1)][j]*wn[(1,1)][k]
                  +h(i,j,j,k)*wn[(1,2)][i]*wn[(2,1)][j]*wn[(1,1)][k]
                  +h(i,j,k,k)*wn[(1,2)][i]*wn[(1,1)][j]*wn[(2,1)][k])/2
            feed=.5*(wn[(2,1)][i]*wn[(1,1)][j]*wn[(1,1)][k]*h(i,i,j,k)
                     +wn[(1,1)][i]*wn[(2,1)][j]*wn[(1,1)][k]*h(i,j,j,k)
                     +wn[(1,1)][i]*wn[(1,1)][j]*wn[(2,1)][k]*h(i,j,k,k))
            center-=2*m[i]*feed
        raw_err.append(abs(exact[3][j,k]-raw));feed_err.append(abs(exact[6][j,k]-center))
    assert max(raw_err+feed_err)<1e-10, {"raw":max(raw_err),"feed":max(feed_err),"A_asym":float(np.linalg.norm(an-an.T)),"B_asym":float(np.linalg.norm(bn-bn.T)),"A_dtype":str(an.dtype),"B_dtype":str(bn.dtype)}
    omega_np=np.random.default_rng(261013952).normal(size=(n,32))
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        q,_=fnp.linalg.qr(b@fnp.asarray(omega_np))
        bq=b@q
        small=q.T@bq
        low=(q@small)@q.T
        diag_correction=fnp.diag(b)-fnp.diag(low)
        approx=low+fnp.diagflat(diag_correction)
    compression_cost=ctx.summary_dict()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        compressed=raw_and_feed(a,approx,ws,mean)
    compressed_cost=ctx.summary_dict()
    labels=("D4","D31","D22","raw_K4power_211cap","birth_feed_D3","birth_feed_D21","feed_mean_center_211cap")
    errors={key:{"relative_Frobenius":float(np.linalg.norm(np.asarray(v)-u)/max(np.linalg.norm(u),1e-30)),
                 "max_absolute":float(np.max(abs(np.asarray(v)-u)))} for key,u,v in zip(labels,exact,compressed)}
    report={"version":"tenfold-full-k4-real-component-r01","complete":True,"width":n,"layer":14,
            "network_id":meta["network_id"],"network_name":entry["name"],"dataset_commit":meta["dataset_commit"],
            "capture":str(capture),"capture_sha256":digest(capture),"weights":str(weight_path),"weights_sha256":digest(weight_path),
            "source_sha256":digest(__file__),"helper_sha256":digest("experiments/phase2/tenfold_full_k4_component_contract.py"),
            "versions":{k:importlib.metadata.version(k) for k in ("flopscope","whestbench","numpy")},
            "metric":"B=W14.T W14, actual current public W; A reconstructs captured surrogate dG/offG, not actual historical transported G",
            "construction_cost":construction_cost,"full_component_cost":full_cost,
            "scalar_contract_12_entries":{"rawcap_max_absolute":max(raw_err),"feedcenter_max_absolute":max(feed_err)},
            "compression":{"rank":32,"seed":261013952,"method":"QR(B Omega), Q(Q.T B Q)Q.T + diag(B-low); no score/reference selection",
                           "relative_metric_Frobenius":float(np.linalg.norm(np.asarray(approx)-bn)/np.linalg.norm(bn)),
                           "max_metric_diagonal_error":float(np.max(abs(np.diag(np.asarray(approx))-np.diag(bn)))),
                           "min_diagonal_correction":float(np.min(np.asarray(diag_correction))),
                           "cost":compression_cost,"USE_feed_operator_errors":errors},
            "compressed_component_cost":compressed_cost,"no_adjusted_score":True,"not_resource_acceptance":True,
            "limitations":["Captured A remains prior surrogate; this is component algebra, not candidate history integration",
                           "Compressed metric materialized dense for operator parity; no claimed rank32 runtime benefit",
                           "Born feed history not compressed or transported in this real component run; smallshape separate covers exact transport",
                           "D4 preservation by diagonal correction does not imply D31/D22/feed preservation"]}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"full_component_flops":full_cost["flops_used"],
                      "compression":{k:v for k,v in report["compression"].items() if k!="cost"}},indent=2))


if __name__=="__main__":main()

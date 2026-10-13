"""Small actual H18 pregraph observer insertion and joined32 timing contract.

Target-free; no cloud, fitting, coefficients, labels or real reference.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext
from experiments.phase2.capture_pregraph_state_r50 import GraphCapture
from experiments.phase2.tenfold_local_features_r48 import prefeatures


def run(n,prune):
    parent=Path('estimators/sprint_public_v37b_h18_r01/estimator.py')
    text=parent.read_text();token='riders = (n == 1024 and L == len(CORR_BETA))';assert text.count(token)==1
    modified=text.replace(token,'riders = True')
    snapshot=Path(f'experiments/phase2/pregraph_h18_snapshot_r50_n{n}_{int(prune)}.py')
    assert not snapshot.exists();snapshot.write_text(modified)
    assert modified.replace('riders = True',token)==text
    spec=importlib.util.spec_from_file_location(f'r50_{n}_{prune}',snapshot);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.PRUNE=prune;mod.LR0=2;depth=4
    mod.S_IN=[n,n,n-4,n-4,n-8,n];mod.S_OUT=[n,n,n,n-4,n-4,n-8];mod.S_ON=[0]*depth;mod.PRUNE_EMU=False
    weights=(np.random.default_rng(261014307).normal(size=(depth,n,n))*math.sqrt(.5/n)).astype(np.float32)
    source=snapshot.resolve();lines=modified.splitlines()
    pre=next(i for i,v in enumerate(lines,1) if 'if riders and not NO_CORR:' in v)
    post=next(i for i,v in enumerate(lines,1) if 'mu = K1v if delta is None else K1v + delta' in v)
    graph=GraphCapture(n,depth);pres={};states={};scales={};nonidentity=0
    def tracer(frame,event,arg):
        nonlocal nonidentity
        if event!='line' or frame.f_code.co_filename!=str(source) or frame.f_code.co_name!='_predict_core':return tracer
        v=frame.f_locals;layer=v.get('li',-1)
        if layer<0 or layer>=depth-1:return tracer
        if frame.f_lineno==pre:
            graph.record_pre(v)
            d3=v['D3'] if v['mode']==1 else fnp.zeros(n,dtype=fnp.float32)
            norm=fnp.sum(fnp.abs(v['D21'].astype(fnp.float64)),axis=1) if v['mode']==1 else fnp.zeros(n,dtype=fnp.float32)
            g4=v['g4row'] if v['mode']==1 else fnp.zeros(n,dtype=fnp.float32)
            k4=v['K4_sigma'] if v['mode']==1 else 0.
            pres[layer]=np.asarray(prefeatures(v['mu'],v['var'],v['sigma'],v['alpha'],v['phi'],v['Phi'],v['pk1v'],d3,norm,g4,k4,layer)).copy()
            scales[layer]=np.stack([np.asarray(v['sigma']).astype(float)**p for p in range(1,5)],axis=1)
        elif frame.f_lineno==post:
            delta=0. if v['delta'] is None else np.asarray(v['delta'])
            states[layer]=np.stack([np.asarray(v['K1v']).astype(float)+delta]+[np.asarray(v[k]).astype(float) for k in ('K2v','K3v','K4v')],axis=1)
            inverse=np.arange(n) if v['inv_perm'] is None else np.asarray(v['inv_perm'])
            nonidentity+=int(np.any(inverse!=np.arange(n)))
            graph.canonicalize(layer,inverse)
            for data in (pres,states,scales):data[layer]=data[layer][inverse]
        return tracer
    def predict(capture):
        estimator=mod.Estimator()
        with flops.BudgetContext(flop_budget=2**41,quiet=True):estimator.setup(SetupContext(width=n,depth=depth,flop_budget=2**41,api_version='0.1',seed=0))
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            mlp=SimpleNamespace(width=n,weights=[fnp.asarray(w) for w in weights]);sys.settrace(tracer if capture else None)
            try:pred=np.asarray(estimator.predict(mlp,2**41)).copy()
            finally:sys.settrace(None)
        return pred,ctx.summary_dict()
    plain,plaincost=predict(False);observed,cost=predict(True)
    prediff=float(np.max(abs(plain-observed)));assert prediff==0
    prearray=np.stack([pres[i] for i in range(depth-1)]);statearray=np.stack([states[i] for i in range(depth-1)]);scalearray=np.stack([scales[i] for i in range(depth-1)])
    joined=graph.join_local32(prearray,statearray,scalearray)
    assert joined.shape==(depth-1,n,32) and np.isfinite(joined).all()
    assert np.array_equal(joined[:,:,:22],prearray)
    assert np.array_equal(joined[:,:,22:26],statearray/scalearray)
    assert np.array_equal(joined[:,:,26:],graph.array())
    assert np.array_equal(graph.array()[0],np.zeros((n,6)))
    if prune:assert nonidentity>0
    return dict(width=n,prune=prune,nonidentity_post_permutations=nonidentity,source_snapshot=str(snapshot),source_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),single_literal_rider_diagnostic=True,targetfree_prediction_unchanged=prediff,joined_features_shape=list(joined.shape),pre_graph_audit=graph.audit,plain_call_cost=plaincost,observer_call_cost=cost,scope='smallactualbodycapturewithoutreset/truth; offlineinstrumentednotcompetitionresource')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    rows=[run(n,prune) for n in (16,32) for prune in (False,True)]
    r=dict(complete=True,version='smallactualpregraphcapture-r50',rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cloud_launched=False,truth_access=False,fit=False,no_scoreclaim=True,limits=['diagnosticforce riders only; no1024fullcall/RPC','no oracleteacherreset here','samepred provesreadonlyobserver not learnedaccuracy','template guarded and not launched'])
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps([dict(width=r['width'],prune=r['prune'],prediff=r['targetfree_prediction_unchanged'],permutations=r['nonidentity_post_permutations']) for r in rows]))


if __name__=='__main__':main()

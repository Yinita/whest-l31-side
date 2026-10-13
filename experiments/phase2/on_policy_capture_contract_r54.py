"""Small full16 H18 stop-capture with actual frozen prefix head arithmetic.

Checks timing/isolation/invariance; frozenR48coeff used as actual arithmetic
fixtures, not claimed to be sequentially trained R54branch coefficients.
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
from experiments.phase2.on_policy_capture_helper_r54 import OnPolicyCapture,CaptureComplete,updated_diagonals


def run(n,prune,target,snapshot_tag):
    parent=Path('estimators/sprint_public_v37b_h18_r01/estimator.py');text=parent.read_text()
    token='riders = (n == 1024 and L == len(CORR_BETA))';assert text.count(token)==1
    body=text.replace(token,'riders = True');snapshot=Path(f'experiments/phase2/on_policy_h18_snapshot_r54_n{n}_p{int(prune)}_l{target}_{snapshot_tag}.py')
    assert not snapshot.exists();snapshot.write_text(body);assert body.replace('riders = True',token)==text
    spec=importlib.util.spec_from_file_location(f'r54_{n}_{prune}_{target}',snapshot);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.PRUNE=prune;mod.LR0=2
    # Keep estimator depth=16, original CORR_BETA/table lengths and allremaining
    # schedules. Tiny prefixrows change only diagnostic shapes, recorded.
    mod.S_IN=[n]*16;mod.S_OUT=[n]*16;mod.S_ON=[0]*16;mod.PRUNE_EMU=False
    weights=(np.random.default_rng(261014353).normal(size=(16,n,n))*math.sqrt(.5/n)).astype(np.float32)
    modelpath=Path('results/phase2/local_joint_closure_cv_r48/frozen_train24_local_closure_26.json');model=json.loads(modelpath.read_text())
    prefix={}
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        for layer in range(1,target):
            lm=model['layer_models'][str(layer)];prefix[layer]={k:fnp.asarray(np.asarray(lm[k],dtype=float)) for k in ('feature_mean','feature_scale','theta')}
    observer=OnPolicyCapture(n,target,prefix)
    source=snapshot.resolve();lines=body.splitlines();pre=next(i for i,v in enumerate(lines,1) if 'if riders and not NO_CORR:' in v);post=next(i for i,v in enumerate(lines,1) if 'mu = K1v if delta is None else K1v + delta' in v)
    beforepost={};nonidentity=0;layersseen=[]
    def tracer(frame,event,arg):
        nonlocal nonidentity
        if event!='line' or frame.f_code.co_filename!=str(source) or frame.f_code.co_name!='_predict_core':return tracer
        state=frame.f_locals;layer=state.get('li',-1)
        if layer<0:return tracer
        if frame.f_lineno==pre:observer.record_pre(state)
        elif frame.f_lineno==post:
            layersseen.append(layer)
            if layer==target:
                delta=0 if state['delta'] is None else np.asarray(state['delta']);inv=np.arange(n) if state['inv_perm'] is None else np.asarray(state['inv_perm'])
                beforepost['states']=np.stack([np.asarray(state['K1v']).astype(float)+delta]+[np.asarray(state[k]).astype(float) for k in ('K2v','K3v','K4v')],axis=1)[inv].copy()
                beforepost['delta']=np.zeros(n,dtype=float) if state['delta'] is None else np.asarray(state['delta'],dtype=float)[inv].copy()
                beforepost['delta_present']=state['delta'] is not None
                nonidentity=int(np.any(inv!=np.arange(n)))
            observer.record_post(state)
        return tracer
    estimator=mod.Estimator()
    assert len(mod.CORR_BETA)==16
    with flops.BudgetContext(flop_budget=2**41,quiet=True):estimator.setup(SetupContext(width=n,depth=16,flop_budget=2**41,api_version='0.1',seed=0))
    stopped=False
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        mlp=SimpleNamespace(width=n,weights=[fnp.asarray(w) for w in weights]);sys.settrace(tracer)
        try:estimator.predict(mlp,2**41)
        except CaptureComplete:stopped=True
        finally:sys.settrace(None)
    assert stopped and sys.gettrace() is None and observer.current is not None
    assert observer.injected==list(range(1,target)) and layersseen==list(range(target+1))
    current=observer.current;assert np.array_equal(current['states'],beforepost['states'])
    assert np.array_equal(current['inherited_delta'],beforepost['delta'])
    assert current['inherited_delta_present']==beforepost['delta_present']
    assert all(all(key in row for key in ('mean_floor_count','variance_floor_count','minimum_mean_before_floor','minimum_variance_before_floor')) for row in observer.audit if row.get('learned_prefix_injected'))
    assert current['features'].shape==(n,26) and current['states'].shape==current['scales'].shape==(n,4)
    assert np.max(abs(current['features'][:,22:]-current['states']/current['scales']))<1e-13
    # Worker interface cannot carry current/future head; negative control.
    rejected=False
    try:OnPolicyCapture(n,target,{**prefix,target:prefix.get(1,{})})
    except AssertionError:rejected=True
    assert rejected
    if prune and target>=2:assert nonidentity
    return dict(width=n,depth=16,prune=prune,target_layer=target,original_depth_tables_length=16,snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),diagnostic_shape_changes='literalridersTrue; smallS_IN/S_OUT/S_ON16vectors only; depth16unchanged',prefix_head_layers=list(prefix),prefix_fixture_source=str(modelpath),prefix_fixture_model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),fixture_not_onpolicy_training_evidence=True,seen_post_layers=layersseen,currentcapture_equal_precorrectionstate=True,current26_ownpost_maxerror=float(np.max(abs(current['features'][:,22:]-current['states']/current['scales']))),canonical_inherited_delta_equal=True,inherited_delta_present=current['inherited_delta_present'],prefix_floor_audit_present=True,nonidentity_current_permutation=bool(nonidentity),currentfuturehead_rejected=True,tracer_removed=True,audit=observer.audit,legal_instrumented_cost=ctx.summary_dict())


def update_contract():
    # Stress floors and a signed inherited delta while comparing the original
    # update expression exactly. Audit reads must not change update values.
    rng=np.random.default_rng(26105402);n=32;rows=[]
    for with_delta in (False,True):
        with flops.BudgetContext(flop_budget=2**41,quiet=True):
            channels=[fnp.asarray(rng.normal(size=n).astype(np.float32)) for _ in range(4)]
            sigma=fnp.asarray(np.exp(rng.normal(size=n)).astype(np.float32))
            correction=fnp.asarray(rng.normal(size=(n,4)))
            delta=fnp.asarray(rng.normal(size=n).astype(np.float32)) if with_delta else None
            s=sigma.astype(fnp.float64);change=correction*fnp.stack([s**p for p in range(1,5)],axis=1)
            m=channels[0].astype(fnp.float64)
            if delta is not None:m=m+delta.astype(fnp.float64)
            before_m=m+change[:,0];before_v=channels[1].astype(fnp.float64)+change[:,1]
            m=fnp.maximum(before_m,0.)
            if delta is not None:m=m-delta.astype(fnp.float64)
            expected=(m.astype(fnp.float32),fnp.maximum(before_v,1e-10).astype(fnp.float32),(channels[2].astype(fnp.float64)+change[:,2]).astype(fnp.float32),(channels[3].astype(fnp.float64)+change[:,3]).astype(fnp.float32))
            audit={};actual=updated_diagonals(*channels,sigma,correction,delta,diagnostic_audit=audit)
            assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(actual,expected))
            assert audit['mean_floor_count']==int((np.asarray(before_m)<0).sum())
            assert audit['variance_floor_count']==int((np.asarray(before_v)<1e-10).sum())
            assert audit['minimum_mean_before_floor']==float(np.asarray(before_m).min())
            assert audit['minimum_variance_before_floor']==float(np.asarray(before_v).min())
            rows.append(dict(inherited_delta=with_delta,bitidentical_original_update=True,**audit))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot-tag',required=True);a=p.parse_args();assert not a.output.exists()
    assert a.snapshot_tag.replace('_','').isalnum()
    rows=[run(n,prune,target,a.snapshot_tag) for n in (16,32) for prune in (False,True) for target in (1,3)]
    r=dict(complete=True,version='full16-small-onpolicy-stopcapture-contract-r54-r02',rows=rows,update_audit_contract=update_contract(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path('experiments/phase2/on_policy_capture_helper_r54.py'))},no_labels_currenttruth_heldtruth=True,no_cloud_launch=True,no_fitting=True,no_scoreclaim=True,scope='actualparentfull16table diagnostic stopcapture and realfrozenprefixarithmetic only; notR54trainedmodel/1024/resourcepass')
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps([dict(n=r['width'],target=r['target_layer'],prune=r['prune'],layers=r['prefix_head_layers'],error=r['current26_ownpost_maxerror']) for r in rows]))


if __name__=='__main__':main()

"""Synthetic context graph oracle/cost and optional actual poststate gate.

Actual poststate use is only a computability example, never prestate signal or
candidate score. No labels/reference/predictions accessed.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from experiments.phase2.tenfold_graph_features_r50 import features,FEATURE_ORDER


def reference(c,s,d,k):
    n=len(s);r=c/(s[:,None]*s[None,:]);dl=d/(s[:,None]**2*s[None,:]);dr=d.T/(s[:,None]*s[None,:]**2);kk=k/(s[:,None]**2*s[None,:]**2)
    return np.stack([(r*r).sum(axis=1)/(n-1),(r*r*r).sum(axis=1)/(n-1),np.diag(r@r@r)/((n-1)*(n-2)),(r*dl).sum(axis=1)/(n-1),(r*dr).sum(axis=1)/(n-1),(r*r*kk).sum(axis=1)/(n-1)],axis=1)


def evaluate(c,s,d,k):
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        actual=features(*[fnp.asarray(v) for v in (c,s,d,k)])
    expected=reference(c.astype(float),s.astype(float),d.astype(float),k.astype(float))
    error=float(np.max(abs(np.asarray(actual)-expected)));assert error<1e-11*max(float(np.max(abs(expected))),1),error
    perm=np.roll(np.arange(len(s)),1);inverse=np.argsort(perm)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        internal=features(fnp.asarray(c[perm][:,perm]),fnp.asarray(s[perm]),fnp.asarray(d[perm][:,perm]),fnp.asarray(k[perm][:,perm]))
    perr=float(np.max(abs(np.asarray(internal)[inverse]-expected)));assert perr<1e-11*max(float(np.max(abs(expected))),1),perr
    return dict(width=len(s),oracle_maxerror=error,oracle_relative_maxerror=error/max(float(np.max(abs(expected))),1),permutation_maxerror=perr,permutation_relative_maxerror=perr/max(float(np.max(abs(expected))),1),all_finite=bool(np.isfinite(np.asarray(actual)).all()),feature_ranges=[dict(min=float(x.min()),max=float(x.max()),rms=float(np.sqrt((x*x).mean()))) for x in np.asarray(actual).T],legal_component_cost=ctx.summary_dict())


def synthetic(n):
    rng=np.random.default_rng(261014278+n);q=rng.normal(size=(n,min(n,32)));c=q@q.T
    sig=np.sqrt(np.diag(c));np.fill_diagonal(c,0)
    d=rng.normal(size=(n,n))*.02;k=rng.normal(size=(n,n))*.03;k=(k+k.T)/2
    np.fill_diagonal(d,0);np.fill_diagonal(k,0)
    row=evaluate(c.astype(np.float32),sig.astype(np.float32),d.astype(np.float32),k.astype(np.float32))
    if n<=8:
        # Explicit cycle triple summation independently checks diag(rho^3).
        r=c/(sig[:,None]*sig[None,:]);dense=np.array([sum(r[i,j]*r[j,l]*r[l,i] for j in range(n) for l in range(n))/((n-1)*(n-2)) for i in range(n)])
        direct=reference(c,sig,d,k)[:,2];row['triplecycle_scalar_error']=float(np.max(abs(dense-direct)));assert row['triplecycle_scalar_error']<1e-13
    # K21 orientation is not symmetric; reject left/right collapse.
    expected=reference(c,sig,d,k);row['nonsymmetric_D21_orientation_gap']=float(np.max(abs(expected[:,3]-expected[:,4])));assert row['nonsymmetric_D21_orientation_gap']>1e-7
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    rows=[synthetic(n) for n in (4,8,32,1024)]
    capture=Path('results/phase2/h18_joint_li14_r25/post_state.npz');actual=None
    if capture.is_file():
        with np.load(capture,allow_pickle=False) as b:c=b['K11'].copy();d=b['K21'].copy();k=b['K22'].copy();v=b['K2v'].copy()
        # Parent postdiagapprox may be tinynegative; only this postcontext
        # computability diagnostic uses max1e-10, not a production prefeature.
        sig=np.sqrt(np.maximum(v.astype(float),1e-10));np.fill_diagonal(c,0)
        actual=evaluate(c,sig,d,k);actual.update(path=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),scope='actualpostlayer14ordinaryslices computabilityonly; not pre-nonlinearfeatures/corpusfit',negative_post_variances=int((v<0).sum()))
    result=dict(complete=True,version='fixed-six-graph-context-contract-r50',feature_order=FEATURE_ORDER,synthetic_rows=rows,actual_poststate_context=actual,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path('experiments/phase2/tenfold_graph_features_r50.py'))},truth_access=False,fit=False,no_scoreclaim=True,scope='target-free fixedgraphfeature algebra/cost; actualprestatecapturerequired; no candidate or baselinegain',limits=['R50requirescurrent preCoff/D21/wk4m matchingσ, not poststate substitutions','formal features of represented approximate graph not truecross labels','onefloat64n³GEMM perlayer may change strictwall/resource','not production resurrection of negativeH17triangle/R29pairgrade4'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(synthetic=[dict(width=r['width'],maxerror=r['oracle_maxerror'],flops=r['legal_component_cost']['flops_used']) for r in rows],actualposterror=None if actual is None else actual['oracle_maxerror'])))


if __name__=='__main__':main()

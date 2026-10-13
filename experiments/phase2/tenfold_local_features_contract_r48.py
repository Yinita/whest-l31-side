"""Target-free local26 timing, permutation, alias and rider feature audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from experiments.phase2.tenfold_local_features_r48 import prefeatures,ownpost_features,FEATURE_ORDER


def run(n):
    rng=np.random.default_rng(261014265+n)
    mu=rng.normal(size=n).astype(np.float32);var=rng.uniform(.1,2,n).astype(np.float32);sigma=np.sqrt(var);alpha=mu/sigma
    phi=(np.exp(-alpha**2/2)/np.sqrt(2*np.pi)).astype(np.float32)
    cdf=np.asarray([.5*(1+math.erf(v/math.sqrt(2))) for v in alpha],dtype=np.float32)
    pk1=(sigma*phi+mu*cdf).astype(np.float32);d3=(rng.normal(size=n)*.002).astype(np.float32)
    d21norm=rng.uniform(0,.01,n).astype(np.float64);g4=(rng.normal(size=n)*.003).astype(np.float32)
    inputs=[mu,var,sigma,alpha,phi,cdf,pk1,d3,d21norm,g4]
    floatinputs=[v.astype(float) for v in inputs];mm,vv,ss,aa,pp,cc,pk,dd,dnorm,gg=floatinputs
    scalar=float(np.float32(.001));layer=6;sk=dd/ss**3;ku=gg/ss**4;hermites=[np.ones(n),aa]
    for degree in range(1,7):hermites.append(aa*hermites[-1]-degree*hermites[-2])
    expectedpre=np.stack([np.ones(n),mm,vv,ss,aa,abs(aa),pp,cc,pk,dd,dnorm,np.full(n,scalar),ss*pp,sk*hermites[2]*pp,ku*hermites[3]*pp,sk**2*hermites[5]*pp,sk*ku*hermites[6]*pp,ku**2*hermites[7]*pp,sk,ku,cc,np.full(n,layer/15)],axis=1)
    base=rng.normal(size=(n,4)).astype(np.float32);base[:,0]=abs(base[:,0]);base[:,1]=abs(base[:,1]);rider=(rng.normal(size=n)*1e-4).astype(np.float32)
    post=base.astype(float);post[:,0]+=rider
    expected=np.concatenate([expectedpre,post/np.stack([ss**p for p in range(1,5)],axis=1)],axis=1)
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        ff=prefeatures(*[fnp.asarray(v) for v in inputs],scalar,layer)
        sigma_live=fnp.asarray(sigma);vals=[fnp.asarray(base[:,j].copy()) for j in range(4)]
        before=[np.asarray(v).copy() for v in vals]
        actual=ownpost_features(ff,sigma_live,*vals,fnp.asarray(rider))
    error=float(np.max(abs(np.asarray(actual)-expected)));assert error<1e-13,error
    assert all(np.array_equal(np.asarray(v),b) for v,b in zip(vals,before))
    # New output is a captured feature, not an alias to a state later corrected.
    captured=np.asarray(actual).copy()
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        fnp.copyto(vals[0],vals[0]+.03);fnp.copyto(vals[1],vals[1]+.04)
    aliaserror=float(np.max(abs(np.asarray(actual)-captured)));assert aliaserror==0
    wrongrider=np.concatenate([expectedpre,base.astype(float)/np.stack([ss**p for p in range(1,5)],axis=1)],axis=1)
    missingerror=float(np.max(abs(wrongrider-expected)));assert missingerror>1e-6
    # Component matches offline canonical saved feature assembly and the live
    # internal ordering. Use a non-involutive perm to detect wrong inverse.
    perm=np.roll(np.arange(n),1);inverse=np.argsort(perm)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        livepre=fnp.asarray(expectedpre[perm]);livevals=[fnp.asarray(base[perm,j]) for j in range(4)]
        live=ownpost_features(livepre,fnp.asarray(sigma[perm]),*livevals,fnp.asarray(rider[perm]))
    permutationerror=float(np.max(abs(np.asarray(live)[inverse]-expected)));assert permutationerror<1e-13
    wronginverseerror=float(np.max(abs(np.asarray(live)[perm]-expected)));assert wronginverseerror>.1
    return dict(width=n,features26_exactshape=list(np.asarray(actual).shape),f64_vs_offline_maxerror=error,post_snapshot_alias_error=aliaserror,source_state_unchanged=True,canonical_internal_feature_error=permutationerror,inverse_orientation_negative_control_error=wronginverseerror,missing_rider_negative_control_error=missingerror,legal_component_cost=ctx.summary_dict())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    result=dict(complete=True,version='target-free-local26-feature-contract-r48',feature_order=FEATURE_ORDER,rows=[run(n) for n in (8,32,1024)],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path('experiments/phase2/tenfold_local_features_r48.py'))},fit=False,no_coefficients=True,no_truth_access=True,no_scoreclaim=True,scope='prefeature/postfeature timing, float64 normalization, rider/permutation/alias component; no actualmodel/fullprediction')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps([dict(width=r['width'],maxerror=r['f64_vs_offline_maxerror'],componentflops=r['legal_component_cost']['flops_used']) for r in result['rows']]))


if __name__=='__main__':main()

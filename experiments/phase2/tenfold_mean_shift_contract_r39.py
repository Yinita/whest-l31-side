"""Exact current/next-layer deterministicshift and optional riderledger cost."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from experiments.phase2.tenfold_mean_shift_r39 import fold_mean_model,features,mean_shift,corrected_mean,FEATURE_ORDER


def run(n,model):
 rng=np.random.default_rng(261014163+n);mu=rng.normal(size=n);var=rng.uniform(.2,2,n);sigma=np.sqrt(var);alpha=mu/sigma;phi=np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi)
 cdf=np.array([.5*(1+math.erf(x/math.sqrt(2))) for x in alpha]);pk1=sigma*phi+mu*cdf;d3=rng.normal(size=n)*.002;norm=rng.uniform(0,.01,n);g4=rng.normal(size=n)*.003;layer=6;k4sigma=.001
 h=[np.ones(n),alpha]
 for degree in range(1,7):h.append(alpha*h[-1]-degree*h[-2])
 sk=d3/sigma**3;ku=g4/sigma**4
 f=np.stack([np.ones(n),mu,var,sigma,alpha,abs(alpha),phi,cdf,pk1,d3,norm,np.full(n,k4sigma),sigma*phi,sk*h[2]*phi,ku*h[3]*phi,sk**2*h[5]*phi,sk*ku*h[6]*phi,ku**2*h[7]*phi,sk,ku,cdf,np.full(n,layer/15)],axis=1)
 coef=np.array(fold_mean_model(model));reference=((f-np.asarray(model['feature_mean']))/np.asarray(model['feature_scale'])@np.asarray(model['theta']))[:,0]*sigma
 with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
  vals=[fnp.asarray(v) for v in (mu,var,sigma,alpha,phi,cdf,pk1,d3,norm,g4)]
  ff=features(*vals,k4sigma,layer);shift=mean_shift(ff,fnp.asarray(sigma),fnp.asarray(coef))
 cost=ctx.summary_dict();shift=np.asarray(shift);error=float(np.max(abs(shift-reference)));assert error<1e-11,error
 w=rng.normal(size=(n,n))/np.sqrt(n);k1=pk1;rider=rng.normal(size=n)*.0001
 current=corrected_mean(k1,rider,shift);nextmean=w@current
 ledger={'without_rider':float(np.max(abs(corrected_mean(k1,None,shift)-(k1+shift)))),
         'with_rider':float(np.max(abs(current-(k1+rider+shift)))),
         'nextlinear':float(np.max(abs(nextmean-(w@k1+w@rider+w@shift))))}
 # Wrong tracer subtract/adddelta catchesdouble-rider.
 wrongdouble=float(np.max(abs((k1+rider+shift)+rider-current)))
 assert max(ledger.values())<1e-11
 return {'width':n,'fixed_model_feature_order':FEATURE_ORDER,'normalized_vs_folded_maxerror':error,'shift/rider/nextlinear_ledger':ledger,'double_rider_negative_control_maxerror':wrongdouble,'legal_component_cost':cost,'current_shift_mean':float(shift.mean())}


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path);a=p.parse_args();assert not a.output.exists()
 if a.model:
  model=json.loads(a.model.read_text());modelsha=hashlib.sha256(a.model.read_bytes()).hexdigest();scope='actualfrozenr38modelmeanonlycomponent'
 else:
  rng=np.random.default_rng(261014164);model={'theta':(rng.normal(size=(22,4))*1e-4).tolist(),'feature_mean':rng.normal(size=22).tolist(),'feature_scale':rng.uniform(.1,2,size=22).tolist()};model['feature_mean'][0]=0;model['feature_scale'][0]=1;modelsha=None;scope='syntheticfixedcoefficientscomponent; no r38modelavailableyet'
 r={'version':'tenfold-mean-shift-contract-r39','complete':True,'rows':[run(n,model) for n in (8,32,1024)],'model_sha256':modelsha,'scope':scope,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/phase2/tenfold_mean_shift_r39.py')]},'no_benchmarktruth':True,'no_adjusted_score':True,'not_resource_acceptance':True,'plannedinsertion':'afterordinarycenter beforemu=K1+delta; computebprefeatures, leaveK2/3/4/centralpairs/oldrawbirth unchanged; reportcurrentmuandpredictionasK1+delta+b','limitations':['componentonly notwholecandidateadmissibility','signedtranslation maymakenonnegativeReLUoutputnegative','modeldatasetlicense unresolved no derivedcoef redistribution','r38thetafrozenvaluesnotreaduntilmodelargumentprovided']}
 a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'scope':scope,'rows':[{'width':x['width'],'maxerror':x['normalized_vs_folded_maxerror'],'flops':x['legal_component_cost']['flops_used']} for x in r['rows']]}))
if __name__=='__main__':main()

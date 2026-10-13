"""Targetfree actual firstGaussian ReLU full budget3 D22 and radial projection.

Independent scalar Gaussian rawmoment oracle; no benchmark reference access.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from experiments.phase2.tenfold_true_k4_producer import Producer,relu_wicks
from experiments.phase2.tenfold_true_k4_contract import numpy_producer


def gaussian22(mu,cov):
 n=len(mu);w=relu_wicks(mu,fnp.sqrt(fnp.diag(cov)),{(0,1),*( (d,p) for d in (1,2,3) for p in (1,2))})
 mean=w[0,1];off=cov-fnp.diag(fnp.diag(cov));p11=fnp.zeros((n,n),dtype=fnp.float64);raw=fnp.zeros((n,n),dtype=fnp.float64)
 for d in (1,2,3):
  core=off**d/math.factorial(d);a=w[d,2]-2*mean*w[d,1]
  raw=raw+core*a[:,None]*a[None,:]
  p11=p11+core*w[d,1][:,None]*w[d,1][None,:]
 return raw-2*p11*p11


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 weights=Path('benchmark/data/rpc_inputs/mlp_0.npz')
 with np.load(weights) as f:w=f['weights'][0].astype(float)
 # canonical firstlayer preCov/inputI. W official storage transposed estimator.
 cov=w.T@w;n=len(w);mu=np.zeros(n)
 with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:d22=gaussian22(fnp.asarray(mu),fnp.asarray(cov))
 d22=np.asarray(d22);small=8
 oracle=numpy_producer(mu[:small],cov[:small,:small],np.zeros((small,)*3),[(np.zeros((small,small)),np.eye(small))])[3]
 exact22=np.einsum('iijj->ij',oracle).copy();np.fill_diagonal(exact22,0)
 delta=float(np.max(abs(d22[:small,:small]-exact22)));assert delta<1e-10,delta
 with flops.BudgetContext(flop_budget=2**41,quiet=True):
  producer=Producer(fnp.asarray(mu),fnp.asarray(cov),fnp.zeros((n,1,1),dtype=fnp.float64),[(fnp.zeros((n,n),dtype=fnp.float64),fnp.eye(n,dtype=fnp.float64))])
  producer.k3=fnp.zeros((n,n,n),dtype=fnp.float64) if n<=32 else None
  producer.input_entry_old=producer.input_entry
  def gaussian_input(v,axes):
   if sum(v)==3:return fnp.zeros(np.broadcast_shapes(*(np.shape(x) for x in axes)),dtype=fnp.float64)
   return producer.input_entry_old(v,axes)
  producer.input_entry=gaussian_input
  # Marginal diagrams do not require threeindex K3 for Gaussianzero.
  d4=np.asarray(producer.entries((np.arange(n),)*4)).copy()
 # Harmonic projection onto identity-metric: trace uses D4+rowD22;
 # offdiag trace needs211 too; here compare exact repeatedtrace component only.
 t=np.diag(d4)+np.diag(d22.sum(axis=1))
 g=6/(n+4)*np.diag(t)-3/((n+2)*(n+4))*np.trace(t)*np.eye(n)
 diag4core=np.diag(g);diag4radial_without22=6/(n+4)*d4-3/((n+2)*(n+4))*d4.sum()
 report={'version':'tenfold-Gaussian-cross22-firstgate-r01','complete':True,'network_id':0,'layer':0,'width':n,'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'legal22cost':ctx.summary_dict(),'scalar_fullordinary_budget3_n8_maxdelta':delta,
  'D4norm':float(np.linalg.norm(d4)),'D22offnorm':float(np.linalg.norm(d22)),'D22rowsumnorm':float(np.linalg.norm(d22.sum(axis=1))),
  'D4mean':float(d4.mean()),'D22rowsummean':float(d22.sum(axis=1).mean()),
  'radialGdiag_with22_norm':float(np.linalg.norm(diag4core)),'radialGdiag_without22_norm':float(np.linalg.norm(diag4radial_without22)),
  'rowtrace22_fraction_abs':float(np.linalg.norm(d22.sum(axis=1))/np.linalg.norm(d4+d22.sum(axis=1))),
  'no_adjusted_score':True,'not_resource_acceptance':True,'formula':'D22off=sum_d1..3 Cij^d/d! adi adj −2 pK11²; ad=w(d,2)−2postmean*w(d,1); pK11=sum_d1..3 Cij^d/d!w(d,1)i*w(d,1)j',
  'limitations':['inputGaussianfirstlayer only; diagramclosurebudget3 not exactbivariateReLU','harmonic diagonal usesrepeat-trace only; no211offdiag','notcandidatescore or proofcrossdeletionsolecause']}
 a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ('D4norm','D22offnorm','D22rowsumnorm','D4mean','D22rowsummean','scalar_fullordinary_budget3_n8_maxdelta','rowtrace22_fraction_abs')}))
if __name__=='__main__':main()

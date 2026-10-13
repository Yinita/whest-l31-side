"""Actual R25 estimator two-step production bookkeeping vs dense CP4/CP3."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,importlib.util,itertools,json,math,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext


def cp4(q,c):return np.einsum('ir,jr,kr,lr,r->ijkl',q,q,q,q,c,optimize=True)
def cp3(q,x):
 t=np.einsum('ir,jr,kr->ijk',x,q,q,optimize=True)
 return sum(t.transpose(p) for p in itertools.permutations(range(3)))/6

def run(n):
 path=Path('estimators/tenfold_sparse_d4_r25/estimator.py').resolve();spec=importlib.util.spec_from_file_location('r25',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);est=mod.Estimator();depth=4
 weights=(np.random.default_rng(261014107+n).normal(size=(depth,n,n))*math.sqrt(2/n)).astype(np.float32)
 with flops.BudgetContext(flop_budget=2**41,quiet=True):est.setup(SetupContext(width=n,depth=depth,flop_budget=2**41,api_version='0.1',seed=0))
 lines=path.read_text().splitlines();pre=next(i for i,l in enumerate(lines,1) if l.strip()=='sigma = fnp.sqrt(var)');rep=next(i for i,l in enumerate(lines,1) if l.strip().startswith('rep21 = D21_new'));post=next(i for i,l in enumerate(lines,1) if l.strip()=='if regen:' and i>1800)
 captures={}
 def trace(frame,event,arg):
  if event=='line' and frame.f_code.co_filename==str(path):
   s=frame.f_locals;li=s.get('li',-1)
   if frame.f_lineno in (pre,rep,post):
    keys={pre:('sparse_q','sparse_x','sparse_c','g4row','wk431','wk4m','sparse_feed_d3','sparse_feed_d21'),rep:('sparse_q','sparse_c','w1','w2','sparse_new_x','sparse_post_q','sparse_birth_feed3','sparse_birth_feed21'),post:('sparse_q','sparse_x','sparse_c','K4v','inherited4')}[frame.f_lineno]
    captures[(li,frame.f_lineno)]={k:np.asarray(s[k]).copy() for k in keys if k in s and s[k] is not None}
  return trace
 with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
  sys.settrace(trace)
  try:pred=est.predict(SimpleNamespace(width=n,depth=depth,weights=[fnp.asarray(w) for w in weights]),2**41)
  finally:sys.settrace(None)
 errors=[]
 for li in range(1,depth):
  c=captures[(li,pre)];q,x,co=c['sparse_q'].astype(float),c['sparse_x'].astype(float),c['sparse_c'].astype(float);h=cp4(q,co);f=cp3(q,x)
  d4=np.einsum('iiii->i',h);d31=np.einsum('iiij->ij',h).copy();d22=np.einsum('iijj->ij',h).copy();d21=np.einsum('iij->ij',f).copy();np.fill_diagonal(d31,0);np.fill_diagonal(d22,0);np.fill_diagonal(d21,0)
  er={'D4':float(np.max(abs(c['g4row']-d4))),'feedpriorD3':float(np.max(abs(c['sparse_feed_d3']-np.einsum('iii->i',f))))}
  if li<depth-1:
   er.update({'D31':float(np.max(abs(c['wk431']-d31))),'D22':float(np.max(abs(c['wk4m']-d22))),'feedpriorD21':float(np.max(abs(c['sparse_feed_d21']-d21)))})
   b=captures[(li,rep)];w1,w2=b['w1'].astype(float),b['w2'].astype(float)
   expected=.5*(np.einsum('i,j,k,iijk->ijk',w2,w1,w1,h)+np.einsum('i,j,k,ijjk->ijk',w1,w2,w1,h)+np.einsum('i,j,k,ijkk->ijk',w1,w1,w2,h))
   actual=cp3(b['sparse_post_q'].astype(float),b['sparse_new_x'].astype(float));expected21=np.einsum('iij->ij',expected).copy();np.fill_diagonal(expected21,0)
   er.update({'complete_feedtensor':float(np.max(abs(actual-expected))),'birthfeedD3':float(np.max(abs(b['sparse_birth_feed3']-np.einsum('iii->i',expected)))),'birthfeedD21':float(np.max(abs(b['sparse_birth_feed21']-expected21)))})
  errors.append({'layer':li,'maxerrors':er})
 for li in range(depth-1):
  p=captures[(li,post)];q,c=p['sparse_q'].astype(float),p['sparse_c'].astype(float)
  err=float(np.max(abs(np.sum(q**4*c,axis=1)-p['K4v'])))
  errors.append({'layer':li,'birthD4exact':err})
 assert all(max(r.get('maxerrors',{'birth':r.get('birthD4exact',0)}).values())<2e-5 for r in errors),errors
 return {'width':n,'depth':depth,'errors':errors,'legal_predict_cost':ctx.summary_dict(),'prediction_finite':bool(np.isfinite(np.asarray(pred)).all())}


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 rows=[run(n) for n in (8,32)];q=Path('estimators/tenfold_sparse_d4_r25/estimator.py')
 report={'version':'tenfold-sparse-d4-production-contract-r01','complete':True,'candidate_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,'no_adjusted_score':True,'not_resource_acceptance':True,'scope':'actual candidate generic n8/32/depth4, full source history and complete feed, independent dense signed CP oracle','limitations':['keeps onlynewD4birth support; omitted newcrosssupport','same frozenparent nonlinear closure not fullK211','not frozenMini8 score or strict1024resource']}
 a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'sha':report['candidate_sha256'],'rows':[{'width':r['width'],'errors':r['errors'],'flops':r['legal_predict_cost']['flops_used']} for r in rows]}))
if __name__=='__main__':main()

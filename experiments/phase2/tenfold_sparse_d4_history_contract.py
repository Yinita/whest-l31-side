"""Two-step dense independent CP4 oracle for sparse D4 histories and fullfeed."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,itertools,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from experiments.phase2.tenfold_sparse_d4_history import DiagonalHistory
from experiments.phase2.tenfold_true_k4_support_lift_contract import feed_numpy


def dense(q,c):return np.einsum('ir,jr,kr,lr,r->ijkl',q,q,q,q,c,optimize=True)


def run(n):
 rng=np.random.default_rng(261014106+n);q=rng.normal(size=(n,3))*.2;c=rng.normal(size=3)
 hist=DiagonalHistory(fnp.asarray(q),fnp.asarray(c));steps=[]
 for step in range(2):
  w=rng.normal(size=(n,n))/math.sqrt(n);gain=rng.uniform(.2,.9,n);mean=rng.normal(size=n)*.2;w1=rng.uniform(.1,.8,n);w2=rng.uniform(.03,.4,n);target=rng.normal(size=n)*.3
  prior=dense(q,c);transported=np.einsum('ia,jb,kc,ld,abcd->ijkl',w*gain,w*gain,w*gain,w*gain,prior,optimize=True)
  missing=target-np.einsum('iiii->i',transported);oracle=transported.copy()
  for i in range(n):oracle[i,i,i,i]+=missing[i]
  with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
   hist=hist.move(fnp.asarray(w),fnp.asarray(gain)).add_exact_d4_birth(fnp.asarray(target))
   use=hist.use_feed(fnp.asarray(mean),fnp.asarray(w1),fnp.asarray(w2));apply=hist.apply(fnp.eye(n,dtype=fnp.float64))
  expected=[np.einsum('iiii->i',oracle),np.einsum('iiij->ij',oracle).copy(),np.einsum('iijj->ij',oracle).copy(),*feed_numpy(oracle,mean,w1,w2)]
  for a in expected[1:3]:np.fill_diagonal(a,0)
  errors={name:float(np.max(abs(np.asarray(x)-y))) for name,x,y in zip(('D4','D31','D22','feedD3','feedD21','center'),use,expected)}
  errors['apply']=float(np.max(abs(np.asarray(apply)-np.einsum('ijkk->ij',oracle))))
  assert max(errors.values())<1e-11,errors
  q,c=np.asarray(hist.legs).copy(),np.asarray(hist.coefs).copy()
  steps.append({'step':step,'columns':len(c),'maxerrors':errors,'legal_cost':ctx.summary_dict()})
 return {'width':n,'steps':steps}


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();assert not a.output.exists()
 report={'version':'tenfold-sparse-d4-history-contract-r01','complete':True,'rows':[run(n) for n in (4,8,16)],
  'sources':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (Path(__file__),Path('experiments/phase2/tenfold_sparse_d4_history.py'))},
  'before':'radial trace history approximation','after':'shared CP4 legs+signed birthD4 residual, leading gain actualW once, exact allUSE and all3feed current contractions',
  'no_adjusted_score':True,'not_resource_acceptance':True,
  'limitations':['birth keeps D4 support only, other new support omitted','all feed histories not yet stored/transported','signed arbitrary toy algebra, not benchmark score','no real1024 cost run']}
 a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'rows':[{'width':r['width'],'maxerrors':[s['maxerrors'] for s in r['steps']],'costs':[s['legal_cost']['flops_used'] for s in r['steps']]} for r in report['rows']]}))
if __name__=='__main__':main()

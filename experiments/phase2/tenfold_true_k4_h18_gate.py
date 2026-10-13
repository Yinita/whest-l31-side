"""Actual H18 targetfree internal li14 slice: support producer cost/residual.

Preserves all source columns; neuron rows fixed first32 internal indices.
Not a random n32 network or independent generalization. No target read.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,itertools,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from experiments.phase2.tenfold_true_k4_producer import Producer
from experiments.phase2.tenfold_true_k4_support_lift import SupportLift


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--width',type=int,default=32);args=p.parse_args();assert not args.output.exists()
 cache=Path('results/phase2/h18_joint_li14_r25');meta=json.loads((cache/'completion.json').read_text());n=args.width
 with np.load(cache/'pre_state.npz') as f:state={k:f[k].copy().astype(float) for k in f.files if f[k].dtype.kind!='O'}
 c=state['C_off'][:n,:n]+np.diag(state['var'][:n]);mu=state['mu'][:n]
 k3=np.zeros((n,)*3);sum3=np.zeros(n);sum21=np.zeros((n,n));hashes={}
 for path in sorted(cache.glob('source_*.npz')):
  hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path) as f:s={k:f[k].copy().astype(float) for k in f.files}
  a,p,z=s['a'][:n],s['p'][:n],s['z'][:n];f1,f2=s['f1'][:n],s['f2'][:n]
  x1=3*a+f1@s['r1'].T;y1=a*s['w2']+f2@s['r2'].T
  m=p*s['s']+3*a*s['e']+z@s['ell'].T;x3=a*s['c1']+p*s['c2']
  blocks=[(x1,p,y1),(m,p,p),(x3,p,np.broadcast_to(s['y'][:n,None],p.shape))]
  for xx,yy,zz in blocks:
   t=np.einsum('ir,jr,kr->ijk',xx,yy,zz,optimize=True)
   k3+=sum(t.transpose(perm) for perm in itertools.permutations(range(3)))/6
 raw3=np.einsum('iii->i',k3);raw21=np.einsum('iij->ij',k3).copy();np.fill_diagonal(raw21,0)
 parity={'D3max':float(np.max(abs(raw3-state['D3'][:n]))),'D21max':float(np.max(abs(raw21-state['D21'][:n,:n])))}
 # Actual current parent surrogate G; retain scope, nottrue historicalK4.
 a=state['C_off'][:n,:n]*float(state['lam_prev']);np.fill_diagonal(a,state['dG'][:n]);a=(a+a.T)/2
 b=np.eye(n)*float(state['METRIC_C']);pairs=[(a,b)]
 start=time.monotonic()
 with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
  producer=Producer(fnp.asarray(mu),fnp.asarray(c),fnp.asarray(k3),[(fnp.asarray(a),fnp.asarray(b))])
  lift=SupportLift.birth(producer)
 cost=ctx.summary_dict();d4,d31,d22=[np.asarray(v).copy() for v in (lift.d4,lift.d31,lift.d22)]
 svals=np.linalg.svd(d22,compute_uv=False);norm=float(np.linalg.norm(d22));norm4=float(np.linalg.norm(d4));norm31=float(np.linalg.norm(d31))
 def spectral(v):return [float(np.linalg.norm(v[k:])/max(np.linalg.norm(v),1e-30)) for k in (1,2,4,8,16) if k<=len(v)]
 # Fixed diagnostic structured HadamardCov² component fit uses represented
 # d22 only; cannot become candidate selector or testtruth projection.
 shape=c*c;np.fill_diagonal(shape,0);alpha=float(np.sum(shape*d22)/max(np.sum(shape*shape),1e-30))
 report={'version':'tenfold-true-k4-H18-gate-r01','complete':True,'width_slice':n,'original_width':1024,'layer':14,
  'parent_sha256':meta['source_sha256'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'capture_sha256':hashes,
  'representedK3_vs_parent_dslice':parity,'source_columns':1024,'source_count':14,'internal_original_neuron_ids':state.get('selected_original_neuron_ids',np.arange(n))[:n].tolist(),
  'birth_legal_cost':cost,'birth_wall_s':time.monotonic()-start,'birthD4norm':norm4,'birthD31norm':norm31,'birthD22norm':norm,
  'weighted_support_norms':{'4':norm4,'31':2*norm31,'22':float(np.sqrt(3)*norm)},
  'D22singular_values':svals.tolist(),'D22offdiag_rank_residual_at1_2_4_8_16':spectral(svals),
  'structured_Cov2_projection_nonoracle':{'alpha':alpha,'relative_residual':float(np.linalg.norm(d22-alpha*shape)/max(norm,1e-30))},
  'no_adjusted_score':True,'not_resource_acceptance':True,
  'limitations':['fixed actualH18 leading32 neuron slice, not n32randomnetwork/fullwidth generalization','inputK4 actualparentsurrogateG rather than completehistory','K3complete representedoldsourceallcols; parentpruningreadout differs','spectrum/projection offline modelinternaldiagnosis, not productionfreecomputation','no target access']}
 args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ('width_slice','representedK3_vs_parent_dslice','birthD4norm','birthD31norm','birthD22norm','D22offdiag_rank_residual_at1_2_4_8_16','structured_Cov2_projection_nonoracle','birth_wall_s')}))
if __name__=='__main__':main()

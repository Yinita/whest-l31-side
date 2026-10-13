"""Independent support masks, real retained birth, USE/feed and cost contracts."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,itertools,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from experiments.phase2.tenfold_true_k4_producer import Producer,project_birth_r2
from experiments.phase2.tenfold_true_k4_contract import numpy_producer,pair_dense,operator_errors
from experiments.phase2.tenfold_true_k4_support_lift import SupportLift


def dense_lift(d4,d31,d22):
    n=len(d4);h=np.zeros((n,)*4)
    for i in range(n):
        h[i,i,i,i]=d4[i]
        for j in range(n):
            if i==j:continue
            for p in set(itertools.permutations((i,i,i,j))):h[p]=d31[i,j]
            for p in set(itertools.permutations((i,i,j,j))):h[p]=(d22[i,j]+d22[j,i])/2
    return h


def feed_numpy(h,mean,w1,w2):
    n=len(mean);f=np.zeros((n,)*3)
    for i,j,k in itertools.product(range(n),repeat=3):
        f[i,j,k]=.5*(w2[i]*w1[j]*w1[k]*h[i,i,j,k]+w1[i]*w2[j]*w1[k]*h[i,j,j,k]+w1[i]*w1[j]*w2[k]*h[i,j,k,k])
    d3=np.einsum('iii->i',f);d21=np.einsum('iij->ij',f).copy();np.fill_diagonal(d21,0)
    center=np.zeros((n,n))
    for j,k in itertools.product(range(n),repeat=2):
        if j!=k:center[j,k]=-2*sum(mean[i]*f[i,j,k] for i in range(n) if i not in (j,k))
    return d3,d21,center


def run(n):
    start=time.monotonic();rng=np.random.default_rng(261014102+n)
    raw=rng.normal(size=(n,n));cov=np.eye(n)+.05*raw@raw.T/n;mu=rng.normal(size=n)*.2
    u=rng.normal(size=(n,2))*.03;k3=np.einsum('ir,jr,kr->ijk',u,u,u)
    pairs=[(np.diag(rng.normal(size=n)*.025),np.eye(n)+cov*.1)]
    oracle=numpy_producer(mu,cov,k3,pairs)[3]
    w=np.eye(n)*.9+rng.normal(size=(n,n))*.08/math.sqrt(n)
    z=rng.normal(size=n);w1=rng.uniform(.1,.9,n);w2=rng.uniform(.05,.5,n);mean=rng.normal(size=n)*.1
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        producer=Producer(fnp.asarray(mu),fnp.asarray(cov),fnp.asarray(k3),[(fnp.asarray(a),fnp.asarray(b)) for a,b in pairs])
        lift=SupportLift.birth(producer)
        birthcost=ctx.summary_dict()
    gain=np.asarray(producer.wicks[(1,1)])
    inherited=sum(pair_dense(a,b) for a,b in pairs)*np.einsum('i,j,k,l->ijkl',gain,gain,gain,gain)
    birth=oracle-inherited;d4=np.einsum('iiii->i',birth);d31=np.einsum('iiij->ij',birth).copy();d22=np.einsum('iijj->ij',birth).copy()
    support=dense_lift(d4,d31,d22)
    errors={'birthD4':float(np.max(abs(np.asarray(lift.d4)-d4))),
            'birthD31':float(np.max(abs(np.asarray(lift.d31)-(d31-np.diag(np.diag(d31)))))),
            'birthD22':float(np.max(abs(np.asarray(lift.d22)-(d22-np.diag(np.diag(d22))))))}
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        metered_entries=lift.entries(tuple(np.indices((n,)*4)))
    # Dense n4 only in bounded contract driver, never component transport.
    errors['partition_entry']=float(np.max(abs(np.asarray(metered_entries)-support)))
    target=np.einsum('ia,jb,kc,ld,abcd->ijkl',w,w,w,w,support,optimize=True)
    tests=[np.eye(n),cov,(lambda a:(a+a.T)/2)(rng.normal(size=(n,n)))]
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        fw,fz=fnp.asarray(w),fnp.asarray(z)
        applied=[lift.transported_apply(fw,fnp.asarray(b)) for b in tests]
        use=lift.use_and_caps(fw,fz);feed=lift.all_feed(fw,fnp.asarray(mean),fnp.asarray(w1),fnp.asarray(w2))
        transportcost=ctx.summary_dict()
    expected=[np.einsum('iiii->i',target),np.einsum('iiij->ij',target).copy(),np.einsum('iijj->ij',target).copy()]
    for a in expected[1:]:np.fill_diagonal(a,0)
    fcap=np.einsum('i,iijk->jk',z,target);pcap=np.einsum('i,ijjk->jk',z,target)
    for j,k in itertools.product(range(n),repeat=2):
        if j==k:fcap[j,k]=pcap[j,k]=0
        else:
            fcap[j,k]-=z[j]*target[j,j,j,k]+z[k]*target[j,k,k,k]
            pcap[j,k]-=z[j]*target[j,j,j,k]+z[k]*target[j,j,k,k]
    expected.extend([fcap,pcap]);expectedfeed=feed_numpy(target,mean,w1,w2)
    for name,x,y in zip(('D4','D31','D22','Fcap','Pcap'),use,expected):errors[name]=float(np.max(abs(np.asarray(x)-y)))
    for name,x,y in zip(('feedD3','feedD21','feedCenter'),feed,expectedfeed):errors[name]=float(np.max(abs(np.asarray(x)-y)))
    errors['apply']=max(float(np.max(abs(np.asarray(a)-np.einsum('ijkl,kl->ij',target,b)))) for a,b in zip(applied,tests))
    assert max(errors.values())<1e-9,errors
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        projected,alpha=project_birth_r2(producer,fnp.asarray(cov))
        projectcost=ctx.summary_dict()
    r2=sum(pair_dense(np.asarray(a),np.asarray(b)) for a,b in projected)
    hybrid=r2+dense_lift(np.einsum('iiii->i',birth-r2),np.einsum('iiij->ij',birth-r2),np.einsum('iijj->ij',birth-r2))
    row={'width':n,'maxerrors':errors,'birth_legal_cost':birthcost,'transport_feed_legal_cost':transportcost,'r2projection_legal_cost':projectcost,
         'birthFrob_before_R2':float(np.linalg.norm(birth-r2)/np.linalg.norm(birth)),
         'birthFrob_after_support_only':float(np.linalg.norm(birth-support)/np.linalg.norm(birth)),
         'birthFrob_after_R2_plus_exact_support':float(np.linalg.norm(birth-hybrid)/np.linalg.norm(birth)),
         'operator_before_R2':operator_errors(birth,r2,tests),'operator_after_support_only':operator_errors(birth,support,tests),
         'operator_after_R2_plus_exact_support':operator_errors(birth,hybrid,tests),'wall_time_s':time.monotonic()-start}
    print(json.dumps({k:row[k] for k in ('width','maxerrors','birthFrob_before_R2','birthFrob_after_support_only','birthFrob_after_R2_plus_exact_support','wall_time_s')}),flush=True)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--widths',type=int,nargs='+',default=[4,8]);a=p.parse_args();assert not a.output.exists();assert max(a.widths)<=16
    report={'version':'tenfold-true-k4-support-lift-r01','complete':True,'rows':[run(n) for n in a.widths],
            'source_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (Path(__file__),Path('experiments/phase2/tenfold_true_k4_support_lift.py'),Path('experiments/phase2/tenfold_true_k4_producer.py'))},
            'before':'fixed identity/Cov2 birth basis (R2)',
            'after':'preserve true retained D4/D31/D22 disjoint support; optional additive residual of R2; exact matrix-free linear transport and USE/all3feed current contractions',
            'no_adjusted_score':True,'not_resource_acceptance':True,
            'limitations':['toy retained producer, not true distribution','n4 dense tensor only independent bounded contract driver','full transported D22 needs O(n4); remaining transport contractions O(n3)','no history feed state implemented','birth source extraction O(n2) output with expensive fixed diagrams','no real1024 candidate or scores']}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()

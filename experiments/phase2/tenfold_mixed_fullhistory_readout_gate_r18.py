"""Target-free full14source exact mixed K111*edge four-output readout gate.

Offline readout diagnosis, no whole candidate and no reference/oracle targets.
"""
# ruff: noqa: E402
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import itertools
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import load_dataset,mlp_at
from experiments.phase2.audit_official_augmented_terms import coefficient,extract_generators
from experiments.phase2.real_k31_production import legal_wicks
from experiments.phase2.tenfold_source_scalar_projection import source_projection

OUTPUT=Path('results/phase2/tenfold_mixed_fullhistory_readout_gate_r18.json')
CANDIDATE_PARENT=Path('estimators/k3_v29/estimator.py')
KEYS=('a','p','z','ell','w2','s','e','c1','c2','y','f1','f2','r1','r2')

def sym(a,b,c):
    return sum(np.einsum('it,jt,kt->ijk',*v) for v in itertools.permutations((a,b,c)))/6

def tiny_contract():
    rng=np.random.default_rng(67118)
    rows=[]
    for n in (4,8):
        a,p=[rng.normal(size=(n,n)) for _ in range(2)]
        z,ell=[rng.normal(size=(n,2)) for _ in range(2)]
        w2,s,e,c1,c2,y,u,v=[rng.normal(size=n) for _ in range(8)]
        f1,f2,r1,r2=[rng.normal(size=(n,2)) for _ in range(4)]
        raw=rng.normal(size=(n,n));cov=(raw+raw.T)/2;np.fill_diagonal(cov,0)
        parts=(a,p,z,ell,w2,s,e,c1,c2,y,f1,f2,r1,r2)
        blocks=((3*a+f1@r1.T,p,a*w2+f2@r2.T),
                (p*s+3*a*e+z@ell.T,p,p),
                (a*c1+p*c2,p,np.broadcast_to(y[:,None],p.shape)))
        tensor=sum(sym(*b) for b in blocks)
        # Independently enumerate allthree labellededge terms, no Sym reduction.
        expected=0.
        for i,j,k in itertools.permutations(range(n),3):
            expected+=tensor[i,j,k]*(cov[i,j]*v[i]*v[j]*u[k]
                                    +cov[i,k]*v[i]*u[j]*v[k]
                                    +cov[j,k]*u[i]*v[j]*v[k])
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            h=fnp.asarray(cov)*fnp.asarray(v)[:,None]*fnp.asarray(v)[None,:]
            got=3*source_projection(h,*map(fnp.asarray,parts),fnp.asarray(u))
        error=abs(float(got)-expected)
        assert error<1e-10*max(abs(expected),1)
        rows.append({'width':n,'labelled_terms':3,'labelled_coefficient':1,'independent_error':error,'legal_flops':ctx.summary_dict()['flops_used']})
    return rows

def main():
    assert not OUTPUT.exists()
    with tempfile.TemporaryDirectory() as td:
        dest=Path(td)/'src/mlp_kprop';dest.mkdir(parents=True)
        for name in ('partitions.py','kprop_harmonic.py'):
            shutil.copyfile(Path('benchmark/data/official_source_93d091a4')/name,dest/name)
        ns,sources=extract_generators(Path(td))
        part=((0,1,1),(1,1,1))
        count=ns['get_all_terms_iso'](3,d_max=4,augment=True)[(1,1,1)][part]
        assert count==3 and coefficient(part,1)==1
        assert not ns['factored_keeps_term'](3,(1,1,1),part)
    toy=tiny_contract()
    allmeta=json.loads(Path('results/phase2/ablation_v30_067_r18_all_source_capture.json').read_text())
    statemeta=json.loads(Path('results/phase2/ablation_v30_067_r24_real_k31_capture.json').read_text())
    assert allmeta['source_sha256']==statemeta['source_sha256']
    assert hashlib.sha256(CANDIDATE_PARENT.read_bytes()).hexdigest()==allmeta['source_sha256']
    cap=Path(statemeta['capture'])
    assert hashlib.sha256(cap.read_bytes()).hexdigest()==statemeta['capture_sha256']
    with np.load(cap) as src:
        c=src['C_off'].astype(np.float64)
        c=(c+c.T)/2;np.fill_diagonal(c,0)
        mu,sigma=[src[k].astype(np.float64) for k in ('mu','sigma')]
    # Weights only; deliberately omit published reference column.
    dataset=load_dataset('benchmark/data/mini_first8',split='mini').select_columns(['mlp_id','mlp_name','mlp_seed','weights'])
    weight=np.asarray(mlp_at(dataset,0).weights[15]).T.astype(np.float64)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        ws=legal_wicks(fnp.asarray(mu),fnp.asarray(sigma))
        wick1,wick2=[np.asarray(ws[(i,1)]).copy() for i in (1,2)]
    operands=[]
    for i in range(14):
        path=Path(f'benchmark/data/v29_real_source_v30_067_r18/source_{i:02d}.npz')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==allmeta['source_files_sha256'][str(path)]
        with np.load(path) as src:
            operands.append({k:src[k].astype(np.float64) for k in KEYS})
    rows=[]
    for output in (0,1,2,3):
        start=time.perf_counter()
        u=weight[output]*wick1;v=weight[output]*wick2
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            h=fnp.asarray(c)*fnp.asarray(v)[:,None]*fnp.asarray(v)[None,:]
            uf=fnp.asarray(u)
            per_source=[]
            for data in operands:
                value=3*source_projection(h,*[fnp.asarray(data[k]) for k in KEYS],uf)
                per_source.append(float(value))
        signal=sum(per_source)
        rows.append({'output':output,'mixed_next_d3_signal':signal,'per_source_signal':per_source,
                     'all14_sources_all1024_columns':True,
                     'legal_flops_float64':ctx.summary_dict()['flops_used'],
                     'wall_seconds':time.perf_counter()-start,
                     'not_predict_resource_acceptance':True})
        print(rows[-1],flush=True)
    report={'version':'tenfold.mixed.fullhistory.readout.r18','complete':True,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_projection_sha256':hashlib.sha256(Path('experiments/phase2/tenfold_source_scalar_projection.py').read_bytes()).hexdigest(),
            'official_generator_sources':sources,'orbit':part,'labelled_coefficient':1,'orbit_size':count,
            'toy_contract':toy,'parent_sha256':allmeta['source_sha256'],
            'state_capture_sha256':statemeta['capture_sha256'],'allsource_capture_sha256':allmeta['capture_sha256'],
            'data_commit':allmeta['dataset_commit'],'network_id':'0','layer':14,'rows':rows,
            'raw_mse':None,'adjusted_score':None,'no_reference_or_target_access':True,
            'scope':'Exact one-next-step D3 functional of omitted K111*edge orbit for allrepresentedV29sources; fixedoutputs0-3; no historycandidate',
            'limitations':['Captured parent K111 inherits pruning/confinement/Mresidual approximations',
                           'Float64 component billing is explicit; submitted candidate may usefloat32 but no unmeasured cost divided claim',
                           'Full outputvector and D21/history transport not computed',
                           'Signal magnitude does not establish accuracy improvement or choose coefficients/outputs']}
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    main()

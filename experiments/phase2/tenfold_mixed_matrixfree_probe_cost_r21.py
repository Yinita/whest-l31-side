"""Matrix-free fullrepresentedK111 mixedorbit probe cost, no denseQ/readtruth."""
# ruff: noqa: E402
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key]="1"
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import load_dataset,mlp_at
from experiments.phase2.real_k31_production import legal_wicks
from experiments.phase2.tenfold_mixed_fullhistory_readout_gate_r18 import KEYS

OUTPUT=Path('results/phase2/tenfold_mixed_matrixfree_probe_cost_r21.json')

def capped_action(a,b,c,u,v,z):
    """Qz whereQ_jk=v_jv_k sum_{i distinctj,k}u_i Sym(a_i,b_j,c_k)."""
    sa,sb,sc=[fnp.sum(u[:,None]*x,axis=0) for x in (a,b,c)]
    vb,vc,va=[x*v[:,None] for x in (b,c,a)]
    pairs=((vb*(sa-2*u[:,None]*a),vc),
           (vc*(sb-2*u[:,None]*b),va),
           (va*(sc-2*u[:,None]*c),vb))
    out=fnp.zeros(a.shape[0],dtype=a.dtype)
    diagonal=fnp.zeros(a.shape[0],dtype=a.dtype)
    for left,right in pairs:
        out=out+left@(right.T@z)+right@(left.T@z)
        diagonal=diagonal+fnp.sum(left*right,axis=1)/3
    return out/6-diagonal*z

def prepare_blocks(src):
    a,p,z,ell,w2,s,e,c1,c2,y,f1,f2,r1,r2=[src[k] for k in KEYS]
    return ((3*a+f1@r1.T,p,a*w2[None,:]+f2@r2.T),
            (p*s[None,:]+3*a*e[None,:]+z@ell.T,p,p),
            (a*c1[None,:]+p*c2[None,:],p,fnp.ones((1,p.shape[1]),dtype=p.dtype)*y[:,None]))

def main():
    assert not OUTPUT.exists()
    helper=Path('estimators/tenfold_fullcore_li14_r01/estimator.py')
    spec=importlib.util.spec_from_file_location('probe21densecheck',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(21114)
    tiny=[]
    for n in (4,8,32):
        a,b,c=[rng.normal(size=(n,3)).astype(np.float64) for _ in range(3)]
        u,v,z=[rng.normal(size=n).astype(np.float64) for _ in range(3)]
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            af,bf,cf,uf,vf,zf=map(fnp.asarray,(a,b,c,u,v,z))
            got=capped_action(af,bf,cf,uf,vf,zf)
            q=module.capped_three_products(af,bf,cf,uf,vf)
            expected=q@zf
        error=float(np.max(abs(np.asarray(got)-np.asarray(expected))))
        assert error<1e-11
        tiny.append({'width':n,'matrixfree_vs_independentdense_action_max_error':error,'legal_flops':ctx.summary_dict()['flops_used']})
    meta=json.loads(Path('results/phase2/ablation_v30_067_r18_all_source_capture.json').read_text())
    sm=json.loads(Path('results/phase2/ablation_v30_067_r24_real_k31_capture.json').read_text())
    path=Path(sm['capture']);assert hashlib.sha256(path.read_bytes()).hexdigest()==sm['capture_sha256']
    with np.load(path) as src:
        c=src['C_off'].astype(np.float64);c=(c+c.T)/2;np.fill_diagonal(c,0)
        mu,sigma=[src[k].astype(np.float64) for k in ('mu','sigma')]
    ds=load_dataset('benchmark/data/mini_first8',split='mini').select_columns(['mlp_id','mlp_name','mlp_seed','weights'])
    weights=np.asarray(mlp_at(ds,0).weights[15]).T.astype(np.float64)
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as prepctx:
        cf=fnp.asarray(c)
        w=legal_wicks(fnp.asarray(mu),fnp.asarray(sigma))
        w1,w2=w[(1,1)],w[(2,1)]
        zf=fnp.asarray(np.random.default_rng(11919).choice((-1.,1.),size=1024))
        cz=cf@zf
        blocks=[]
        for name,sha in sorted(meta['source_files_sha256'].items()):
            p=Path(name);assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
            with np.load(p) as src:
                data={k:fnp.asarray(src[k]).astype(fnp.float64) for k in KEYS}
            blocks.extend(prepare_blocks(data))
    rows=[]
    for output in (0,1,2,3):
        start=time.perf_counter()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            wf=fnp.asarray(weights[output]);u=wf*w1;v=wf*w2
            qz=fnp.zeros(1024,dtype=fnp.float64)
            for a,b,cleg in blocks:
                qz=qz+capped_action(a,b,cleg,u,v,zf)
            probe=3*fnp.sum(cz*qz)
        rows.append({'output':output,'same_fixedz_matrixfree_probe':float(probe),'probe_flops_after_sharedprep':ctx.summary_dict()['flops_used'],'wall_seconds':time.perf_counter()-start})
        print(rows[-1],flush=True)
    knownq=json.loads(Path('results/phase2/tenfold_mixed_fullhistory_variance_gate_r19.json').read_text())
    err=abs(rows[0]['same_fixedz_matrixfree_probe']-knownq['one_probe_value_not_accuracy'])
    assert err<1e-10*max(abs(knownq['one_probe_value_not_accuracy']),1e-8)
    # Bound oneoutput0 only; excludes1023outputs/D21/layerhistory/prep/resampling.
    probes=knownq['optimistic_budget_after_parent_flops']//rows[0]['probe_flops_after_sharedprep']
    relative_se=float(np.sqrt(knownq['single_probe_variance']/max(probes,1))/abs(knownq['signal']))
    report={'version':'tenfold.mixed.matrixfree.probe.r21','complete':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'small_contract':tiny,'rows':rows,'sharedprep_legal_flops':prepctx.summary_dict()['flops_used'],
            'fixedoutput0_matrixfree_vs_r19_knownq_probe_error':err,
            'fixedoutput0_probes_required_for_10pct_relative_se':knownq['probes_required_for_10pct_relative_se'],
            'optimistic_onlyoutput0_probes_after_parent':probes,'optimistic_onlyoutput0_se_to_signal':relative_se,
            'passes_optimistic_onlyoutput0_se_gate':bool(relative_se<=.1),
            'network_id':'0','layer':14,'all14_sources_all1024_columns':True,
            'data_commit':meta['dataset_commit'],'state_capture_sha256':sm['capture_sha256'],'source_capture_sha256':meta['capture_sha256'],
            'no_reference_or_target_access':True,'raw_mse':None,'adjusted_score':None,
            'conclusion':'Use actualmatrixfreeprobefee rather than impossibleknownQcheapfee; productioncandidate not established',
            'limitations':['Probe outputs0-3 are fixed; no selection bysignal orreference','OnlynextD3functional; D21/history cost omitted',
                           'Sharedprep fee excluded from optimisticbound; actual is worse','Wholeoutput1024 probe cost at least1024fixedoutputcost before history, no sharedCz saves source contractions',
                           'Float64 diagnosed. No unmeasuredfloat32fee/variance extrapolation used aspass']}
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    main()

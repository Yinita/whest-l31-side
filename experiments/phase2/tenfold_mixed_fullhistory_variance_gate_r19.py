"""Exact target-free Rademacher noise gate for allhistory mixed orbit.

Fixed publicnetwork0 layer14/output0 only. Matrix cap is diagnostic, not candidate.
"""
# ruff: noqa: E402
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key]="1"
import hashlib
import importlib.util
import itertools
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

OUTPUT=Path('results/phase2/tenfold_mixed_fullhistory_variance_gate_r19.json')
KEYS=('a','p','z','ell','w2','s','e','c1','c2','y','f1','f2','r1','r2')

def formula_contract():
    rng=np.random.default_rng(67119)
    rows=[]
    for n in (4,7):
        c,q=[rng.normal(size=(n,n)) for _ in range(2)]
        c,q=(c+c.T)/2,(q+q.T)/2
        np.fill_diagonal(c,0);np.fill_diagonal(q,0)
        b=3*c@q
        z=np.array(list(itertools.product((-1.,1.),repeat=n)))
        sample=np.einsum('si,ij,sj->s',z,b,z)
        sym=(b+b.T)/2;np.fill_diagonal(sym,0)
        variance=float(2*np.sum(sym*sym))
        assert abs(variance-float(sample.var()))<1e-10
        assert abs(float(sample.mean())-float(np.trace(b)))<1e-10
        rows.append({'width':n,'trace':float(np.trace(b)),'variance_exact':variance,'variance_enumerated':float(sample.var())})
    return rows

def main():
    assert not OUTPUT.exists()
    toy=formula_contract()
    meta=json.loads(Path('results/phase2/ablation_v30_067_r18_all_source_capture.json').read_text())
    sm=json.loads(Path('results/phase2/ablation_v30_067_r24_real_k31_capture.json').read_text())
    cap=Path(sm['capture'])
    assert hashlib.sha256(cap.read_bytes()).hexdigest()==sm['capture_sha256']
    with np.load(cap) as src:
        c=src['C_off'].astype(np.float64);c=(c+c.T)/2;np.fill_diagonal(c,0)
        mu,sigma=[src[k].astype(np.float64) for k in ('mu','sigma')]
    dataset=load_dataset('benchmark/data/mini_first8',split='mini').select_columns(['mlp_id','mlp_name','mlp_seed','weights'])
    weight=np.asarray(mlp_at(dataset,0).weights[15]).T.astype(np.float64)[0]
    helper=Path('estimators/tenfold_fullcore_li14_r01/estimator.py')
    spec=importlib.util.spec_from_file_location('fullsourcecap_r19',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    start=time.perf_counter()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        w=legal_wicks(fnp.asarray(mu),fnp.asarray(sigma))
        u=fnp.asarray(weight)*w[(1,1)];v=fnp.asarray(weight)*w[(2,1)]
        q=fnp.zeros((1024,1024),dtype=fnp.float64)
        source_cost=[]
        for name,sha in sorted(meta['source_files_sha256'].items()):
            path=Path(name);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
            with np.load(path) as src:
                values=[src[k].astype(np.float64) for k in KEYS]
            before=ctx.summary_dict()['flops_used']
            value=module.capped_structured_source(*map(fnp.asarray,values),u,v)
            q=q+value
            source_cost.append(ctx.summary_dict()['flops_used']-before)
        b=3*fnp.asarray(c)@q
        signal=fnp.sum(fnp.diag(b))
        sym=(b+b.T)/2
        fnp.fill_diagonal(sym,0)
        variance=2*fnp.sum(sym*sym)
        # legal direct 1fixedoutput probe against alreadyformed Q diagnostic.
        z=fnp.asarray(np.random.default_rng(11919).choice((-1.,1.),size=1024))
        before_probe=ctx.summary_dict()['flops_used']
        probe=3*fnp.sum((fnp.asarray(c)@z)*(q@z))
        knownq_probe_flops=ctx.summary_dict()['flops_used']-before_probe
    signal,variance=float(signal),float(variance)
    previous=json.loads(Path('results/phase2/tenfold_mixed_fullhistory_readout_gate_r18.json').read_text())['rows'][0]['mixed_next_d3_signal']
    signal_error=abs(signal-previous)
    assert signal_error<1e-10*max(abs(previous),1e-8)
    required=int(np.ceil(variance/max((0.1*abs(signal))**2,1e-30)))
    # Extremelyoptimistic singleoutput budget upper; excludes Qformation, allother
    # output/ages and all source construction. Failure of this is decisive.
    parent_flops=428219415145
    remaining=2**41-parent_flops
    maximum=remaining//knownq_probe_flops
    se=np.sqrt(variance/maximum)
    report={'version':'tenfold.mixed.fullhistory.variance.r19','complete':True,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'capped_helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),
            'scope':'Fixedoutput0 exactRademacher variance for full14source mixedK111*edge functional; diagnosticdenseQ only',
            'network_id':'0','layer':14,'output':0,'data_commit':meta['dataset_commit'],
            'state_capture_sha256':sm['capture_sha256'],'allsource_capture_sha256':meta['capture_sha256'],
            'small_exhaustive_probe_contract':toy,'signal':signal,'signal_vs_r18_abs_error':signal_error,
            'single_probe_variance':variance,'single_probe_se_to_signal':float(np.sqrt(variance)/abs(signal)),
            'prespecified_relative_se_gate':0.1,'probes_required_for_10pct_relative_se':required,
            'legal_knownQ_singleoutput_probe_flops':knownq_probe_flops,
            'optimistic_budget_after_parent_flops':remaining,'optimistic_max_probes_singleoutput_knownQ':maximum,
            'optimistic_se_to_signal':float(se/abs(signal)),
            'passes_even_optimistic_singleoutput_gate':bool(se/abs(signal)<=.1),
            'allsource_cap_flops':sum(source_cost),'diagnostic_legal_flops':ctx.summary_dict()['flops_used'],
            'diagnostic_wall_seconds':time.perf_counter()-start,'one_probe_value_not_accuracy':float(probe),
            'no_reference_or_target_access':True,'raw_mse':None,'adjusted_score':None,
            'limitations':['KnownQ cheapprobe is not a productionimplementation; actualQ formation peroutput and history is prohibitive',
                           'Budget bound omits Qformation,1023otheroutputs, sourceconstruction and history; passing is insufficient',
                           'Fullrank representedsources inherit parentapproximation; no truthalignment orscore claim',
                           'Fullmatrixfloat64 diagnostic within2**41 is not120/.4s candidateacceptance']}
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':
    main()

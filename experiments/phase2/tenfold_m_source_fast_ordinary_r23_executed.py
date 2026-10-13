"""Exact low-cost ordinary M-component outputfunctionals, no denseM/H/tensor."""
# ruff: noqa: E402
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
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

OUTPUT=Path('results/phase2/tenfold_m_source_fast_ordinary_r23.json')


def ordinary_components(gu,a,p,z,ell,s,e):
    """gu=weights*currentw1, fixedoutputs byn, exactallsourceM contraction."""
    cp,ca,cz=gu@p,gu@a,gu@z
    square=cp*cp
    rank=z.shape[1]-2
    return (fnp.sum(cp*s[None,:]*square,axis=1),
            fnp.sum(3*ca*e[None,:]*square,axis=1),
            fnp.sum((cz[:,:rank]@ell[:,:rank].T)*square,axis=1),
            fnp.sum((cz[:,rank:rank+1]@ell[:,rank:rank+1].T)*square,axis=1))


def main():
    assert not OUTPUT.exists()
    frozen=json.loads(Path('results/phase2/tenfold_m_source_age_readout_r22.json').read_text())
    meta=json.loads(Path('results/phase2/ablation_v30_067_r18_all_source_capture.json').read_text())
    sm=json.loads(Path('results/phase2/ablation_v30_067_r24_real_k31_capture.json').read_text())
    path=Path(sm['capture']);assert hashlib.sha256(path.read_bytes()).hexdigest()==sm['capture_sha256']
    with np.load(path) as src:
        mu,sigma=[src[k].astype(np.float64) for k in ('mu','sigma')]
    ds=load_dataset('benchmark/data/mini_first8',split='mini').select_columns(['mlp_id','mlp_name','mlp_seed','weights'])
    weights=np.asarray(mlp_at(ds,0).weights[15]).T.astype(np.float64)[:4]
    sourcekeys=('a','p','z','ell','s','e')
    start=time.perf_counter()
    rows=[]
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        wick=legal_wicks(fnp.asarray(mu),fnp.asarray(sigma))[(1,1)]
        gu=fnp.asarray(weights)*wick[None,:]
        totals=[fnp.zeros(4,dtype=fnp.float64) for _ in range(4)]
        for index,(name,sha) in enumerate(sorted(meta['source_files_sha256'].items())):
            p=Path(name);assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
            with np.load(p) as src:
                data=[fnp.asarray(src[k]).astype(fnp.float64) for k in sourcekeys]
            before=ctx.summary_dict()['flops_used']
            values=ordinary_components(gu,*data)
            fee=ctx.summary_dict()['flops_used']-before
            for i,value in enumerate(values):
                totals[i]=totals[i]+value
            expected=np.array([[out['source_rows'][index]['components'][comp]['ordinary_gain_next_d3']
                                for out in frozen['outputs']] for comp in
                               ('diagonal','ssep','rank16_residual','feedM_rank1')])
            got=np.array([np.asarray(v) for v in values])
            error=float(np.max(abs(got-expected)))
            assert error<2e-12
            rows.append({'source_index':index,'age_at_pre_layer14':14-index,'fouroutput_components':got.tolist(),
                         'independent_previous_matrixfunctional_max_error':error,
                         'legal_fouroutput_readout_flops_excluding_conversion':fee})
    report={'version':'tenfold.Msource.fastordinary.r23','complete':True,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_rows':rows,'component_order':['diagonal','ssep','rank16_residual','feedM_rank1'],
            'fixed_outputs':[0,1,2,3],'component_totals_fouroutputs':[np.asarray(x).tolist() for x in totals],
            'legal_full_diagnostic_flops':ctx.summary_dict()['flops_used'],
            'legal_fouroutput_readout_flops_excluding_conversion':sum(r['legal_fouroutput_readout_flops_excluding_conversion'] for r in rows),
            'wall_seconds':time.perf_counter()-start,'no_reference_or_target_access':True,
            'data_commit':meta['dataset_commit'],'parent_source_sha256':meta['source_sha256'],
            'allsource_capture_sha256':meta['capture_sha256'],'state_capture_sha256':sm['capture_sha256'],
            'raw_mse':None,'adjusted_score':None,
            'scope':'Exactbatchedfixed4outputordinaryMfunctional; reusescurrentgu@P/gu@A andthinZ; no denseM construction',
            'limitations':['OrdinaryMfunctional already represented; adding it toparent mean would doublecount',
                           'CachedV29 differsH18; noMCalignment oraccuracy claim',
                           'Scalarfixedoutputs only; completeD21 and allhistoryclosure needdifferentrepresentation',
                           'Actualchargedfee iscomponentonly, no wholepredict/RPC acceptance']}
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_rows'},indent=2,allow_nan=False))

if __name__=='__main__':
    main()

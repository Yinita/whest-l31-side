"""Full14source M decomposition, ordinary and omittedmixed nextD3 functionals.

Fixednetwork0 outputs0-3; all1024birthcolumns; no reference/MC target access.
"""
# ruff: noqa: E402
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
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
from experiments.phase2.tenfold_source_scalar_projection import sym_source_projection

OUTPUT=Path('results/phase2/tenfold_m_source_age_readout_r22.json')


def ordinary_m_readout(m,p,gain_weight):
    """Fullordinary contraction Sym(M,P,P)[g,g,g], including repeatedrows."""
    cm=gain_weight@m
    cp=gain_weight@p
    return fnp.sum(cm*cp*cp)


def tiny_contract():
    rng=np.random.default_rng(22114)
    rows=[]
    for n in (4,8):
        rank=n
        m,p=[rng.normal(size=(n,rank)) for _ in range(2)]
        u,v,g=[rng.normal(size=n) for _ in range(3)]
        raw=rng.normal(size=(n,n));c=(raw+raw.T)/2;np.fill_diagonal(c,0)
        tensor=(np.einsum('it,jt,kt->ijk',m,p,p)
                +np.einsum('it,jt,kt->ijk',p,m,p)
                +np.einsum('it,jt,kt->ijk',p,p,m))/3
        mixed=0.
        for i,j,k in itertools.permutations(range(n),3):
            mixed+=tensor[i,j,k]*(c[i,j]*v[i]*v[j]*u[k]
                                  +c[i,k]*v[i]*u[j]*v[k]
                                  +c[j,k]*u[i]*v[j]*v[k])
        expected_plain=np.einsum('ijk,i,j,k',tensor,g,g,g)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            mf,pf,uf,vf,gf,cf=map(fnp.asarray,(m,p,u,v,g,c))
            h=cf*vf[:,None]*vf[None,:]
            hp=h@pf;hm=h@mf
            actual_mixed=3*sym_source_projection(mf,pf,pf,hm,hp,hp,uf)
            actual_plain=ordinary_m_readout(mf,pf,gf)
        em,ep=abs(float(actual_mixed)-mixed),abs(float(actual_plain)-expected_plain)
        assert max(em,ep)<1e-10*max(abs(mixed),abs(expected_plain),1)
        rows.append({'width':n,'mixed_strictdistinct_error':em,'ordinary_allindices_error':ep,'legal_flops':ctx.summary_dict()['flops_used']})
    return rows


def main():
    assert not OUTPUT.exists()
    toy=tiny_contract()
    meta=json.loads(Path('results/phase2/ablation_v30_067_r18_all_source_capture.json').read_text())
    sm=json.loads(Path('results/phase2/ablation_v30_067_r24_real_k31_capture.json').read_text())
    path=Path(sm['capture']);assert hashlib.sha256(path.read_bytes()).hexdigest()==sm['capture_sha256']
    with np.load(path) as src:
        c=src['C_off'].astype(np.float64);c=(c+c.T)/2;np.fill_diagonal(c,0)
        mu,sigma=[src[k].astype(np.float64) for k in ('mu','sigma')]
    dataset=load_dataset('benchmark/data/mini_first8',split='mini').select_columns(['mlp_id','mlp_name','mlp_seed','weights'])
    weights=np.asarray(mlp_at(dataset,0).weights[15]).T.astype(np.float64)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        w=legal_wicks(fnp.asarray(mu),fnp.asarray(sigma))
        wick1,wick2=[np.asarray(w[(k,1)]).copy() for k in (1,2)]
    sources=[]
    keys=('a','p','z','ell','s','e','w2','c1','c2','y','f1','f2','r1','r2')
    for name,sha in sorted(meta['source_files_sha256'].items()):
        path=Path(name);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        with np.load(path) as src:
            sources.append({k:src[k].astype(np.float64) for k in keys})
    outputs=[]
    for output in (0,1,2,3):
        start=time.perf_counter()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            u=fnp.asarray(weights[output]*wick1)
            v=fnp.asarray(weights[output]*wick2)
            h=fnp.asarray(c)*v[:,None]*v[None,:]
            rows=[]
            for index,source in enumerate(sources):
                data={k:fnp.asarray(value) for k,value in source.items()}
                a,p,z,ell,s,e=[data[k] for k in ('a','p','z','ell','s','e')]
                rank=z.shape[1]-2
                # Shared H@A/H@P retains exactrepresentedsource, no agerank truncation.
                before=ctx.summary_dict()['flops_used']
                ha,hp=h@a,h@p
                shared_fee=ctx.summary_dict()['flops_used']-before
                mdiag=p*s[None,:]
                mssep=3*a*e[None,:]
                mrank=z[:,:rank]@ell[:,:rank].T
                mfeed=z[:,rank:rank+1]@ell[:,rank:rank+1].T
                mats=(mdiag,mssep,mrank,mfeed)
                hms=(hp*s[None,:],3*ha*e[None,:],
                     (h@z[:,:rank])@ell[:,:rank].T,
                     (h@z[:,rank:rank+1])@ell[:,rank:rank+1].T)
                names=('diagonal','ssep','rank16_residual','feedM_rank1')
                vals={}
                for name,m,hm in zip(names,mats,hms):
                    before=ctx.summary_dict()['flops_used']
                    plain=ordinary_m_readout(m,p,u)
                    mixed=3*sym_source_projection(m,p,p,hm,hp,hp,u)
                    vals[name]={'ordinary_gain_next_d3':float(plain),'omitted_mixed_next_d3':float(mixed),
                                'legal_readout_flops_excluding_shared_products':ctx.summary_dict()['flops_used']-before}
                # Whole represented nextD3 provides a component direction comparator
                # only, not true missingD3. No targets or empiricalfitting.
                x1=3*a+data['f1']@data['r1'].T
                y1=a*data['w2'][None,:]+data['f2']@data['r2'].T
                x3=a*data['c1'][None,:]+p*data['c2'][None,:]
                cp=u@p
                full=(fnp.sum((u@x1)*cp*(u@y1))
                      +sum(vv['ordinary_gain_next_d3'] for vv in vals.values())
                      +fnp.sum((u@x3)*cp)*fnp.sum(u*data['y']))
                rows.append({'source_index':index,'born_layer':index,'age_at_pre_layer14':14-index,
                             'source_columns':1024,'represented_residual_rank':rank,
                             'shared_HA_HP_flops':shared_fee,'components':vals,
                             'complete_represented_ordinary_next_d3':float(full)})
        totals={name:{field:sum(row['components'][name][field] for row in rows)
                      for field in ('ordinary_gain_next_d3','omitted_mixed_next_d3')}
                for name in ('diagonal','ssep','rank16_residual','feedM_rank1')}
        # Fixed agebins for presentation, not computation/tensortruncation.
        bins={}
        for label,lo,hi in (('age1to4',1,4),('age5to8',5,8),('age9to14',9,14)):
            selected=[row for row in rows if lo<=row['age_at_pre_layer14']<=hi]
            bins[label]={name:{field:sum(row['components'][name][field] for row in selected)
                               for field in ('ordinary_gain_next_d3','omitted_mixed_next_d3')}
                         for name in totals}
        plain=sum(total['ordinary_gain_next_d3'] for total in totals.values())
        mixed=sum(total['omitted_mixed_next_d3'] for total in totals.values())
        row={'output':output,'source_rows':rows,'fixed_age_bins':bins,'component_totals':totals,
             'ordinary_M_next_d3':plain,'omitted_mixed_M_next_d3':mixed,
             'complete_represented_ordinary_next_d3':sum(rr['complete_represented_ordinary_next_d3'] for rr in rows),
             'mixed_M_cancellation_fraction':1-abs(mixed)/max(sum(abs(x['omitted_mixed_next_d3']) for rr in rows for x in rr['components'].values()),1e-30),
             'ordinary_M_cancellation_fraction':1-abs(plain)/max(sum(abs(x['ordinary_gain_next_d3']) for rr in rows for x in rr['components'].values()),1e-30),
             'legal_diagnostic_flops':ctx.summary_dict()['flops_used'],'wall_seconds':time.perf_counter()-start}
        outputs.append(row)
        print({k:v for k,v in row.items() if k not in ('source_rows','fixed_age_bins')},flush=True)
    result={'version':'tenfold.Msource.age.readout.r22','complete':True,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'sym_source_projection_sha256':hashlib.sha256(Path('experiments/phase2/tenfold_source_scalar_projection.py').read_bytes()).hexdigest(),
            'toy_contract':toy,'outputs':outputs,'network_id':'0','current_pre_layer':14,'next_layer':15,
            'parent_source_sha256':meta['source_sha256'],'state_capture_sha256':sm['capture_sha256'],
            'allsource_capture_sha256':meta['capture_sha256'],'data_commit':meta['dataset_commit'],
            'raw_mse':None,'adjusted_score':None,'no_reference_or_target_access':True,
            'scope':'All14actualV29representedM source components, all1024columns; exactordinary and omittedmixed nextD3 functionals fixedoutputs0-3',
            'limitations':['V29 cache differsfromH18 source and seedtrace; no MCdirection orsuccess claim',
                           'Ordinaryfunctional includesrepeatedcurrentindices; mixedprojection excludesbirthdiagramrepeatedindices correctly',
                           'Rank16component is representedretainedM only, not full discardedRres; rootr20 separatelyrestores discardedpart',
                           'Source agebins onlysummarizeexactallsource values; no pruning/selection',
                           'Float64 diagnostics not wholecandidate billing/resourceacceptance']}
    OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    main()

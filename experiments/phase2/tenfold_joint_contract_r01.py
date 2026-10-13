"""Target-free small-row parity gate for complete core to scalar projection."""
import hashlib
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    output = Path('results/phase2/tenfold_joint_contract_r01.json')
    assert not output.exists()
    paths = [Path('estimators/tenfold_fullcore_li14_r01/estimator.py'),
             Path('estimators/tenfold_corelambda_li14_r02/estimator.py')]
    modules = [load(p, 'joint_'+str(i)) for i, p in enumerate(paths)]
    metadata = json.loads(Path('results/phase2/ablation_v30_067_r35_real_k4_core_capture.json').read_text())
    cp = Path(metadata['capture'])
    assert hashlib.sha256(cp.read_bytes()).hexdigest() == metadata['capture_sha256']
    with np.load(cp) as src:
        state = {k: src[k].copy() for k in src.files}
    source_meta = json.loads(Path('results/phase2/ablation_v30_067_r18_all_source_capture.json').read_text())
    sources = []
    for p, sha in source_meta['source_files_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
        with np.load(p) as src:
            sources.append({k: src[k].copy() for k in src.files})
    rows = []
    for n in (8, 32):
        results, costs = [], []
        for i, mod in enumerate(modules):
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                vals = [fnp.asarray(state[k][:n, :n] if state[k].ndim == 2 else state[k][:n])
                        for k in ('mu','sigma','C_off','D3','D21','g4row','wk4m','wk431',
                                  'pk1v','pk2v','pk21','pk11')]
                vals += [float(state['lam_prev']), float(state['METRIC_C'])]
                for k in ('a','p','z','ell','w2','s','e','c1','c2'):
                    vs = [x[k][:n] if k in ('a','p','z') else x[k] for x in sources]
                    vals.append(fnp.stack([fnp.asarray(v) for v in vs]))
                vals.append(fnp.stack([fnp.asarray(np.concatenate((x['f1'][:n],x['f2'][:n]),axis=1)) for x in sources]))
                vals += [fnp.stack([fnp.asarray(x[k]) for x in sources]) for k in ('r1','r2')]
                vals += [sources[0]['z'].shape[1]-2,sources[0]['f1'].shape[1]]
                got = mod.tenfold_fullcore(*vals)
                if i == 0:
                    cout = (vals[11].astype(fnp.float64)+vals[11].T.astype(fnp.float64))/2
                    got = fnp.sum(got*cout)/fnp.maximum(fnp.sum(cout*cout),1e-20)
            results.append(float(got))
            costs.append(ctx.summary_dict()['flops_used'])
        delta = abs(results[1]-results[0])
        assert delta < 2e-5*max(abs(results[0]),1e-10)+1e-10,(n,results,delta)
        rows.append({'width':n,'all_source_slots':len(sources),'all_birth_columns':1024,
                     'before_lambda':results[0],'after_lambda':results[1],
                     'lambda_absolute_delta':delta,'before_after_flops':costs})
    report = {'complete':True,'version':'tenfold.joint.contract.r01','rows':rows,
              'candidate_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'scope':'target-free V29 captured small-row joint core versus direct scalar numerator; no whole-network score',
              'source_state':'Actual V29 public network0 layer14; all14 slots and all1024 columns',
              'precision_difference':'r01 float32 CP construction; r02 float64 CP construction and float32 final lambda',
              'no_adjusted_score':True}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

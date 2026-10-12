"""Targetfree actuallayer14 triangle signal exact variance on fixedoutputs."""
# ruff: noqa: E402
import os

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path

import numpy as np
from whestbench import load_dataset, mlp_at


def main():
    report=Path('results/phase2/ablation_v30_061_r03_actual_state_noise.json')
    assert not report.exists()
    state=dict(np.load('benchmark/data/v29_real_source_v30_059_r05/network_0.npz'))
    c=state['covariance'].astype('float64')
    np.fill_diagonal(c,0)
    data=load_dataset('benchmark/data/mini_first8',split='mini').select_columns(['mlp_id','mlp_name','mlp_seed','weights'])
    w=np.asarray(mlp_at(data,0).weights[15]).T.astype('float64')
    rows=[]
    for output in (0,1,2,3):
        a=w[output]*state['w2']
        matrix=a[:,None]*c
        b=matrix@matrix@matrix
        signal=float(np.trace(b))
        sym=(b+b.T)/2
        np.fill_diagonal(sym,0)
        variance=float(2*np.sum(sym*sym))
        maximum_probes=(2**41-587262754287)//(6*1024**3)
        se=float(np.sqrt(variance/maximum_probes))
        rows.append(dict(output=output,trace_signal=signal,probe_variance=variance,optimistic_budget_probes=maximum_probes,standard_error=se,se_to_signal=se/max(abs(signal),1e-30)))
    report.write_text(json.dumps(dict(rows=rows,scope='Offline exactmatrix powers of actualpublicnetwork0 layer14state, fixedoutput0-3 only, no targets; optimisticprobe budget excludes extras and sourceconstruction; not wholecandidate'),indent=2)+'\n')
    print(rows)


if __name__=='__main__':
    main()

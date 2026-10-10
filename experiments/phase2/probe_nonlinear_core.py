"""Exact finite-support ReLU production oracle validates sliced harmonic core.

This is an offline bookkeeping check, not the truncated kprop diagram algorithm.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from probe_k211_conversion import cumulant


def main():
    rng = np.random.default_rng(23)
    rows = []
    for n in (4, 6):
        inputs = rng.normal(size=(n, 41))
        weights = rng.uniform(size=41)
        weights /= weights.sum()
        layer = rng.normal(size=(n, n)) / np.sqrt(n)
        x = np.maximum(layer @ inputs + rng.normal(size=(n, 1)) * 0.3, 0)
        means = x @ weights
        centered = x-means[:, None]
        covariance = (centered*weights)@centered.T
        # Dense independent fourth cumulant, including all repeated-index patterns.
        fourth = np.empty((n,n,n,n))
        for i,j,k,last_axis in itertools.product(range(n), repeat=4):
            fourth[i,j,k,last_axis] = cumulant([x[i],x[j],x[k],x[last_axis]], weights)
        direct_trace = np.einsum('ijkk->ij', fourth)
        sliced_trace = np.zeros((n,n))
        conversion_errors = []
        for i in range(n):
            sliced_trace[i,i] = cumulant([x[i]]*4,weights)
            for k in range(n):
                if k != i:
                    sliced_trace[i,i] += cumulant([x[i],x[i],x[k],x[k]],weights)
            for j in range(n):
                if i == j:
                    continue
                # K31 obtained independently by exact power-cumulant conversion.
                def k31(a,b):
                    pk31 = cumulant([x[a]**3,x[b]],weights)
                    pk21 = cumulant([x[a]**2,x[b]],weights)
                    return pk31-3*means[a]*pk21+(6*means[a]**2-3*np.dot(weights,x[a]**2))*covariance[a,b]
                sliced_trace[i,j] = k31(i,j)+k31(j,i)
                for k in range(n):
                    if k in (i,j):
                        continue
                    converted = (cumulant([x[k]**2,x[i],x[j]],weights)
                        -2*means[k]*cumulant([x[k],x[i],x[j]],weights)
                        -2*covariance[k,i]*covariance[k,j])
                    conversion_errors.append(abs(converted-fourth[k,k,i,j]))
                    sliced_trace[i,j] += converted
        def project(trace):
            return 6*trace/(n+4)-3*np.trace(trace)*np.eye(n)/((n+2)*(n+4))
        trace_error = float(np.max(np.abs(direct_trace-sliced_trace)))
        core_error = float(np.max(np.abs(project(direct_trace)-project(sliced_trace))))
        assert max(trace_error,core_error,max(conversion_errors)) < 2e-11
        rows.append({'n':n,'support_points':41,'trace_max_error':trace_error,'core_max_error':core_error,'k211_conversion_max_error':float(max(conversion_errors))})
    result = {'mode':'offline exact finite-support nonlinear production oracle; not deployable candidate',
        'seed':23,'activation':'ReLU','source':'paper S3.3 power-cumulant conversion and S4.2 harmonic projection',
        'rows':rows,'validated':['Exact ReLU moments to K31/K211 conversion','Complete repeated-index partition of fourth-order trace','Identity-metric harmonic core projection'],
        'not_validated':['Truncated Hermite/diagram production of power cumulants','Gaussian-input integration or full public network accuracy','Budget feasibility at width1024','Stage score improvement']}
    p=Path('results/phase2/nonlinear_core_exact_oracle.json')
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(result,indent=2)+'\n')
    print('Two exact nonlinear slice/core checks passed',rows)


if __name__=='__main__':
    main()

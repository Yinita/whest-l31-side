"""Fixed analytic zeroalpha covariance Pade tail lawful operator contract."""
import json
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rho = np.linspace(-.99,.99,101).astype('float32')
    with flops.BudgetContext(flop_budget=2**30) as ctx:
        x = fnp.array(rho)
        q = x*x
        value = q*q / (48*math.pi*(1-.3*q))
    exact = (np.sqrt(1-rho.astype('float64')**2)+(math.pi-np.arccos(rho.astype('float64')))*rho-1)/(2*math.pi)
    quadratic = rho.astype('float64')/4+rho.astype('float64')**2/(4*math.pi)
    expected = rho.astype('float64')**4/(48*math.pi*(1-.3*rho.astype('float64')**2))
    err = float(np.max(abs(np.asarray(value)-expected)))
    assert err < 1e-7
    p=Path('results/phase2/ablation_v30_035_r02_zeroalpha_charged_contract.json')
    assert not p.exists()
    p.write_text(json.dumps(dict(maximum_absolute_error=err, flops=ctx.summary_dict()['flops_used'], original_maxgap=float(np.max(abs(exact-quadratic))), pade_maxgap=float(np.max(abs(exact-quadratic-np.asarray(value)))), scope='Zeroalpha covarianceonly subset, positive denominator throughout[-1,1], no targetfitting or fullnonzero kernel claim'),indent=2)+'\n')


if __name__=='__main__':
    main()

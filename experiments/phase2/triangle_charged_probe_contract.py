"""H31 exact exhaustive expectation with legally billed batched matvec probes."""
import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    rng=np.random.default_rng(6103)
    rows=[]
    for n in (3,4,7):
        c=rng.normal(size=(n,n)).astype('float32')
        c=(c+c.T)/2
        np.fill_diagonal(c,0)
        a=rng.normal(size=(n,n)).astype('float32')
        probes=np.array(list(itertools.product([-1,1],repeat=n)),dtype='float32')
        expected=np.array([np.trace(np.linalg.matrix_power(x[:,None]*c,3)) for x in a])
        outputs=[]
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            cc,aa,z=fnp.array(c),fnp.array(a),fnp.array(probes)
            for index in range(n):
                v=z
                for _ in range(3):
                    v=(v@cc.T)*aa[index]
                outputs.append(fnp.mean(fnp.sum(z*v,axis=1)))
            actual=fnp.stack(outputs)
        # Recurrence uses C diag(a), whose trace equals diag(a) C by cyclicity.
        error=float(np.max(abs(np.asarray(actual)-expected)))
        assert np.allclose(np.asarray(actual),expected,rtol=2e-5,atol=2e-5)
        rows.append(dict(n=n,probes=len(probes),maximum_absolute_error=error,flops=ctx.summary_dict()['flops_used']))
    p=Path('results/phase2/ablation_v30_061_r02_charged_probe_contract.json')
    assert not p.exists()
    p.write_text(json.dumps(dict(rows=rows,scope='Syntheticalloutput exhaustiveRademacher expectation; legalops no densecandidatetensor. Actualstate variance/mean gain gates remain'),indent=2)+'\n')
    print(rows)


if __name__=='__main__':
    main()

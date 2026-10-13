"""Target-free exact shifted-Gaussian and Phi*Jv local response contract."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from experiments.phase2.tenfold_gaussian_mean_response_r43 import gaussian_relu_mean,exact_response,linear_response,quadratic_response


def cdf(x):return np.asarray([.5*(1+math.erf(v/math.sqrt(2))) for v in x])


def gmean(mu,sigma):
    a=mu/sigma
    return sigma*np.exp(-a*a/2)/np.sqrt(2*np.pi)+mu*cdf(a)


def run(n):
    rng=np.random.default_rng(261014221+n)
    sigma=rng.uniform(.3,2,size=n)
    alpha=rng.uniform(-2,2,size=n);mu=alpha*sigma
    w=rng.normal(size=(n,n))/np.sqrt(n);previous_error=rng.normal(size=n)*.1
    transported=w@previous_error
    transported=transported/max(np.max(abs(transported)),1e-12)
    rows=[];previous_first=previous_second=None
    scale_sequence=(.08,.04,.02,.01,.005)
    derivative_error=[]
    for delta in scale_sequence:
        h=sigma*transported*delta
        exact=gmean(mu+h,sigma)-gmean(mu,sigma)
        phi=np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi)
        first=cdf(alpha)*h;second=first+.5*phi/sigma*h*h
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            mm,ss,hh=[fnp.asarray(v) for v in (mu,sigma,h)]
            actual=exact_response(mm,ss,hh);linear=linear_response(mm,ss,hh);quadratic=quadratic_response(mm,ss,hh)
        actual=np.asarray(actual);linear=np.asarray(linear);quadratic=np.asarray(quadratic)
        exacterror=float(np.max(abs(actual-exact)));linearerror=float(np.max(abs(linear-first)));quadraticerror=float(np.max(abs(quadratic-second)))
        assert max(exacterror,linearerror,quadraticerror)<5e-13,(exacterror,linearerror,quadraticerror)
        # Dimensionless errors make sigma changes irrelevant for order audit.
        first_residual=float(np.max(abs((exact-first)/sigma)))
        second_residual=float(np.max(abs((exact-second)/sigma)))
        first_order=None if previous_first is None else math.log2(previous_first/first_residual)
        second_order=None if previous_second is None else math.log2(previous_second/second_residual)
        assert first_order is None or 1.8<first_order<2.2,(n,first_order)
        assert second_order is None or 2.7<second_order<3.3,(n,second_order)
        wrong=float(np.max(abs((h-exact)/sigma)))
        assert wrong>10*first_residual,(n,wrong,first_residual)
        rows.append(dict(delta=delta,exact_formula_error=exacterror,phi_response_error=linearerror,quadratic_formula_error=quadraticerror,firstorder_normalized_residual=first_residual,secondorder_normalized_residual=second_residual,observed_firstorder_error_order=first_order,observed_secondorder_error_order=second_order,ungated_Jv_negative_control_residual=wrong,legal_allresponses_component_cost=ctx.summary_dict()))
        previous_first,previous_second=first_residual,second_residual
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        mm,ss=fnp.asarray(mu),fnp.asarray(sigma)
        zero=exact_response(mm,ss,fnp.zeros(n,dtype=fnp.float64))
        base=gaussian_relu_mean(mm,ss)
        eps=1e-5*sigma;ee=fnp.asarray(eps)
        derivative=(gaussian_relu_mean(mm+ee,ss)-gaussian_relu_mean(mm-ee,ss))/(2*ee)
    zeroerror=float(np.max(abs(np.asarray(zero))));baseerror=float(np.max(abs(np.asarray(base)-gmean(mu,sigma))))
    derivativeerror=float(np.max(abs(np.asarray(derivative)-cdf(alpha))))
    assert zeroerror==0 and baseerror<5e-13 and derivativeerror<2e-10
    # Shift additivity uses updated base; a static baseline double response fails.
    h=sigma*transported*.04;k=-.5*h
    composed=gmean(mu+h,sigma)-gmean(mu,sigma)+gmean(mu+h+k,sigma)-gmean(mu+h,sigma)
    combined=gmean(mu+h+k,sigma)-gmean(mu,sigma)
    additiveerror=float(np.max(abs(composed-combined)))
    wrongstatic=float(np.max(abs((gmean(mu+h,sigma)-gmean(mu,sigma)+gmean(mu+k,sigma)-gmean(mu,sigma))-combined)))
    assert additiveerror<1e-14 and wrongstatic>1e-7
    return dict(width=n,zero_response_error=zeroerror,baseline_formula_error=baseerror,centered_difference_vs_Phi_error=derivativeerror,exact_composition_error=additiveerror,static_base_negative_control_error=wrongstatic,rows=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    rows=[run(n) for n in (8,32,1024)]
    paths=[Path(__file__),Path('experiments/phase2/tenfold_gaussian_mean_response_r43.py')]
    result=dict(complete=True,version='target-free-gaussian-mean-response-r43',rows=rows,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},hypothesis='preactivation h=W*previouspostmeanerror enters postReLU mean as Phi*h firstorder; exactfixedsigma Gaussian response removes quadratic remainder',truth_access=False,fit=False,no_benchmark_or_corpus_truth=True,no_scoreclaim=True,limits=['fixedsigmaGaussian response only','true nonGaussian ReLU JacobianneednotPhi','preReLUshift changes covariance/mixedhigher sources; these omitted','notwholecandidatecorrection or fullcausal stateclosure','notcompetitionresourceacceptance'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(width=r['width'],baseerror=r['baseline_formula_error'],derivativeerror=r['centered_difference_vs_Phi_error'],firstorderobserved=[x['observed_firstorder_error_order'] for x in r['rows'][1:]],secondorderobserved=[x['observed_secondorder_error_order'] for x in r['rows'][1:]]) for r in rows]))


if __name__=='__main__':main()

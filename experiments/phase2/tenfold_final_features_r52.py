"""Lawful final-only26 component, no coefficients or fitting.

RawPK1 centralization uses float64; emitted parent mean evaluates inherited
rider in original float32 before promotion to match final actual row.
"""
import flopscope.numpy as fnp

FEATURE_ORDER=('ones','mu','var','sigma','alpha','abs_alpha','phi','Phi','pk1','D3','D21norm','K4_sigma','sigma_phi','skew_H2phi','kurt_H3phi','skew2_H5phi','skewkurt_H6phi','kurt2_H7phi','skew','kurt','gate','depth_fraction','own_post_mean_over_sigma','own_post_variance_over_sigma2','own_post_kappa3_over_sigma3','own_post_kappa4_over_sigma4')


def prefeatures(mu,var,sigma,alpha,phi,cdf,pk1,d3,g4,k4sigma):
    mu,var,sigma,alpha,phi,cdf,pk1,d3,g4=[v.astype(fnp.float64) for v in (mu,var,sigma,alpha,phi,cdf,pk1,d3,g4)]
    one=fnp.ones(len(mu),dtype=fnp.float64);zero=fnp.zeros(len(mu),dtype=fnp.float64)
    sk=d3/sigma**3;ku=g4/sigma**4;hs=[one,alpha]
    for degree in range(1,7):hs.append(alpha*hs[-1]-degree*hs[-2])
    return fnp.stack([one,mu,var,sigma,alpha,fnp.abs(alpha),phi,cdf,pk1,d3,zero,one*k4sigma,sigma*phi,sk*hs[2]*phi,ku*hs[3]*phi,sk**2*hs[5]*phi,sk*ku*hs[6]*phi,ku**2*hs[7]*phi,sk,ku,cdf,one],axis=1)


def raw_post_cumulants(raw_power,inherited_delta=None):
    # Raw central quantities describe original ReLU marginal distribution;
    # deterministic mean rider is applied only to the emitted mean channel.
    p1,p2,p3,p4=[raw_power[j].astype(fnp.float64) for j in range(4)]
    mean32=raw_power[0] if inherited_delta is None else raw_power[0]+inherited_delta
    mean=mean32.astype(fnp.float64)
    var=p2-p1**2;k3=p3-3*p1*p2+2*p1**3
    k4=p4-4*p1*p3-3*p2**2+12*p1**2*p2-6*p1**4
    return fnp.stack([mean,var,k3,k4],axis=1)


def final_features(pre22,sigma,raw_power,inherited_delta=None):
    own=raw_post_cumulants(raw_power,inherited_delta)
    sig=sigma.astype(fnp.float64)
    scales=fnp.stack([sig**p for p in range(1,5)],axis=1)
    return fnp.concatenate([pre22,own/scales],axis=1)

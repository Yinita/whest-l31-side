"""Lawful 26 local closure features: originalpre22 + normalizedownpost4.

No coefficient model or fitting. This is a feature component, not a candidate.
"""
import flopscope.numpy as fnp

PREFEATURE_ORDER=('ones','mu','var','sigma','alpha','abs_alpha','phi','Phi','pk1','D3','D21norm','K4_sigma','sigma_phi','skew_H2phi','kurt_H3phi','skew2_H5phi','skewkurt_H6phi','kurt2_H7phi','skew','kurt','gate','depth_fraction')
FEATURE_ORDER=PREFEATURE_ORDER+('own_post_mean_over_sigma','own_post_variance_over_sigma2','own_post_kappa3_over_sigma3','own_post_kappa4_over_sigma4')


def prefeatures(mu,var,sigma,alpha,phi,cdf,pk1,d3,d21norm,g4,k4sigma,layer):
    mu,var,sigma,alpha,phi,cdf,pk1,d3,d21norm,g4=[v.astype(fnp.float64) for v in (mu,var,sigma,alpha,phi,cdf,pk1,d3,d21norm,g4)]
    one=fnp.ones(len(mu),dtype=fnp.float64)
    sk=d3/(sigma**3);ku=g4/(sigma**4)
    hs=[one,alpha]
    for degree in range(1,7):hs.append(alpha*hs[-1]-degree*hs[-2])
    return fnp.stack([one,mu,var,sigma,alpha,fnp.abs(alpha),phi,cdf,pk1,d3,d21norm,one*k4sigma,sigma*phi,sk*hs[2]*phi,ku*hs[3]*phi,sk**2*hs[5]*phi,sk*ku*hs[6]*phi,ku**2*hs[7]*phi,sk,ku,cdf,one*(layer/15)],axis=1)


def ownpost_features(pre,sigma,k1,k2,k3,k4,inherited_delta=None):
    # Capture before current-layer intervention. Mean includes the inherited
    # deterministic rider once; other channels are ordinary centered cumulants.
    sigma64=sigma.astype(fnp.float64)
    mean=k1.astype(fnp.float64)
    if inherited_delta is not None:mean=mean+inherited_delta.astype(fnp.float64)
    channels=[mean,k2.astype(fnp.float64),k3.astype(fnp.float64),k4.astype(fnp.float64)]
    post=fnp.stack([channel/(sigma64**degree) for degree,channel in enumerate(channels,1)],axis=1)
    return fnp.concatenate([pre,post],axis=1)

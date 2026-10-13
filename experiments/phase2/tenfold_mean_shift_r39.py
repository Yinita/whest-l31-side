"""Permitted-flops mean-only post-centering correction component.

Consumes frozen deployment coefficients; no labels or in-predict fitting.
Model may include fourchannels from research, but onlytheta[:,0] is used.
"""
import flopscope as flops
import flopscope.numpy as fnp

FEATURE_ORDER=['ones','mu','var','sigma','alpha','abs_alpha','phi','Phi','pk1','D3','D21norm','K4_sigma','sigma_phi','skew_H2phi','kurt_H3phi','skew2_H5phi','skewkurt_H6phi','kurt2_H7phi','skew','kurt','gate','depth_fraction']


def fold_mean_model(model):
 """Offline setup: foldfeature normalization into one fixed linear vector."""
 theta=model['theta'];mean=model['feature_mean'];scale=model['feature_scale']
 coef=[row[0]/s for row,s in zip(theta,scale)]
 offset=-sum(m*c for m,c in zip(mean,coef))
 coef[0]+=offset
 return coef


def features(mu,var,sigma,alpha,phi,cdf,pk1,d3,d21norm,g4,k4sigma,layer):
 n=len(mu);ones=fnp.ones(n,dtype=mu.dtype);skew=d3/sigma**3;kurt=g4/sigma**4
 h=[ones,alpha]
 for degree in range(1,7):h.append(alpha*h[-1]-degree*h[-2])
 return fnp.stack([ones,mu,var,sigma,alpha,fnp.abs(alpha),phi,cdf,pk1,d3,d21norm,
      ones*k4sigma,sigma*phi,skew*h[2]*phi,kurt*h[3]*phi,skew**2*h[5]*phi,
      skew*kurt*h[6]*phi,kurt**2*h[7]*phi,skew,kurt,cdf,ones*(layer/15)],axis=1)


def mean_shift(f,sigma,coef):return (f@coef)*sigma


def corrected_mean(k1,base_delta,shift):
 """Existing deterministic rider plus newshift once; centralcarriers unchanged."""
 return k1+shift if base_delta is None else k1+base_delta+shift

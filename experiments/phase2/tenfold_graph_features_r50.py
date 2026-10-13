"""Six fixed directional graph context features, permitted arithmetic only.

Input current pre-nonlinear C_off, ordinary K21 slice and K22 proxy wk4m,
with matching preactivation sigma. No fitting or reference data.
"""
import flopscope.numpy as fnp

FEATURE_ORDER=('rho2_rowmean','rho3_rowmean','rho_triangle_diagmean','rho_D21_left_rowmean','rho_D21_right_rowmean','rho2_K22_rowmean')


def features(cov_off,sigma,d21,k22):
    n=len(sigma);assert n>2
    s=sigma.astype(fnp.float64)
    cov=cov_off.astype(fnp.float64)
    rho=cov/(s[:,None]*s[None,:])
    rho2=rho*rho
    dleft=d21.astype(fnp.float64)/(s[:,None]**2*s[None,:])
    dright=d21.T.astype(fnp.float64)/(s[:,None]*s[None,:]**2)
    fourth=k22.astype(fnp.float64)/(s[:,None]**2*s[None,:]**2)
    rho_product=rho@rho
    # diag(rho^3)_i=sum_j (rho@rho)_ij*rho_ji, exact; no second GEMM.
    triangle=fnp.sum(rho_product*rho.T,axis=1)/((n-1)*(n-2))
    return fnp.stack([fnp.sum(rho2,axis=1)/(n-1),fnp.sum(rho2*rho,axis=1)/(n-1),triangle,fnp.sum(rho*dleft,axis=1)/(n-1),fnp.sum(rho*dright,axis=1)/(n-1),fnp.sum(rho2*fourth,axis=1)/(n-1)],axis=1)

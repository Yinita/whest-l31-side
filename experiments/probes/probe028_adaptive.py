"""Adaptive λ / analytic-hybrid / last-hidden CV on v0.2.20s."""
from __future__ import annotations
import importlib.util, math, time, sys
from pathlib import Path
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, load_dataset

ROOT = Path("/home/sti/Desktop/projects/whest-l31-side")
SPEC = importlib.util.spec_from_file_location("e20", ROOT / "experiments" / "v0220s_pooled_corr.py")
e = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(e)
B=int(2.72e11); INV=e._INV_SQRT_2PI

def apply_corr(mu_cv, mu_g, pred, zpre, alpha_g, phi_g, sig_g):
    width=pred.shape[0]
    zc=zpre-zpre.mean(0); m2=np.maximum((zc*zc).mean(0),1e-12)
    m3=(zc**3).mean(0); m4=(zc**4).mean(0); s=np.sqrt(m2)
    skew=m3/(m2*s); kurt=m4/(m2*m2)-3.0
    alpha_s=zpre.mean(0)/s; phi_s=np.exp(-0.5*alpha_s*alpha_s)/math.sqrt(2*math.pi)
    st=-(skew/6.)*phi_s; kt=(kurt/24.)*(alpha_s**2-1.)*phi_s
    gap=mu_cv-mu_g; glayer=np.full(width,float(gap.mean())); a=np.asarray(alpha_g,float)
    feats=np.stack([np.ones(width),a,a**2,a**3,a**4,np.asarray(phi_g,float),
        gap/(np.asarray(sig_g,float)+1e-8),glayer/(np.asarray(sig_g,float)+1e-8),
        st,kt,(mu_cv-pred)/(s+1e-8),(mu_g-pred)/(s+1e-8),alpha_s,phi_s],0)
    beta=np.asarray(e._CORR_BETA,float); mu=np.asarray(e._CORR_MU,float); sd=np.asarray(e._CORR_SD,float)
    return pred+s*(beta@((feats-mu.reshape(-1,1))/sd.reshape(-1,1))), s, skew, kurt, alpha_s

def stream_parts(mlp, n_half, seed):
    weights=list(mlp.weights); width=mlp.width
    x=e._rqmc_standard_normal(n_half,width,seed=seed); x=fnp.concatenate([x,-x],0)
    a1=fnp.maximum(x@weights[0],0.0); h=a1
    for w in weights[1:-1]: h=fnp.maximum(h@w,0.0)
    z=h@weights[-1]; last=fnp.maximum(z,0.0)
    return a1,last,np.asarray(z,float),np.asarray(h,float)

def cv_pool(a1,last,w0,ridge=1e-4):
    col_norm=fnp.sqrt(fnp.maximum(fnp.sum(w0*w0,0),1e-30))
    e_a1=col_norm*INV; q1=fnp.sum(a1*a1,1); e_q1=0.5*float(fnp.sum(col_norm*col_norm))
    C=fnp.concatenate([a1,q1.reshape(-1,1)],1)
    E=fnp.concatenate([e_a1,fnp.asarray([e_q1],dtype=e_a1.dtype)],0)
    return np.asarray(e._multi_cv_mean(last,C,E,ridge),float)

def predict(mlp, mode="base"):
    width=mlp.width; depth=mlp.depth; weights=list(mlp.weights)
    n2=float(width*width)
    flops_per=depth*e._MATMUL_PER_LAYER*n2+depth*width*20.0
    target=max(float(B)*e._FLOOR_FRAC-e._GAIN_RESERVE,flops_per)
    n_samples=int(max(64,min(target/flops_per,200000))); n_half=max(32,n_samples//4)
    seed0=int(mlp.seed)&0x7FFFFFFF
    a1a,la,z1,h1=stream_parts(mlp,n_half,seed0)
    a1b,lb,z2,h2=stream_parts(mlp,n_half,seed0^0xA5A5A5A5)
    a1=fnp.concatenate([a1a,a1b],0); last=fnp.concatenate([la,lb],0)
    zpre=np.concatenate([z1,z2],0); hlast=np.concatenate([h1,h2],0)
    mu1=cv_pool(a1a,la,weights[0]); mu2=cv_pool(a1b,lb,weights[0])
    mu_cv=cv_pool(a1,last,weights[0])
    mu_g_raw,ag,pg,sg=e._gain_and_last_stats(weights)
    mu_g=e._GAIN_SCALE*np.asarray(mu_g_raw,float)
    ag=np.asarray(ag,float); sg=np.asarray(sg,float)

    if mode=="base":
        pred=0.85*mu_cv+0.15*mu_g
    elif mode=="lam_adapt_gap":
        # more gain where |mu_cv-mu_g|/sig large (distrust MC)
        gap=np.abs(mu_cv-mu_g)/(sg+1e-8)
        med=float(np.median(gap))+1e-12
        w=1.0/(1.0+gap/med)  # high gap -> lower MC weight
        lam=0.70+0.25*w  # in [0.70,0.95]
        pred=lam*mu_cv+(1-lam)*mu_g
    elif mode=="lam_adapt_agree":
        dis=np.abs(mu1-mu2)/(sg+1e-8)
        med=float(np.median(dis))+1e-12
        w=1.0/(1.0+dis/med)
        lam=0.70+0.25*w
        pred=lam*mu_cv+(1-lam)*mu_g
    elif mode=="analytic_tail":
        # when |α| large, trust gain more
        w=1.0/(1.0+np.abs(ag)/2.0)
        lam=0.70+0.25*w  # extreme α -> more gain
        # actually extreme α gain is MORE accurate; MC also fine. Invert:
        # extreme |α|: prefer gain (analytic ReLU mean is exact under Gaussian)
        # small |α|: prefer MC (non-Gaussian matters near boundary)
        w_mc=1.0/(1.0+np.abs(ag))  # small |α| -> higher? wait 1/(1+0)=1 high MC near 0
        # Near 0 non-Gaussian hurts gain → want MORE MC near 0 → w_mc high when |α| small ✓
        lam=0.75+0.20*w_mc  # [0.75,0.95]
        pred=lam*mu_cv+(1-lam)*mu_g
    elif mode=="h2cv":
        # add ||h_last||^2 control; E approx from gain mean/var of last hidden
        # Approximate: propagate gain to depth-1, E[||h||^2]=sum(var+mean^2)
        mu_h,cov_h=None,None
        width=weights[0].shape[0]
        mu=fnp.zeros(width,dtype=fnp.float32); cov=fnp.eye(width,dtype=fnp.float32)
        for w in weights[:-1]:
            mu_pre=w.T@mu
            cov_pre=fnp.einsum("ij,ia,jb->ab",cov,w,w)
            var_pre=fnp.maximum(fnp.diag(cov_pre),1e-12)
            sigma=fnp.sqrt(var_pre); alpha=mu_pre/sigma
            phi=flops.stats.norm.pdf(alpha); Phi=flops.stats.norm.cdf(alpha)
            mu=mu_pre*Phi+sigma*phi
            ez2=(mu_pre*mu_pre+var_pre)*Phi+mu_pre*sigma*phi
            var_post=fnp.maximum(ez2-mu*mu,0.0)
            gain=fnp.where(sigma>1e-12,Phi,fnp.zeros_like(Phi))
            cov=fnp.multiply(fnp.outer(gain,gain),cov_pre)
            fnp.fill_diagonal(cov,var_post)
        e_h2=float(fnp.sum(mu*mu+fnp.diag(cov)))
        h2=np.sum(hlast*hlast,1,keepdims=True)
        col_norm=fnp.sqrt(fnp.maximum(fnp.sum(weights[0]*weights[0],0),1e-30))
        e_a1=col_norm*INV; q1=fnp.sum(a1*a1,1); e_q1=0.5*float(fnp.sum(col_norm*col_norm))
        C=fnp.concatenate([a1,q1.reshape(-1,1),fnp.asarray(h2,dtype=fnp.float32)],1)
        E=fnp.concatenate([e_a1,fnp.asarray([e_q1,e_h2],dtype=e_a1.dtype)],0)
        mu_cv=np.asarray(e._multi_cv_mean(last,C,E,1e-4),float)
        pred=0.85*mu_cv+0.15*mu_g
    elif mode=="pool_mean_mix":
        mu_mean=0.5*(mu1+mu2)
        mu_cv=0.7*mu_cv+0.3*mu_mean
        pred=0.85*mu_cv+0.15*mu_g
    else:
        raise ValueError(mode)

    out,s,skew,kurt,alpha_s=apply_corr(mu_cv,mu_g,pred,zpre,ag,pg,sg)
    return out, dict(s=s,skew=skew,kurt=kurt,alpha_s=alpha_s,mu_cv=mu_cv,mu_g=mu_g,ag=ag,sg=sg)

def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 40
    ds=load_dataset("aicrowd/arc-whestbench-public-2026",revision="v1-phase1",split="mini")
    mlps,Y=[],[]
    for i in range(n):
        r=ds[i]; W=[fnp.asarray(w,dtype=fnp.float32) for w in r["weights"]]
        mlps.append(MLP(width=256,depth=len(W),weights=W,seed=int(r["mlp_seed"])))
        Y.append(np.asarray(r["final_means"],np.float64))
    print(f"loaded n={n}",flush=True)

    # error structure on base
    errs=[]; metas=[]
    for mlp,y in zip(mlps,Y):
        with flops.BudgetContext(flop_budget=B,quiet=True):
            p,m=predict(mlp,"base")
        err=(p-y)**2
        errs.append(err); metas.append(m)
    E=np.stack(errs); # (n,256)
    print(f"base raw={E.mean():.4e} p90={np.percentile(E,90):.4e} p99={np.percentile(E,99):.4e}",flush=True)
    # correlate unit MSE with |α|, kurt, etc
    all_mse=E.ravel()
    for key in ["ag","sg","skew","kurt","alpha_s"]:
        v=np.concatenate([np.asarray(m[key]).ravel() for m in metas])
        if v.shape!=all_mse.shape: continue
        c=np.corrcoef(np.abs(v), all_mse)[0,1]
        print(f"  corr(|{key}|, mse)={c:.4f}",flush=True)
    gap=np.concatenate([np.abs(m["mu_cv"]-m["mu_g"]).ravel() for m in metas])
    print(f"  corr(|gap|, mse)={np.corrcoef(gap,all_mse)[0,1]:.4f}",flush=True)

    t0=time.time()
    for mode in ["base","lam_adapt_gap","lam_adapt_agree","analytic_tail","h2cv","pool_mean_mix"]:
        mses=[]
        for mlp,y in zip(mlps,Y):
            with flops.BudgetContext(flop_budget=B,quiet=True):
                p,_=predict(mlp,mode)
            mses.append(float(np.mean((p-y)**2)))
        raw=float(np.mean(mses))
        print(f"{mode:16s} raw={raw:.4e} vsbase={3.9373e-6/raw if raw else 0:.4f}x t={time.time()-t0:.0f}",flush=True)
    print("DONE",flush=True)
main()

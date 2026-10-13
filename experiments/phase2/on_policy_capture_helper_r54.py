"""Offline observer: lawful frozen prefix26 injection, stop at uncorrectedpost.

No labels/reference/fitting. Every matrix operation uses permitted fnp. Saved
NumPy copies are diagnostic instrumentation, not submission arithmetic.
"""
import numpy as np
import flopscope.numpy as fnp
from tenfold_local_features_r48 import prefeatures,ownpost_features


class CaptureComplete(Exception):
    pass


def updated_diagonals(k1,k2,k3,k4,sigma,normalized_delta,inherited_delta=None,diagnostic_audit=None):
    s=sigma.astype(fnp.float64)
    change=normalized_delta*fnp.stack([s**p for p in range(1,5)],axis=1)
    mean=k1.astype(fnp.float64)
    if inherited_delta is not None:mean=mean+inherited_delta.astype(fnp.float64)
    before_floor_mean=mean+change[:,0]
    before_floor_var=k2.astype(fnp.float64)+change[:,1]
    if diagnostic_audit is not None:
        a,b=np.asarray(before_floor_mean),np.asarray(before_floor_var)
        diagnostic_audit.update(mean_floor_count=int((a<0.).sum()),variance_floor_count=int((b<1e-10).sum()),minimum_mean_before_floor=float(a.min()),minimum_variance_before_floor=float(b.min()))
    mean=fnp.maximum(before_floor_mean,0.)
    if inherited_delta is not None:mean=mean-inherited_delta.astype(fnp.float64)
    return (mean.astype(fnp.float32),fnp.maximum(before_floor_var,1e-10).astype(fnp.float32),(k3.astype(fnp.float64)+change[:,2]).astype(fnp.float32),(k4.astype(fnp.float64)+change[:,3]).astype(fnp.float32))


class OnPolicyCapture:
    def __init__(self,width,target_layer,prefix_models):
        assert 1<=target_layer<=14
        assert set(prefix_models)==set(range(1,target_layer))
        self.width=width;self.target=target_layer;self.models=prefix_models
        self.pre={};self.current=None;self.injected=[];self.audit=[]

    def record_pre(self,state):
        layer=state['li'];assert layer not in self.pre
        if layer==0:return
        assert layer<=self.target
        d3=state['D3'];g4=state['g4row']
        dnorm=fnp.sum(fnp.abs(state['D21'].astype(fnp.float64)),axis=1)
        self.pre[layer]=prefeatures(state['mu'],state['var'],state['sigma'],state['alpha'],state['phi'],state['Phi'],state['pk1v'],d3,dnorm,g4,state['K4_sigma'],layer)

    def record_post(self,state):
        layer=state['li']
        if layer==0:
            self.audit.append(dict(layer=0,injection_skipped=True));return
        assert layer in self.pre and layer<=self.target
        delta=state['delta'];live=[state[k] for k in ('K1v','K2v','K3v','K4v')]
        before=[np.asarray(v).copy() for v in live]
        mixedbefore={k:np.asarray(state[k]).copy() for k in ('pk1v','K21','K22')}
        feature=ownpost_features(self.pre.pop(layer),state['sigma'],*live,delta)
        # Target-layer capture precedes any currenthead; constructor has already
        # rejected current/future heads, preventing accidental current fitting.
        if layer==self.target:
            inverse=np.arange(self.width) if state['inv_perm'] is None else np.asarray(state['inv_perm'])
            emitted=live[0].astype(fnp.float64)
            if delta is not None:emitted=emitted+delta.astype(fnp.float64)
            post=fnp.stack([emitted]+[v.astype(fnp.float64) for v in live[1:]],axis=1)
            s=state['sigma'].astype(fnp.float64);powers=fnp.stack([s**p for p in range(1,5)],axis=1)
            inherited=np.zeros(self.width,dtype=np.float64) if delta is None else np.asarray(delta,dtype=np.float64).copy()
            self.current=dict(layer=layer,features=np.asarray(feature)[inverse].copy(),states=np.asarray(post)[inverse].copy(),scales=np.asarray(powers)[inverse].copy(),inherited_delta=inherited[inverse].copy(),inherited_delta_present=delta is not None,inverse=np.asarray(inverse).copy())
            assert all(np.array_equal(np.asarray(v),b) for v,b in zip(live,before))
            assert all(np.array_equal(np.asarray(state[k]),v) for k,v in mixedbefore.items())
            self.audit.append(dict(layer=layer,currenthead_absent=True,capture_before_current_injection=True,current_state_unchanged=True))
            raise CaptureComplete()
        model=self.models[layer]
        normalized=((feature-model['feature_mean'])/model['feature_scale'])@model['theta']
        floor_audit={}
        values=updated_diagonals(*live,state['sigma'],normalized,delta,diagnostic_audit=floor_audit)
        for target,value in zip(live,values):fnp.copyto(target,value)
        assert all(np.array_equal(np.asarray(state[k]),v) for k,v in mixedbefore.items())
        self.injected.append(layer)
        self.audit.append(dict(layer=layer,learned_prefix_injected=True,original_pk1_mixed_pairs_unchanged=True,current26_before_injection=True,**floor_audit))

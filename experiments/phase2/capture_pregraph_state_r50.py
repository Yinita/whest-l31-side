"""Read-only offline pregraph capture observer for actual parent trajectories.

Graph arithmetic itself uses legal fnp helper. NumPy is used only by this
non-submission diagnostic observer for copies/serialization/comparison.
"""
import numpy as np
import flopscope.numpy as fnp
from tenfold_graph_features_r50 import features


class GraphCapture:
    def __init__(self,width,depth):
        self.width=width;self.depth=depth;self.pending={};self.rows={};self.audit=[]

    def record_pre(self,state):
        layer=state['li'];n=self.width
        assert layer not in self.pending and layer not in self.rows
        if layer==0:
            self.pending[layer]=np.zeros((n,6),dtype=np.float64)
            self.audit.append(dict(layer=layer,known_firstlayer_skipped=True))
            return
        assert layer<self.depth-1
        c=state['C_off'];sigma=state['sigma'];d=state['D21'];k=state['wk4m']
        assert c is not None and d is not None and k is not None
        before={key:np.asarray(v).copy() for key,v in dict(C_off=c,sigma=sigma,D21=d,wk4m=k).items()}
        graph=np.asarray(features(c,sigma,d,k)).copy()
        assert graph.shape==(n,6) and np.isfinite(graph).all()
        assert all(np.array_equal(np.asarray(v),before[key]) for key,v in dict(C_off=c,sigma=sigma,D21=d,wk4m=k).items())
        self.pending[layer]=graph
        self.audit.append(dict(layer=layer,prestate_arrays_unchanged=True,min_sigma=float(np.min(before['sigma'])),max_abs_rho=float(np.max(abs(before['C_off'].astype(float)/(before['sigma'].astype(float)[:,None]*before['sigma'].astype(float)[None,:])))),maxabs_per_feature=np.max(abs(graph),axis=0).tolist()))

    def canonicalize(self,layer,inverse):
        assert layer in self.pending and layer not in self.rows
        graph=self.pending.pop(layer);self.rows[layer]=graph[np.asarray(inverse,dtype=int)].copy()

    def array(self):
        assert not self.pending and set(self.rows)==set(range(self.depth-1))
        return np.stack([self.rows[layer] for layer in range(self.depth-1)])

    def join_local32(self,pre22,ownpost4,scales4):
        graph=self.array()
        assert pre22.shape==(self.depth-1,self.width,22)
        assert ownpost4.shape==scales4.shape==(self.depth-1,self.width,4)
        return np.concatenate([pre22,ownpost4/scales4,graph],axis=-1)

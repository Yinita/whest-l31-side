"""Target-free full14 observer for one fixed complete 16-layer parity check.

Separate from frozen R54 capture helper: no stop, no current/future fitting.
Diagnostic snapshots are NumPy copies; all injected arithmetic uses fnp.
"""

import flopscope.numpy as fnp
import numpy as np
from tenfold_local_features_r48 import ownpost_features, prefeatures


def updated_diagonals(k1, k2, k3, k4, sigma, normalized_delta, inherited_delta=None):
    s = sigma.astype(fnp.float64)
    change = normalized_delta * fnp.stack([s**p for p in range(1, 5)], axis=1)
    mean = k1.astype(fnp.float64)
    if inherited_delta is not None:
        mean = mean + inherited_delta.astype(fnp.float64)
    mean = fnp.maximum(mean + change[:, 0], 0.)
    if inherited_delta is not None:
        mean = mean - inherited_delta.astype(fnp.float64)
    return (mean.astype(fnp.float32),
            fnp.maximum(k2.astype(fnp.float64) + change[:, 1], 1e-10).astype(fnp.float32),
            (k3.astype(fnp.float64) + change[:, 2]).astype(fnp.float32),
            (k4.astype(fnp.float64) + change[:, 3]).astype(fnp.float32))


class WholeObserver:
    def __init__(self, heads):
        assert set(heads) == set(range(1, 15))
        self.heads = heads
        self.pre = {}
        self.pre_seen = []
        self.post_seen = []
        self.injected = []
        self.audit = []

    def record_pre(self, state):
        layer = state['li']
        assert layer not in self.pre_seen
        self.pre_seen.append(layer)
        if not 0 < layer < 15:
            self.audit.append(dict(layer=layer, pre_injection_skipped=True))
            return
        dnorm = fnp.sum(fnp.abs(state['D21'].astype(fnp.float64)), axis=1)
        self.pre[layer] = prefeatures(state['mu'], state['var'], state['sigma'],
            state['alpha'], state['phi'], state['Phi'], state['pk1v'], state['D3'],
            dnorm, state['g4row'], state['K4_sigma'], layer)

    def record_post(self, state):
        layer = state['li']
        assert layer not in self.post_seen
        self.post_seen.append(layer)
        if layer == 0:
            self.audit.append(dict(layer=layer, post_injection_skipped=True))
            return
        assert 0 < layer < 15 and layer in self.pre
        live = [state[k] for k in ('K1v', 'K2v', 'K3v', 'K4v')]
        mixed = {k: np.asarray(state[k]).copy() for k in ('pk1v', 'K21', 'K22')}
        feature = ownpost_features(self.pre.pop(layer), state['sigma'], *live, state['delta'])
        head = self.heads[layer]
        correction = ((feature - head['feature_mean']) / head['feature_scale']) @ head['theta']
        values = updated_diagonals(*live, state['sigma'], correction, state['delta'])
        for target, value in zip(live, values):
            fnp.copyto(target, value)
        assert all(np.array_equal(np.asarray(state[k]), v) for k, v in mixed.items())
        self.injected.append(layer)
        self.audit.append(dict(layer=layer, injected=True, original_mixed_pairs_unchanged=True))

    def validate(self):
        assert self.pre_seen == list(range(16))
        assert self.post_seen == list(range(15))
        assert self.injected == list(range(1, 15)) and not self.pre
        return dict(pre_seen=self.pre_seen, post_seen=self.post_seen,
                    injected=self.injected, audit=self.audit,
                    first0_final15_skipped=True, complete16layer_return=True)

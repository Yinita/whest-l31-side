"""Synthetic observer boundary/update contract; no model/wholeprediction claim."""

import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from onpolicy_whole_observer_r59 import WholeObserver


def main():
    out = Path("results/phase2/onpolicy_whole_observer_contract_r59_r01.json")
    assert not out.exists()
    generator = np.random.default_rng(591003)
    n = 7
    heads = {layer: dict(feature_mean=fnp.zeros(26, dtype=fnp.float64),
                        feature_scale=fnp.ones(26, dtype=fnp.float64),
                        theta=fnp.asarray(generator.normal(0, 1e-5, (26, 4))))
             for layer in range(1, 15)}
    observer = WholeObserver(heads)
    errors = []
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
        for layer in range(16):
            values = dict(mu=generator.normal(.2, .01, n), var=np.full(n, .3),
                          sigma=np.full(n, .5), alpha=np.full(n, .4), phi=np.full(n, .3),
                          Phi=np.full(n, .7), pk1v=np.full(n, .4), D3=np.full(n, .02),
                          D21=np.zeros((n, n)), g4row=np.full(n, .01),
                          K1v=np.full(n, .2, dtype=np.float32), K2v=np.full(n, .3, dtype=np.float32),
                          K3v=np.full(n, .01, dtype=np.float32), K4v=np.full(n, .005, dtype=np.float32),
                          K21=np.zeros((n, n)), K22=np.zeros((n, n)), delta=np.full(n, .01, dtype=np.float32))
            state = {k: fnp.asarray(v) for k, v in values.items()}
            state.update(li=layer, K4_sigma=.01)
            observer.record_pre(state)
            if layer == 15:
                continue
            if layer > 0:
                from tenfold_local_features_r48 import ownpost_features

                features = np.asarray(ownpost_features(observer.pre[layer], state['sigma'],
                    *[state[k] for k in ('K1v', 'K2v', 'K3v', 'K4v')], state['delta']))
                correction = features @ np.asarray(heads[layer]['theta'])
                expected = np.stack([values[k].astype(np.float64) for k in ('K1v', 'K2v', 'K3v', 'K4v')], axis=1)
                expected[:, 0] += values['delta'].astype(np.float64)
                expected += correction * np.stack([values['sigma']**p for p in range(1, 5)], axis=1)
                expected[:, 0] = np.maximum(expected[:, 0], 0.) - values['delta'].astype(np.float64)
                expected[:, 1] = np.maximum(expected[:, 1], 1e-10)
            observer.record_post(state)
            if layer > 0:
                actual = np.stack([np.asarray(state[k]) for k in ('K1v', 'K2v', 'K3v', 'K4v')], axis=1)
                error = float(np.max(abs(actual - expected.astype(np.float32))))
                assert error < 1e-7
                errors.append(dict(layer=layer, update_maxerror=error))
            else:
                assert all(np.array_equal(np.asarray(state[k]), values[k])
                           for k in ('K1v', 'K2v', 'K3v', 'K4v'))
    audit = observer.validate()
    rejected = []
    for layers in (range(1, 14), range(15), range(1, 16)):
        try:
            WholeObserver(dict.fromkeys(layers, {}))
        except AssertionError:
            rejected.append(list(layers))
    assert len(rejected) == 3
    report = dict(version="r59-syntheticobserver-boundary-component-contract", complete=True,
        observer_sha256=hashlib.sha256(Path("experiments/phase2/onpolicy_whole_observer_r59.py").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        synthetic_fixture_only=True, model_read=False, labels_reference_read=False,
        whole_predict_parity=False, strict_acceptance=False, audit=audit,
        errors=errors, invalid_layer_sets_rejected=rejected, cost=context.summary_dict())
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ('audit', 'errors', 'cost')}))


if __name__ == "__main__":
    main()

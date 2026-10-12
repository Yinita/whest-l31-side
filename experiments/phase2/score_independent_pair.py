"""Shard-wise paired loss differences; reference noise cancels in expectation."""
import json
from pathlib import Path

import numpy as np


def main():
    predictions = json.loads(Path('results/phase2/independent_v29_leaf8_r01_predictions.json').read_text())
    assert predictions['complete']
    reference = json.loads(Path('results/phase2/modal_independent_reference_r03.json').read_text())
    rows = []
    for net in range(8):
        pair = [r for r in predictions['rows'] if r['network_id'] == net]
        if any('error' in r for r in pair):
            rows.append(dict(network_id=net, failures=pair))
            continue
        p, q = [np.asarray(r['predictions'][-1], dtype=np.float64) for r in pair]
        shards = [r for r in reference['rows'] if r['network_id'] == net]
        assert all(r['network_sha256'] == pair[0]['network_sha256'] for r in shards)
        means = np.asarray([r['final_mean'] for r in shards])
        # Loss(q)-loss(p) cancels reference square term exactly.
        difference = np.mean(q*q-p*p-2*means*(q-p), axis=1)
        se = difference.std(ddof=1)/np.sqrt(len(shards))
        mean = means.mean(axis=0)
        flops = [r['budget_summary']['flops_used'] for r in pair]
        multipliers = [max(.1, f/2**41) for f in flops]
        adjusted = multipliers[1]*np.mean((q-means)**2, axis=1)-multipliers[0]*np.mean((p-means)**2, axis=1)
        rows.append(dict(network_id=net, baseline_noisy_mse=float(np.mean((p-mean)**2)),
                         candidate_noisy_mse=float(np.mean((q-mean)**2)),
                         raw_paired_delta=float(difference.mean()),
                         raw_delta_normal95=[float(difference.mean()-1.96*se),float(difference.mean()+1.96*se)],
                         flops_before=flops[0],flops_after=flops[1],
                         adjusted_shard_delta=float(adjusted.mean()),
                         adjusted_scope='per-shard noisy loss includes reference variance weighted differently; not unbiased adjusted gain',
                         reference_shards=len(shards)))
    path = Path('results/phase2/independent_v29_leaf8_r01_scored.json')
    assert not path.exists()
    path.write_text(json.dumps(dict(rows=rows, scope='Normal95 approximation across independent shards; no independent neuron assumption; absolute MSE remains reference-noisy; math-local not grade'),indent=2)+'\n')


if __name__ == '__main__':
    main()

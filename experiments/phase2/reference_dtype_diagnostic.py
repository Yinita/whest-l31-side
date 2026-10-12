"""Offline quantization scale diagnostic; never changes reference or scoring."""
import json
from pathlib import Path

import numpy as np
from datasets import Dataset


def main():
    dataset = Dataset.from_file('benchmark/data/mini_first8/prepared/mini/data.arrow')
    rows = []
    for row in dataset:
        target = np.asarray(row['final_means'], dtype=np.float32)
        ulp = np.abs(np.spacing(target).astype(np.float64))
        rows.append(dict(network_id=row['mlp_id'], name=row['mlp_name'], count=target.size,
                         mean_ulp_squared=float(np.mean(ulp**2)),
                         max_half_ulp_squared=float(np.max((ulp / 2)**2)),
                         uniform_rounding_variance_proxy=float(np.mean(ulp**2) / 12)))
    result = dict(scope='Original Mini8 final targets; quantization scale only, no rescore',
                  limitation='ULP/12 assumes uniform independent rounding; not reference sampling uncertainty or exact quantization error',
                  rows=rows,
                  mean_uniform_rounding_variance_proxy=float(np.mean([r['uniform_rounding_variance_proxy'] for r in rows])))
    output = Path('results/phase2/ablation_v30_127_reference_dtype.json')
    assert not output.exists(), 'Never overwrite evidence'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

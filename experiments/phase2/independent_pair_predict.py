"""Frozen charged math-local independent paired predictions; no official grade."""
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import resource
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import SetupContext
from whestbench.domain import MLP


def main():
    output = Path('results/phase2/independent_v29_leaf8_r01_predictions.json')
    assert not output.exists()
    lock = Path('benchmark/data/ablation_execution.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads(Path('results/phase2/ablation_v30_115_independent_protocol.json').read_text())
    rows = []
    sources = ['estimators/k3_v29/estimator.py', 'estimators/discussion_leaf8_r01/estimator.py']
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    for network in range(8):
        file = Path(f'benchmark/data/independent_v30_115/network_{network}.npz')
        for source in sources:
            spec = importlib.util.spec_from_file_location('paired_estimator', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            estimator = module.Estimator()
            with flops.BudgetContext(flop_budget=2**41):
                estimator.setup(SetupContext(width=1024, depth=16, flop_budget=2**41, api_version='0.1', seed=0))
                weights = fnp.load(str(file))['weights']
                mlp = MLP(width=1024, depth=16, weights=[weights[j] for j in range(16)],
                          seed=protocol['weight_rng_seeds'][network], name=f'independent_{network}')
            try:
                with flops.BudgetContext(flop_budget=2**41) as ctx:
                    predictions = estimator.predict(mlp, 2**41)
                summary = ctx.summary_dict()
                with flops.BudgetContext(flop_budget=2**41):
                    values = predictions.tolist()
                row = dict(network_id=network, source=source, source_sha256=hashes[source],
                           network_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                           predictions=values, budget_summary=summary,
                           peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            except Exception as error:
                row = dict(network_id=network, source=source, source_sha256=hashes[source],
                           error=repr(error))
            rows.append(row)
            output.write_text(json.dumps(dict(rows=rows, complete=False,
                                              scope='charged math-local, not official grade'), indent=2)+'\n')
            print(f'saved network={network} source={source} error={row.get("error")}', flush=True)
            del estimator, module, weights, mlp
            import gc
            gc.collect()
    output.write_text(json.dumps(dict(rows=rows, complete=True, sources=hashes,
                                      versions={p: importlib.metadata.version(p) for p in ['flopscope', 'whestbench']},
                                      scope='charged math-local; RPC/resource acceptance separate'), indent=2)+'\n')


if __name__ == '__main__':
    main()

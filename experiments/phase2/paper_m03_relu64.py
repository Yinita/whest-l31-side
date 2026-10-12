"""Frozen complete-method small matrix, paper FLOPs distinct from contest billing."""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/tmp/whest-official-cumulant-reference/src')
import torch
from mlp_kprop.flop_utils import NamedFlopCounter
from mlp_kprop.kprop_harmonic import AUGMENT, SIMPLE, mlp_kprop
from mlp_kprop.mlp import MLP


def main():
    path = Path('results/phase2/paper_m03_r06_relu64.json')
    assert not path.exists()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    rows, networks = [], []
    for width in (64,):
        for layers in (4,):
            for seed in (13026031, 13026032):
                torch.manual_seed(seed)
                mlp = MLP(input_dim=width, hidden_dim=width, output_dim=width, num_layers=layers+1)
                digest = hashlib.sha256(b''.join(w.weight.numpy().tobytes() for w in mlp.Ws)).hexdigest()
                rng = torch.Generator().manual_seed(seed+1000)
                samples = 131072
                sums = torch.zeros(width)
                squares = torch.zeros(width)
                for _ in range(samples//1024):
                    x = mlp(torch.randn(1024, width, generator=rng), up_to_layer=f"act{layers-1}").out
                    sums += x.sum(0)
                    squares += (x*x).sum(0)
                mean = sums/samples
                variance = (squares-sums*mean)/(samples-1)
                network = dict(width=width, layers=layers, seed=seed, network_sha256=digest,
                               samples=samples, reference_sampling_mse=float(variance.mean()/samples))
                networks.append(network)
                import numpy as np
                np.savez(f"benchmark/data/paper_m03_network64_{seed}.npz", weights=np.stack([w.weight.numpy().T for w in mlp.Ws[:layers]]), reference_mean=mean.numpy(), reference_variance=variance.numpy(), reference_samples=samples)
                for order in (2, 3, 4):
                    for kind in (SIMPLE, AUGMENT):
                        for factor in (True,):
                            start = time.monotonic()
                            row = dict(width=width, layers=layers, seed=seed, order=order,
                                       kind=kind.name, factor=factor, network_sha256=digest)
                            try:
                                with NamedFlopCounter() as counter:
                                    output = mlp_kprop(mlp, {1:torch.zeros(width),2:torch.eye(width)},
                                                       k_max=order,kind=kind,factor=factor,
                                                       use_avg_metric=True,output_all=True,
                                                       up_to_layer=f"act{layers-1}")
                                prediction = output[f"act{layers-1}"][1].to_tensor()
                                state_tensors = {}
                                def visit(value, seen):
                                    if id(value) in seen:
                                        return
                                    seen.add(id(value))
                                    if isinstance(value, torch.Tensor):
                                        state_tensors[id(value)] = value.numel()*value.element_size()
                                    elif isinstance(value, dict):
                                        for v in value.values():
                                            visit(v, seen)
                                    elif isinstance(value, (list, tuple)):
                                        for v in value:
                                            visit(v, seen)
                                    elif hasattr(value, "__dict__"):
                                        visit(vars(value), seen)
                                visit(output, set())
                                row["all_returned_state_tensor_logical_bytes"] = sum(state_tensors.values())
                                import resource
                                row["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                row.update(prediction=prediction.tolist(),
                                           noisy_mse=float(((prediction-mean)**2).mean()),
                                           paper_adjusted_flops=counter.total(),
                                           torch_raw_counted_flops=counter.mode.get_total_flops())
                            except Exception as error:
                                row['error'] = repr(error)
                            row['elapsed_s'] = time.monotonic()-start
                            rows.append(row)
                            path.write_text(json.dumps(dict(rows=rows,networks=networks,complete=False),indent=2)+'\n')
                            print(f'saved {len(rows)}/12 error={row.get("error")}',flush=True)
    path.write_text(json.dumps(dict(rows=rows,networks=networks,complete=True,
                                    official_commit='93d091a4c26c042bfffa28f2e76a81bc0aba94bb',
                                    scope='2networks12factorvariants,width64,fourReLU; allstate returned, logicalbytes may alias; cumulative processRSS; not competition billing'),indent=2)+'\n')


if __name__ == '__main__':
    main()

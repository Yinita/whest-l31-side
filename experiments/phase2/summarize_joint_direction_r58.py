"""Fixed two-stream true direction-vs-diagonal joint-gap diagnostics.

No fitting. Cross-noise removal is an uncertain estimator of signal energy,
never a measured MSE or competitor score. Positive residuals also retained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from joint_direction_stats_r58 import merge,joint_targets,readout_effect
from joint_direction_config_r58 import NETWORKS,LAYERS,COLUMNS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combine(rows):
    result=None
    for row in sorted(rows,key=lambda r:r['shard']):
        assert sha(row['path'])==row['payload_sha256']
        b=np.load(row['path'],allow_pickle=False);value={k:b[k] for k in ('count','mean','m2','m3','m4')}
        result=value if result is None else merge(result,value)
    assert result is not None
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    journal=a.root/'journal.jsonl';records=[json.loads(v) for v in journal.read_text().splitlines()]
    successful=[r for r in records if r.get('complete') and r.get('phase')=='MC']
    failures=[r for r in records if not r.get('complete')]
    assert len(successful)>=4
    result=[]
    for network,config in NETWORKS.items():
        weights=np.load(config['weights'],allow_pickle=False)['weights'];assert sha(config['weights'])==config['weight_file_sha256']
        directions=weights[list(LAYERS)][:,:,list(COLUMNS)].astype(float)
        base=np.load(a.root/f'baseline_network{network}.npz',allow_pickle=False)
        pre=base['pre5'];baseline_pre4=pre[...,:4];baseline_diagonal=base['diagonal4'];baseline_joint=baseline_pre4-baseline_diagonal
        streamrows=[[r for r in successful if r['network']==network and r['stream']==s] for s in (0,1)]
        streams=[joint_targets(combine(rows),directions,1024) for rows in streamrows]
        gaps=[s['joint_pre4']-baseline_joint for s in streams]
        effects=[readout_effect(pre,g) for g in gaps]
        chunks=[];k2chunks=[]
        if min(map(len,streamrows))>=8:
            for split0,split1 in zip(np.array_split(sorted(streamrows[0],key=lambda r:r['shard']),8),np.array_split(sorted(streamrows[1],key=lambda r:r['shard']),8)):
                targets=[joint_targets(combine(list(part)),directions,1024) for part in (split0,split1)]
                eg=[readout_effect(pre,t['joint_pre4']-baseline_joint) for t in targets]
                chunks.append(eg[0]*eg[1])
                k2gaps=[]
                for t in targets:
                    g=t['joint_pre4']-baseline_joint;g[...,2:]=0;k2gaps.append(readout_effect(pre,g))
                k2chunks.append(k2gaps[0]*k2gaps[1])
        for slot,layer in enumerate(LAYERS):
            e0,e1=effects[0][slot],effects[1][slot]
            cross=float(np.mean(e0*e1));noise=float(np.mean((e0-e1)**2)/4)
            k2effects=[]
            for gap in gaps:
                just=gap.copy();just[...,2:]=0;k2effects.append(readout_effect(pre,just)[slot])
            k2cross=float(np.mean(k2effects[0]*k2effects[1]));k2noise=float(np.mean((k2effects[0]-k2effects[1])**2)/4)
            interval=None;k2interval=None
            if chunks:
                energies=np.asarray([chunk[slot].mean() for chunk in chunks]);rng=np.random.default_rng(261058+network+layer)
                boot=energies[rng.integers(0,8,size=(20000,8))].mean(axis=1);interval=np.quantile(boot,[.025,.975]).tolist()
                k2energies=np.asarray([chunk[slot].mean() for chunk in k2chunks])
                k2boot=k2energies[rng.integers(0,8,size=(20000,8))].mean(axis=1);k2interval=np.quantile(k2boot,[.025,.975]).tolist()
            physical={key:[float(np.mean(s[key][slot]**2)) for s in streams] for key in ('joint_pre4','projection_float32_difference4')}
            totalerrors=[s['projected_pre4'][slot]-baseline_pre4[slot] for s in streams]
            diagerrors=[s['diagonal_pre4'][slot]-baseline_diagonal[slot] for s in streams]
            row=dict(network=network,layer=layer,samples_by_stream=[sum(r['samples'] for r in rows) for rows in streamrows],
                joint_gap_positive_mse_per_channel=[np.mean(g[slot]**2,axis=0).tolist() for g in gaps],
                total_pre_gap_positive_mse_per_channel=[np.mean(e**2,axis=0).tolist() for e in totalerrors],
                diagonal_gap_positive_mse_per_channel=[np.mean(e**2,axis=0).tolist() for e in diagerrors],
                joint_gap_cross_energy_per_channel=np.mean(gaps[0][slot]*gaps[1][slot],axis=0).tolist(),
                joint_effect_cross_energy=cross,joint_effect_average_noise_proxy=noise,joint_effect_noise_over_cross=noise/cross if cross>0 else None,
                jointK2_effect_cross_energy=k2cross,jointK2_effect_average_noise_proxy=k2noise,jointK2_effect_noise_over_cross=k2noise/k2cross if k2cross>0 else None,
                joint_effect_whole_direction_chunkbootstrap95=interval,physical_diagnostic=physical,
                jointK2_effect_whole_direction_chunkbootstrap95=k2interval,
                graph_available=bool(base['graph_available'][slot]),causal32_available_before_currentpost=True)
            result.append(row)
    gate_rows=[r for r in result if r['layer']==14]
    gate=not failures and len(successful)==256 and all(r['jointK2_effect_cross_energy']>0 and r['jointK2_effect_noise_over_cross']<.25 and r['jointK2_effect_whole_direction_chunkbootstrap95'] is not None and r['jointK2_effect_whole_direction_chunkbootstrap95'][0]>0 for r in gate_rows)
    report=dict(complete=len(successful)==256 and not failures,version='r58-fixedjoint-direction-noise-diagnostic',successful_jobs=len(successful),failed_records=len(failures),rows=result,fixed_pilot2_passed=gate,
        journal_sha256=sha(journal),source_sha256=sha(Path(__file__)),helper_sha256=sha('experiments/phase2/joint_direction_stats_r58.py'),
        caveats=['Two public already-used development networks; not independent validation.','Whole direction chunks preserve neuron/cross-moment correlations; eight MCchunks cannot certify population-network confidence.','Nonzero crossenergy is diagnostic signal estimate; negative values/intervals are uncertainty, not negative MSE.','Scalar jointdirection labels capture contractions of true jointcumulants, not fullcov/tensors.','No fittedalgorithm/rawMSE/adjusted/RPC or tenfold achievement.'])
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()

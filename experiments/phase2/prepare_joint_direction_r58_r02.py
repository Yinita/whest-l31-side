"""Freeze and review R58 exact inputs/source/cost contract without cloud."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from joint_direction_config_r58 import PREPARATION,SOURCE,SOURCE_SHA,DATA_REV,LAYERS,COLUMNS,SAMPLES,SHARDS,NETWORKS,job_seed


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output=PREPARATION.with_name('joint_direction_preparation_r58_r04.json')
    assert not output.exists()
    contract=Path('results/phase2/joint_direction_contract_r58_r02.json')
    c=json.loads(contract.read_text());assert c['complete'] and c['no_cloud']
    assert sha(SOURCE)==SOURCE_SHA
    files=[Path('experiments/phase2')/n for n in ('modal_joint_direction_mc_r58_r02.py','joint_direction_config_r58.py','joint_direction_stats_r58.py','joint_direction_contract_r58.py','prepare_joint_direction_r58_r02.py')]
    for p in files:ast.parse(p.read_text())
    network_rows=[]
    for network,r in NETWORKS.items():
        assert sha(r['weights'])==r['weight_file_sha256']
        inputs=json.loads(Path(r['inputs']).read_text());row=inputs[0]
        assert row['name']==r['name'] and row['seed']==r['setup_seed']
        assert Path(row['weights']).resolve()==Path(r['weights']).resolve()
        weights=np.load(r['weights'],allow_pickle=False)['weights'];assert weights.shape==(16,1024,1024) and weights.dtype==np.float32
        provenance=Path(r['dataset'])/'provenance.json';assert json.loads(provenance.read_text())['revision']==DATA_REV
        arrow=Path(r['dataset'])/'prepared/mini/data.arrow'
        assert arrow.is_file()
        directions=weights[list(LAYERS)][:,:,list(COLUMNS)]
        network_rows.append(dict(network=network,**r,weights_bytes_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
            inputs_sha256=sha(r['inputs']),dataset_provenance_sha256=sha(provenance),original_reference_arrow_sha256=sha(arrow),
            directions_float32_bytes_sha256=hashlib.sha256(directions.tobytes()).hexdigest(),
            used_public_development_network=True,fresh_holdout=False,corpus_idx_identity=False))
    prior=Path('results/phase2/modal_covariance_causal_r02/completion.json');runtime=json.loads(prior.read_text())
    t=np.asarray([r['elapsed_s'] for r in runtime['successes']]);cpu_seconds=256*t.mean()
    seed_values=[job_seed(n,s,k) for n in NETWORKS for s in (0,1) for k in range(SHARDS)];assert len(set(seed_values))==256
    helpers={n:sha(Path('experiments/phase2')/n) for n in ('tenfold_local_features_r48.py','tenfold_graph_features_r50.py')}
    r=dict(complete=True,version='r58-joint-direction-fixed-publictwo-preparation',no_cloud_launch=True,
        source_sha256={str(p):sha(p) for p in files},capture_helper_hashes=helpers,
        stats_helper_sha256=sha('experiments/phase2/joint_direction_stats_r58.py'),contract_path=str(contract),contract_sha256=sha(contract),
        h18_source_sha256=SOURCE_SHA,dataset_revision=DATA_REV,networks=network_rows,
        target_prelayers=list(LAYERS),canonical_output_columns=list(COLUMNS),previous_layers=[l-1 for l in LAYERS],
        samples_per_shard=SAMPLES,shards_per_network_stream=SHARDS,samples_per_stream=SAMPLES*SHARDS,streams=2,
        total_jobs=256,total_network_samples=2*2*SAMPLES*SHARDS,cpu_per_worker=1,max_workers=96,
        MC_memory_mib=2048,baseline_capture_memory_mib=16384,timeout_s=300,retries=0,threads=1,batch=256,
        forward_dtype='float32 canonical weights/input/activation',projection_dtype='float64 previousf32activation@nextWselectedf64',
        stats_dtype='float64 centered Pébay mean/M2/M3/M4; unbiased cumulants after merge',
        same_samples_preserved_for_diagonal_and_joint=True,no_full_covariance_tensor=True,no_full_k3_k4_tensor=True,
        no_labels_in_candidate=True,no_fitting=True,no_inference_truth=True,raw_archive_volume='whest-joint-direction-r58-raw',
        MC_compute_volume_commit=False,archive_mode='separate serial batches8 after local durable payloads',
        causal_feature_order='pre22 where pk1 slot replaced Gaussianmean + previouspost4 nextWpower contractions/sigma^p + pregraph6; no current ownpost; final15graphzero unavailable',
        pilot1='all4smoke hash/finite/firstbatchPébay splitrelative<1e-11/f32projectionMSE<1e-10; baseline traced-versus-untouched sameweights RMS/errorRMS<.01 and rawratio drift<.01; firstanalyticGaussian calibration nominal mean-noise ratio .2..4; zero runtime failures required for signal acceptance',
        pilot2='pre14 jointK2effect two-streammean noise MSE / crosssignal <.25 and crosssignal>0 with whole-neuron-chunk bootstrap95 lower>0; no K3K4-only pass; all64 directions/pre8/14/15 and positive physical residuals retained; no negativecross estimate interpreted as negativeMSE',
        runtime_evidence=dict(path=str(prior),sha256=sha(prior),successes=len(t),failures=len(runtime['failures']),
            samples_success=sum(x['samples'] for x in runtime['successes']),wall_s=runtime['elapsed_s'],success_cpu_h=t.sum()/3600,
            per65536mean_s=t.mean(),per65536median_s=float(np.median(t)),per65536p95_s=float(np.quantile(t,.95)),
            forecast_MC256_success_cpu_h=cpu_seconds/3600,ideal_capacity_lower_minutes=cpu_seconds/96/60,
            wave_estimate_minutes_p95=3*np.quantile(t,.95)/60,
            standard_CPU_plus2GiB_nominal_USD=cpu_seconds*(.0000131+2*.00000222),actual_billing_unknown=True,
            limitations='new compactstats fewerreductions plus64f64directionGEMM not measured; prior300stimeout9; startup/archive/preemption/region/taxes excluded'),
        sources=[dict(url='https://www.sandia.gov/research/publications/details/formulas-for-robust-one-pass-parallel-computation-of-covariances-and-arbitr-2008-09-01/',title='Pébay SAND2008-6212 centered parallel moments'),
                 dict(url='https://modal.com/pricing',title='Official standardFunctions CPU.0000131/core/s + memory.00000222/GiB/s; actual billunknown')],
        scope='mathematical/source/input/cost preparation only; two alreadyused public development diagnostic networks, no independent generalization/candidate/score/tenfold')
    output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'output':str(output),'jobs':256,'networks':[x['name'] for x in network_rows],'cpu_hours_estimate':cpu_seconds/3600,'no_cloud_launch':True}))


if __name__=='__main__':main()

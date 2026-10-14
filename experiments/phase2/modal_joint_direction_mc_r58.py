"""Offline fixed two-public-development-network joint-direction MC pilot.

No estimator training or candidate truth. Launch only after frozen preparation
review and R54 completion. MC produces compact stable sufficient statistics.
"""
import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal
from joint_direction_config_r58 import OUT,PREPARATION,SOURCE,SOURCE_SHA,DATA_REV,LAYERS,COLUMNS,SAMPLES,SHARDS,NETWORKS,job_seed

app=modal.App('whest-joint-direction-mc-r58')
volume=modal.Volume.from_name('whest-joint-direction-r58-raw',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.11')
 .pip_install('numpy==2.2.6','flopscope==0.12.1','whestbench==0.16.1')
 .env({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 .add_local_file(str(SOURCE),'/code/h18.py')
 .add_local_file('experiments/phase2/joint_direction_stats_r58.py','/code/joint_direction_stats_r58.py')
 .add_local_file('experiments/phase2/tenfold_local_features_r48.py','/code/tenfold_local_features_r48.py')
 .add_local_file('experiments/phase2/tenfold_graph_features_r50.py','/code/tenfold_graph_features_r50.py'))
for network,record in NETWORKS.items():
    image=image.add_local_file(record['weights'],f'/data/network{network}.npz')


@app.function(image=image,cpu=(1,1),memory=(2048,2048),timeout=300,retries=0,max_containers=96,scaledown_window=2,volumes={'/raw':volume})
def moments(job):
    import platform
    import resource
    import sys
    import numpy as np
    sys.path.insert(0,'/code')
    from joint_direction_stats_r58 import empty,block,merge
    assert set(job)=={'network','stream','shard','seed','helper_sha256'}
    network,stream,shard=[job[k] for k in ('network','stream','shard')]
    assert network in NETWORKS and stream in (0,1) and 0<=shard<SHARDS
    assert job['seed']==job_seed(network,stream,shard)
    assert hashlib.sha256(Path('/code/joint_direction_stats_r58.py').read_bytes()).hexdigest()==job['helper_sha256']
    started=time.monotonic()
    try:
        path=Path(f'/data/network{network}.npz')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==NETWORKS[network]['weight_file_sha256']
        weights=np.load(path,allow_pickle=False)['weights']
        assert weights.shape==(16,1024,1024) and weights.dtype==np.float32
        directions=weights[list(LAYERS)][:,:,list(COLUMNS)].astype(np.float64)
        rng=np.random.default_rng(job['seed'])
        stats=empty((3,1152));calibration=empty((64,));firstbatch_merge_error=None
        max_projection_difference=0.;difference_square=0.;difference_count=0
        for _ in range(SAMPLES//256):
            x=rng.standard_normal((256,1024),dtype=np.float32)
            packed=[]
            for layer,w in enumerate(weights):
                previous=x
                pre=previous@w
                if layer in LAYERS:
                    slot=LAYERS.index(layer)
                    projected=previous.astype(np.float64)@directions[slot]
                    actual=pre[:,list(COLUMNS)].astype(np.float64)
                    difference=actual-projected
                    max_projection_difference=max(max_projection_difference,float(abs(difference).max()))
                    difference_square+=float(np.sum(difference*difference));difference_count+=difference.size
                    packed.append(np.concatenate([previous.astype(np.float64),projected,actual],axis=1))
                x=np.maximum(pre,0)
                if layer==0:
                    calibration=merge(calibration,block(x[:,list(COLUMNS)]))
            packed_batch=np.stack(packed,axis=1)
            if firstbatch_merge_error is None:
                whole=block(packed_batch);halves=merge(block(packed_batch[:128]),block(packed_batch[128:]))
                firstbatch_merge_error=max(float(np.max(abs(whole[k]-halves[k])))/(1+float(np.max(abs(whole[k])))) for k in ('mean','m2','m3','m4'))
                assert firstbatch_merge_error<1e-11
            stats=merge(stats,block(packed_batch))
        assert stats['count']==calibration['count']==SAMPLES
        buffer=io.BytesIO()
        np.savez_compressed(buffer,**{k:np.asarray(v) for k,v in stats.items()},
            **{f'first_{k}':np.asarray(v) for k,v in calibration.items()},
            layers=np.asarray(LAYERS),columns=np.asarray(COLUMNS))
        payload=buffer.getvalue()
        return dict(complete=True,**job,samples=SAMPLES,elapsed_s=time.monotonic()-started,
            weights_file_sha256=NETWORKS[network]['weight_file_sha256'],weights_bytes_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
            columns=list(COLUMNS),layers=list(LAYERS),max_projection_f32_difference=max_projection_difference,
            projection_f32_difference_mse=difference_square/difference_count,
            firstbatch_pebay_split_relative_error=firstbatch_merge_error,
            payload_sha256=hashlib.sha256(payload).hexdigest(),statistics_bytes=len(payload),compute_only_no_volume_commit=True,
            numpy_version=np.__version__,blas_config=str(np.__config__.CONFIG),machine=platform.machine(),
            maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            truth_reference_absent=True,payload=payload)
    except Exception as error:
        return dict(complete=False,**job,error_type=type(error).__name__,error=str(error),elapsed_s=time.monotonic()-started)


@app.function(image=image,cpu=(1,1),memory=(16384,16384),timeout=600,retries=0,max_containers=2,scaledown_window=2,volumes={'/raw':volume})
def baseline_capture(job):
    import importlib.util
    import re
    import sys
    from types import SimpleNamespace
    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np
    from whestbench import SetupContext
    sys.path.insert(0,'/code')
    from tenfold_local_features_r48 import prefeatures
    from tenfold_graph_features_r50 import features as graphfeatures
    assert set(job)=={'network','helper_hashes'}
    network=job['network'];assert network in NETWORKS
    for name,expected in job['helper_hashes'].items():assert hashlib.sha256(Path('/code',name).read_bytes()).hexdigest()==expected
    source=Path('/code/h18.py');assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA
    for key in tuple(os.environ):
        if re.match(r'V\d+_',key):os.environ.pop(key)
    path=Path(f'/data/network{network}.npz');assert hashlib.sha256(path.read_bytes()).hexdigest()==NETWORKS[network]['weight_file_sha256']
    weights=np.load(path,allow_pickle=False)['weights'];directions=weights[list(LAYERS)][:,:,list(COLUMNS)].astype(np.float64)
    lines=source.read_text().splitlines()
    pretrigger=next(i for i,v in enumerate(lines,1) if 'if riders and not NO_CORR:' in v)
    posttrigger=next(i for i,v in enumerate(lines,1) if 'mu = K1v if delta is None else K1v + delta' in v)
    def run(capture):
        spec=importlib.util.spec_from_file_location('r58_h18_actual',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        estimator=module.Estimator();previous={};rows={};audit=[]
        with flops.BudgetContext(flop_budget=2**41,quiet=True):
            estimator.setup(SetupContext(width=1024,depth=16,flop_budget=2**41,api_version='0.1',seed=NETWORKS[network]['setup_seed']))
        def tracer(frame,event,arg):
            if event!='line' or frame.f_code.co_filename!=str(source) or frame.f_lineno not in (pretrigger,posttrigger):return tracer
            state=frame.f_locals;layer=state['li'];inv=np.arange(1024) if state['inv_perm'] is None else np.asarray(state['inv_perm'])
            if frame.f_lineno==posttrigger and layer in (7,13,14):
                delta=state['delta'];mean=np.asarray(state['K1v'],dtype=np.float64)
                if delta is not None:mean=mean+np.asarray(delta,dtype=np.float64)
                previous[layer]=np.stack([mean]+[np.asarray(state[k],dtype=np.float64) for k in ('K2v','K3v','K4v')],axis=1)[inv].copy()
            if frame.f_lineno==pretrigger and layer in LAYERS:
                originals={k:np.asarray(state[k]).copy() for k in ('mu','var','D3','g4row')}
                pre=np.stack([np.asarray(state[k],dtype=np.float64) for k in ('mu','var','D3','g4row','phi')],axis=1)[inv]
                slot=LAYERS.index(layer);prior=previous[layer-1]
                diagonal=np.stack([(directions[slot]**p).T@prior[:,p-1] for p in range(1,5)],axis=1)
                sigma=state['sigma'];gaussian=state['sigma']*state['phi']+state['mu']*state['Phi']
                dnorm=fnp.zeros(1024,dtype=fnp.float64) if state['D21'] is None else fnp.sum(fnp.abs(state['D21'].astype(fnp.float64)),axis=1)
                pre22=prefeatures(state['mu'],state['var'],sigma,state['alpha'],state['phi'],state['Phi'],gaussian,state['D3'],dnorm,state['g4row'],state['K4_sigma'],layer)
                # The original pk1 slot becomes Gaussian mean, available before
                # nonlinearity. No current own-post4 is a model input here.
                graph=fnp.zeros((1024,6),dtype=fnp.float64) if layer==15 else graphfeatures(state['C_off'],sigma,state['D21'],state['wk4m'])
                s=np.asarray(sigma,dtype=np.float64)[inv][list(COLUMNS)]
                causal32=np.concatenate([np.asarray(pre22)[inv][list(COLUMNS)],diagonal/(s[:,None]**np.arange(1,5)),np.asarray(graph)[inv][list(COLUMNS)]],axis=1)
                assert all(np.array_equal(np.asarray(state[k]),v) for k,v in originals.items())
                rows[layer]=dict(pre5=pre[list(COLUMNS)].copy(),diagonal4=diagonal,causal32=causal32,graph_available=layer!=15)
                audit.append(dict(layer=layer,canonical_projection=True,no_current_ownpost_feature=True,prestate_unmodified=True,graph_available=layer!=15,NO_CORR=module.NO_CORR))
            return tracer
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as context:
            mlp=SimpleNamespace(width=1024,weights=[fnp.asarray(w) for w in weights])
            if capture:sys.settrace(tracer)
            try:prediction=np.asarray(estimator.predict(mlp,2**41)).copy()
            finally:sys.settrace(None)
        return prediction,rows,audit,context.summary_dict()
    started=time.monotonic();prediction,rows,audit,cost=run(True);control,_,_,controlcost=run(False)
    assert set(rows)==set(LAYERS) and np.isfinite(prediction).all()
    buffer=io.BytesIO();np.savez_compressed(buffer,
        pre5=np.stack([rows[l]['pre5'] for l in LAYERS]),diagonal4=np.stack([rows[l]['diagonal4'] for l in LAYERS]),
        causal32=np.stack([rows[l]['causal32'] for l in LAYERS]),graph_available=np.asarray([rows[l]['graph_available'] for l in LAYERS]),
        prediction=prediction,baseline_control=control,layers=np.asarray(LAYERS),columns=np.asarray(COLUMNS))
    payload=buffer.getvalue();archive=Path('/raw/r58_r01')/f'baseline_network{network}.npz';archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists();archive.write_bytes(payload);volume.commit()
    return dict(complete=True,network=network,source_sha256=SOURCE_SHA,weights_file_sha256=NETWORKS[network]['weight_file_sha256'],
        elapsed_s=time.monotonic()-started,audit=audit,trace_removed=sys.gettrace() is None,
        baseline_control_maxabs=float(np.max(abs(prediction-control))),baseline_control_difference_mse=float(np.mean((prediction[-1].astype(float)-control[-1].astype(float))**2)),
        capture_cost=cost,control_cost=controlcost,raw_archive_path=str(archive),raw_archive_sha256=hashlib.sha256(payload).hexdigest(),
        source_weight_same_control=True,reference_truth_absent=True,payload=payload)


@app.function(image=image,cpu=(1,1),memory=(2048,2048),timeout=300,retries=0,max_containers=1,scaledown_window=2,volumes={'/raw':volume})
def archive_batch(items):
    # Archive separately and serially after compute. A storage timeout cannot
    # be mistaken for MC numeric failure or lose already-returned local data.
    started=time.monotonic();rows=[]
    for item in items:
        assert set(item)=={'network','stream','shard','payload','sha256'}
        assert hashlib.sha256(item['payload']).hexdigest()==item['sha256']
        path=Path('/raw/r58_r01')/f"network{item['network']}"/f"stream{item['stream']}_shard{item['shard']:03d}.npz"
        path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
        path.write_bytes(item['payload']);rows.append(dict(path=str(path),sha256=item['sha256']))
    write_seconds=time.monotonic()-started;volume.commit()
    return dict(complete=True,rows=rows,write_seconds=write_seconds,commit_seconds=time.monotonic()-started-write_seconds)


@app.local_entrypoint()
def main():
    import numpy as np
    from datasets import Dataset
    from joint_direction_stats_r58 import cumulants,merge
    assert PREPARATION.exists() and not OUT.exists()
    preparation=json.loads(PREPARATION.read_text())
    assert preparation['complete'] and preparation['no_cloud_launch']
    assert preparation['source_sha256']['experiments/phase2/modal_joint_direction_mc_r58.py']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert preparation['samples_per_stream']==SAMPLES*SHARDS
    # Avoid concurrency with the expensive sequential training run.
    r54root=Path('results/phase2/modal_onpolicy_local_joint_r54')
    r54manifest=json.loads((r54root/'manifest.json').read_text())
    assert not Path(f"/proc/{r54manifest['pid']}").exists(), 'No overlap with actual R54 orchestrator PID'
    assert (r54root/'completion.json').exists() or list(r54root.glob('failure_layer*.json')), 'R54 terminal evidence required'
    OUT.mkdir();manifest={**preparation,'pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'two already-used public development networks only; no fitting/freshheld/score/candidate'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (OUT/'journal.jsonl').open('x') as journal:
        baseline_rows=[]
        for network in NETWORKS:
            row=baseline_capture.remote(dict(network=network,helper_hashes=preparation['capture_helper_hashes']))
            payload=row.pop('payload');path=OUT/f'baseline_network{network}.npz';path.write_bytes(payload)
            assert hashlib.sha256(payload).hexdigest()==row['raw_archive_sha256']
            row.update(path=str(path),payload_sha256=hashlib.sha256(payload).hexdigest(),phase='baseline')
            baseline_rows.append(row)
            journal.write(json.dumps(row)+'\n');journal.flush()
            assert row['complete'] and row['trace_removed']
        numeric_rows=[]
        for row in baseline_rows:
            network=row['network'];config=NETWORKS[network]
            reference_path=Path(config['dataset'])/'prepared/mini/data.arrow'
            expected=next(r for r in preparation['networks'] if r['network']==network)
            assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==expected['original_reference_arrow_sha256']
            reference=np.asarray(Dataset.from_file(str(reference_path)).select_columns(['all_layer_means'])[config['local_row']]['all_layer_means'],dtype=np.float64)
            capture=np.load(row['path'],allow_pickle=False)
            before=float(np.mean((capture['prediction'][-1].astype(float)-reference[-1])**2))
            control=float(np.mean((capture['baseline_control'][-1].astype(float)-reference[-1])**2))
            drift=float(np.sqrt(row['baseline_control_difference_mse']/before))
            numeric_rows.append(dict(network=network,original_reference_sha256=expected['original_reference_arrow_sha256'],baseline_capture_raw=before,baseline_control_raw=control,control_over_capture=control/before,drift_RMS_over_error_RMS=drift,passed=drift<.01 and abs(control/before-1)<.01))
        numeric_gate=dict(complete=True,passed=all(r['passed'] for r in numeric_rows),rows=numeric_rows,scope='samebody/sameworker numeric observer control versus unchanged references; not candidate gain')
        (OUT/'baseline_numeric_gate.json').write_text(json.dumps(numeric_gate,indent=2)+'\n')
        assert numeric_gate['passed'],'No MC expansion after baseline numeric control failure'
        jobs=[dict(network=n,stream=s,shard=k,seed=job_seed(n,s,k),helper_sha256=preparation['stats_helper_sha256']) for k in range(SHARDS) for n in NETWORKS for s in (0,1)]
        rows=[];archive_pending=[]
        for phase,phase_jobs in [('smoke4',jobs[:4]),('remaining252',jobs[4:])]:
            smoke_rows=[]
            for row in moments.map(phase_jobs,order_outputs=False,return_exceptions=True):
                if isinstance(row,Exception):row=dict(complete=False,error_type=type(row).__name__,error=str(row),phase='MCunknownjob')
                elif row.get('complete'):
                    payload=row.pop('payload');path=OUT/f"network{row['network']}_stream{row['stream']}_shard{row['shard']:03d}.npz"
                    assert not path.exists();assert hashlib.sha256(payload).hexdigest()==row['payload_sha256'];path.write_bytes(payload);row.update(path=str(path),phase='MC')
                    captured=np.load(io.BytesIO(payload),allow_pickle=False)
                    assert int(captured['count'])==SAMPLES and captured['mean'].shape==(3,1152)
                    assert all(np.isfinite(captured[k]).all() for k in ('mean','m2','m3','m4'))
                    archive_pending.append(dict(network=row['network'],stream=row['stream'],shard=row['shard'],payload=payload,sha256=row['payload_sha256']))
                rows.append(row);smoke_rows.append(row);journal.write(json.dumps(row)+'\n');journal.flush()
                if len(rows)%8==0 or not row.get('complete'):print('MC',len(rows),sum(r.get('complete',False) for r in rows),flush=True)
                if len(archive_pending)==8:
                    archive=archive_batch.remote(archive_pending);archive_pending=[]
                    journal.write(json.dumps(dict(phase='archive',**archive))+'\n');journal.flush()
            if phase=='smoke4':
                calibrations=[]
                for network in NETWORKS:
                    parts=[]
                    for row in smoke_rows:
                        if row.get('complete') and row['network']==network:
                            payload=np.load(row['path'],allow_pickle=False)
                            parts.append({key:payload[f'first_{key}'] for key in ('count','mean','m2','m3','m4')})
                    assert len(parts)==2
                    joined=merge(parts[0],parts[1]);mc=cumulants(joined)
                    weights=np.load(NETWORKS[network]['weights'],allow_pickle=False)['weights']
                    variance=np.sum(weights[0].astype(float)**2,axis=0)[list(COLUMNS)]
                    true_mean=np.sqrt(variance)/np.sqrt(2*np.pi)
                    true_var=variance*(.5-1/(2*np.pi));ratio=float(np.mean((mc[:,0]-true_mean)**2)/(true_var.mean()/int(joined['count'])))
                    calibrations.append(dict(network=network,combined_samples=int(joined['count']),analytic_first_mean_noise_ratio=ratio,passed=.2<ratio<4))
                passed=len(smoke_rows)==4 and all(r.get('complete') and r['truth_reference_absent'] and r['compute_only_no_volume_commit'] and r['projection_f32_difference_mse']<1e-10 and r['firstbatch_pebay_split_relative_error']<1e-11 for r in smoke_rows) and all(r['passed'] for r in calibrations)
                gate=dict(complete=True,passed=passed,rows=smoke_rows,first_calibrations=calibrations,scope='hash/finite/stablemoment/analyticGaussian/numeric integrity only; no signal or score')
                (OUT/'smoke_gate.json').write_text(json.dumps(gate,indent=2)+'\n');assert passed,'Keep smoke evidence; no remaining252 launch'
        if archive_pending:
            archive=archive_batch.remote(archive_pending);journal.write(json.dumps(dict(phase='archive',**archive))+'\n');journal.flush()
    result=dict(complete=len(rows)==256 and all(r.get('complete') for r in rows),returned=len(rows),failures=sum(not r.get('complete') for r in rows),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_fitting=True,no_scoreclaim=True)
    (OUT/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

"""Recover local R58 artifacts without fabricating missing cloud receipts."""
import hashlib
import importlib.metadata
import json
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
from joint_direction_stats_r58 import cumulants, merge


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('results/phase2/joint_direction_mc_r58_r02')
    output = Path('results/phase2/joint_direction_recovery_r67_r01.json')
    assert not output.exists()
    assert not Path('/proc/2516325').exists()
    journal = root / 'journal.jsonl'
    raw = journal.read_bytes()
    valid, corrupt = [], []
    for index, line in enumerate(raw.splitlines()):
        try:
            valid.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            corrupt.append(dict(line=index + 1, bytes=len(line), all_NUL=not line.strip(b'\0')))
    assert len(valid) == 5 and len(corrupt) == 1 and corrupt[0]['all_NUL']
    prep_path = Path('results/phase2/joint_direction_preparation_r58_r04.json')
    prep = json.loads(prep_path.read_text())
    for path, digest in prep['source_sha256'].items():
        assert sha(path) == digest
    smoke = [r for r in valid if r['phase'] == 'packaging_smoke_retained']
    assert len(smoke) == 4
    rows, calibrations = [], []
    for config in prep['networks']:
        network = config['network']
        path = root / f'baseline_network{network}.npz'
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {k: bundle[k].copy() for k in bundle.files}
        assert arrays['pre5'].shape == (3, 64, 5)
        assert arrays['causal32'].shape == (3, 64, 32)
        assert arrays['prediction'].shape == arrays['baseline_control'].shape == (16, 1024)
        assert all(np.isfinite(a).all() for a in arrays.values())
        assert np.array_equal(arrays['layers'], [8, 14, 15])
        assert np.array_equal(arrays['columns'], prep['canonical_output_columns'])
        assert np.array_equal(arrays['graph_available'], [True, True, False])
        assert np.array_equal(arrays['prediction'], arrays['baseline_control'])
        assert np.array_equal(arrays['causal32'][2, :, 26:], np.zeros((64, 6)))
        reference_path = Path(config['dataset']) / 'prepared/mini/data.arrow'
        assert sha(reference_path) == config['original_reference_arrow_sha256']
        with pa.memory_map(str(reference_path), 'r') as source:
            table = pa.ipc.open_stream(source).read_all()
            truth = np.asarray(table.column('all_layer_means')[config['local_row']].as_py(), dtype=np.float64)
            final = np.asarray(table.column('final_means')[config['local_row']].as_py(), dtype=np.float64)
        assert truth.shape == (16, 1024) and np.array_equal(truth[-1], final)
        mse = float(np.mean((arrays['prediction'][-1].astype(float) - truth[-1])**2))
        receipt = next((r for r in valid if r.get('phase') == 'baseline' and r['network'] == network), None)
        if receipt:
            assert sha(path) == receipt['payload_sha256']
        rows.append(dict(network=network, path=str(path), payload_sha256=sha(path),
                         receipt_available=receipt is not None, recovered_from_local_payload=receipt is None,
                         original_cloud_receipt_reconstructed=False, original_reference_sha256=sha(reference_path),
                         before_raw=mse, control_raw=mse, control_ratio=1., prediction_control_maxabs=0.,
                         numeric_gate_passed=True, zip_crc_passed=True, arrays_finite=True,
                         missing_metadata=[] if receipt else ['elapsed_s', 'capture_cost', 'control_cost', 'audit', 'trace_removed', 'raw_archive_receipt']))
        parts = []
        for item in smoke:
            if item['network'] != network:
                continue
            assert sha(item['path']) == item['payload_sha256']
            assert item['complete'] and item['truth_reference_absent']
            assert item['projection_f32_difference_mse'] < 1e-10
            assert item['firstbatch_pebay_split_relative_error'] < 1e-11
            with np.load(item['path'], allow_pickle=False) as bundle:
                parts.append({k: bundle[f'first_{k}'].copy() for k in ['count', 'mean', 'm2', 'm3', 'm4']})
        assert len(parts) == 2
        joined = merge(*parts)
        with np.load(config['weights'], allow_pickle=False) as bundle:
            assert sha(config['weights']) == config['weight_file_sha256']
            var = np.sum(bundle['weights'][0].astype(float)**2, axis=0)[prep['canonical_output_columns']]
        values = cumulants(joined)
        ratio = float(np.mean((values[:, 0] - np.sqrt(var)/np.sqrt(2*np.pi))**2) /
                      ((var*(.5-1/(2*np.pi))).mean()/int(joined['count'])))
        calibrations.append(dict(network=network, samples=int(joined['count']), ratio=ratio, passed=.2 < ratio < 4))
    report = dict(version='r67-localpayload-arrowcompat-recovery', complete=True,
                  source_sha256=sha(Path(__file__)), original_journal_sha256=sha(journal),
                  original_log_sha256=sha('results/phase2/joint_direction_mc_r58_r02.launcher.log'),
                  preparation_sha256=sha(prep_path), original_failure='datasets List metadata incompatible in Modal host Python3.10',
                  versions={k: importlib.metadata.version(k) for k in ['datasets', 'pyarrow', 'numpy']},
                  corruption=corrupt, valid_receipts=5, smoke_unique=4, baseline_payloads=2, rows=rows,
                  calibrations=calibrations, numeric_data_gate_passed=all(r['numeric_gate_passed'] for r in rows) and all(r['passed'] for r in calibrations),
                  full_original_metadata_gate_passed=False, remaining252_started=False,
                  no_cloud_samples_repeated=True, no_original_evidence_modified=True,
                  continuation_requires='Explicit provenance exception for lost network16 receipt; array/numeric gates pass but original prestate/trace audit cannot be recovered. Do not silently claim original full integrity gate.',
                  score_claim=False, stage_accepted=False)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

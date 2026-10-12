"""Sequential immutable screens for explicitly implemented backlog candidates."""

import argparse
import fcntl
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / 'experiments/phase2/ablation_queue.json'
STATE = ROOT / 'experiments/phase2/research_state.json'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys', nargs='+', default=['C07', 'C15', 'C14', 'C08', 'C10'])
    parser.add_argument('--n-mlps', type=int, default=1)
    parser.add_argument('--revision', default='r01')
    args = parser.parse_args()
    keys = args.keys
    lock = (ROOT / 'benchmark/data/ablation_execution.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    baseline_path = 'results/phase2/v29_mini1_control_20261012.json' if args.n_mlps == 1 else 'results/phase2/v29_mini8_local.json'
    baseline = json.loads((ROOT / baseline_path).read_text())
    before = baseline['results']['per_mlp'][0] if args.n_mlps == 1 else baseline['results']
    for key in keys:
        queue = json.loads(QUEUE.read_text())
        task = next(t for t in queue['tasks'] if t['key'] == key)
        if task['status'].startswith('completed') and args.revision == 'r01':
            continue
        candidate = ROOT / task['candidate_path']
        if 'candidate_sha256' not in task:
            task['candidate_sha256'] = json.loads(candidate.with_name('provenance.json').read_text())['sha256']
        assert hashlib.sha256(candidate.read_bytes()).hexdigest() == task['candidate_sha256']
        output = ROOT / (task['output_prefix'] + f'_{args.revision}_mini{args.n_mlps}_local.json')
        assert not output.exists(), 'Never overwrite evidence'
        validation = output.with_suffix('.validate.log')
        with validation.open('w') as stream:
            result = subprocess.run(['uv', 'run', '--frozen', 'whest', 'validate', '--estimator', str(candidate)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            task.update(status='failed_validation', evidence=[str(validation.relative_to(ROOT))])
            save(QUEUE, queue)
            continue
        command = ['uv', 'run', '--frozen', 'python', 'benchmark/phase2.py', 'run', '--estimator', str(candidate), '--dataset', 'benchmark/data/mini_first8', '--n-mlps', str(args.n_mlps), '--runner', 'local', '--output', str(output)]
        state = json.loads(STATE.read_text())
        assert state['active_experiment'] is None
        launcher = output.with_suffix('.launcher.log')
        with launcher.open('w') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            record = dict(id=task['id'], version=task['version'], key=key, candidate=task['candidate_path'], sha256=task['candidate_sha256'], command=command, pid=process.pid, pid_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21], started_at_utc=datetime.now(timezone.utc).isoformat(), report=str(output.relative_to(ROOT)), single_change=task['after'], status='running', network_ids=[str(i) for i in range(args.n_mlps)], data_commit=queue['protocol']['data_commit'])
            state['active_experiment'] = record
            state['history'].append(record)
            save(STATE, state)
            task['status'] = 'running'
            save(QUEUE, queue)
            process.wait()
        evidence = [str(p.relative_to(ROOT)) for p in [output, output.with_suffix('.manifest.json'), output.with_suffix('.log'), launcher, validation] if p.exists()]
        try:
            report = json.loads(output.read_text())
            after = report['results']['per_mlp'][0] if args.n_mlps == 1 else report['results']
            if args.n_mlps != 1:
                before = dict(before, flops_used=sum(r['flops_used'] for r in baseline['results']['per_mlp']))
                after = dict(after, flops_used=sum(r['flops_used'] for r in report['results']['per_mlp']))
            ratios = {metric: after[metric] / before[metric] for metric in ['final_layer_mse', 'adjusted_final_layer_score', 'flops_used']}
            passed = report['results']['n_failed_mlps'] == 0
            status = 'completed_positive' if passed and ratios['adjusted_final_layer_score'] < 1 else 'completed_negative' if passed else 'completed_failed'
            result = dict(scope=f'Mini{args.n_mlps} development only; no independent generalization claim', before_report=baseline_path, after_report=record['report'], before=before, after=after, ratios=ratios, failures=report['results']['n_failed_mlps'])
        except (ValueError, KeyError):
            status = 'completed_failed'
            result = dict(exit_code=process.returncode, reason='Missing parseable benchmark evidence')
        if task['kind'] == 'diagnostic' and status != 'completed_failed':
            status = 'completed_diagnostic'
            result['interpretation'] = 'Control diagnostic; not an algorithm improvement'
        if task.get('result'):
            task.setdefault('previous_results', []).append(task['result'])
        if task.get('evidence'):
            task.setdefault('previous_evidence', []).append(task['evidence'])
        if task['key'] in ('C03', 'C05') and result.get('ratios'):
            result['cost_gate_passed'] = result['ratios']['flops_used'] <= 1
            if not result['cost_gate_passed']:
                result['acceptance'] = 'Rejected: item requires FLOPs not to increase; lower score alone is insufficient'
        task.update(status=status, result=result, evidence=evidence)
        save(QUEUE, queue)
        state = json.loads(STATE.read_text())
        record.update(status=status, result=result, evidence=evidence)
        state['history'][-1] = record
        state['previous_experiment'] = record
        state['active_experiment'] = None
        state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        save(STATE, state)
        with (ROOT / 'experiments/phase2/RESEARCH_LOG.md').open('a') as stream:
            stream.write(f'\n## {task["version"]} {key}\n\n{task["before"]} → {task["after"]}\n\n{json.dumps(result, ensure_ascii=False)}\n\n状态{status}；Mini{args.n_mlps}开发集，独立确认及RPC尚未完成。\n')
        rows = [t for t in queue['tasks'] if t.get('result')]
        lines = ['# 消融执行结果\n\n逐项保留before与after；单网结果不代表Mini8或独立确认。\n\n', '| 版本 | key | 状态 | before adjusted | after adjusted | adjusted变化 | raw变化 | FLOPs变化 |\n|---|---|---|---:|---:|---:|---:|---:|\n']
        for t in rows:
            r = t['result']
            if 'ratios' in r:
                lines.append(f'| {t["version"]} | {t["key"]} | {t["status"]} | {r["before"]["adjusted_final_layer_score"]:.12g} | {r["after"]["adjusted_final_layer_score"]:.12g} | {(r["ratios"]["adjusted_final_layer_score"]-1)*100:+.6f}% | {(r["ratios"]["final_layer_mse"]-1)*100:+.6f}% | {(r["ratios"]["flops_used"]-1)*100:+.6f}% |\n')
        (ROOT / 'experiments/phase2/ABLATION_RESULTS.md').write_text(''.join(lines))
        print(json.dumps(dict(key=key, status=status, ratios=result.get('ratios'), failures=result.get('failures')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Rank evidence-based research prospects without resetting results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    path = ROOT / 'experiments/phase2/ablation_queue.json'
    queue = json.loads(path.read_text())
    leading = ['M01', 'M03', 'O03', 'O05', 'O06', 'H15', 'H16', 'H21',
               'L19', 'O29', 'L20', 'M04', 'M05', 'M02', 'H28', 'H14',
               'H23', 'L25', 'L26', 'H25', 'H26', 'H29', 'H30', 'H32']
    for task in queue['tasks']:
        key = task['key']
        done = task['status'].startswith('completed')
        promising = key in leading and not done
        task['is_promising'] = promising
        task['execution_priority'] = ('P0' if key in leading[:11] else 'P1') if promising else 'P2' if not done and 'negative' not in task['status'] else 'P3'
        task['promising_reason'] = ('结构性机制或论文完整对照；尚未证明收益' if promising else
                                    '已有负结果或已完成；保留证据，需新机制才重开' if done or 'negative' in task['status'] else
                                    '推导储备；当前证据不足，排在结构性任务之后')
        task['priority_basis'] = 'Research judgment, not measured success probability; dependencies and cost gates remain mandatory'
    queue['tasks'].sort(key=lambda t: (not t['is_promising'], t['execution_priority'],
                                       leading.index(t['key']) if t['key'] in leading else 999, t['id']))
    path.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 按 is_promising 排序的执行优先级\n\n',
             '134项＝原129项＋5项论文方法。is_promising 是研究判断，不是已验证收益；false不表示数学否决。原status/result/evidence均保留。先满足依赖、公式和成本门槛。\n\n',
             '| 顺序 | is_promising | 执行优先级 | key | 版本 | 名称 | 状态 | 判断依据 |\n',
             '|---:|---|---|---|---|---|---|---|\n']
    for i, t in enumerate(queue['tasks'], 1):
        lines.append(f"| {i} | {str(t['is_promising']).lower()} | {t['execution_priority']} | {t['key']} | {t['version']} | {t['name']} | {t['status']} | {t['promising_reason']} |\n")
    (ROOT / 'experiments/phase2/PROMISING_PRIORITY.md').write_text(''.join(lines))


if __name__ == '__main__':
    main()

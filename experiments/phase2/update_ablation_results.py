"""Render actual execution evidence without rebuilding or resetting the queue."""

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    queue = json.loads((ROOT / 'experiments/phase2/ablation_queue.json').read_text())
    counts = collections.Counter(t['status'] for t in queue['tasks'])
    lines = ['# WhestBench 消融执行结果\n\n',
             f'总任务129。当前状态：{dict(counts)}。完成快筛不等于独立确认；阶段1尚未达标。\n\n',
             '| 版本 | key | 状态 | before raw | after raw | before adjusted | after adjusted | adjusted变化 | FLOPs变化 | 失败 |\n',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n']
    for task in queue['tasks']:
        result = task.get('result', {})
        if 'ratios' not in result:
            continue
        before, after, ratios = result['before'], result['after'], result['ratios']
        if task['kind'] == 'diagnostic':
            continue
        lines.append(f'| {task["version"]} | {task["key"]} | {task["status"]} | {before["final_layer_mse"]:.12g} | {after["final_layer_mse"]:.12g} | {before["adjusted_final_layer_score"]:.12g} | {after["adjusted_final_layer_score"]:.12g} | {(ratios["adjusted_final_layer_score"]-1)*100:+.6f}% | {(ratios["flops_used"]-1)*100:+.6f}% | {result["failures"]} |\n')
    lines.append('\n## 合法计费网络交叉验证\n\n每折模型仅评测训练时排除的网络；开发集交叉验证不替代独立确认。\n\n| 版本 | key | before raw | after raw | before adjusted | after adjusted | adjusted变化 | 失败 |\n|---|---|---:|---:|---:|---:|---:|---:|\n')
    for task in queue['tasks']:
        cv = task.get('charged_full_cv', {})
        if 'before_adjusted' in cv:
            lines.append(f'| {task["version"]} | {task["key"]} | {cv["before_raw"]:.12g} | {cv["after_raw"]:.12g} | {cv["before_adjusted"]:.12g} | {cv["after_adjusted"]:.12g} | {(cv["after_adjusted"]/cv["before_adjusted"]-1)*100:+.6f}% | {cv["failures"]} |\n')
    lines.append('\n## 离线学习快筛\n\n以下为离线开发网络LOO raw误差；已完成合法计费的模型见上表，其余没有adjusted分数。\n\n| 版本 | key | before raw | after raw | raw变化 | adjusted |\n|---|---|---:|---:|---:|---|\n')
    for task in queue['tasks']:
        cv = task.get('result', {}).get('offline_cv', {})
        if 'before_raw' in cv and 'after_raw' in cv:
            lines.append(f'| {task["version"]} | {task["key"]} | {cv["before_raw"]:.12g} | {cv["after_raw"]:.12g} | {(cv["after_raw"]/cv["before_raw"]-1)*100:+.6f}% | 未计费验证 |\n')
    lines.append('\n## 公共配对确认\n\n已查看公共状态，上游调参重叠未知；不替代独立验收。\n\n| key | before raw | after raw | before adjusted | after adjusted | 失败 |\n|---|---:|---:|---:|---:|---:|\n')
    for task in queue['tasks']:
        cv = task.get('public_confirmation', {})
        if 'before_adjusted' in cv:
            lines.append(f'| {task["key"]} | {cv["before_raw"]:.12g} | {cv["after_raw"]:.12g} | {cv["before_adjusted"]:.12g} | {cv["after_adjusted"]:.12g} | {cv["failures"]} |\n')
    lines.append('\n## 部分来源整网筛选\n\n这些候选仅覆盖固定来源子集，不能将负结果推广到完整轨道。\n\n| key | version | before raw | after raw | before adjusted | after adjusted | 失败 |\n|---|---|---:|---:|---:|---:|---:|\n')
    for task in queue['tasks']:
        result = task.get('target_subset_r10', {})
        if result:
            lines.append(f'| {task["key"]} | v30.059-subset-r10 | {result["before_raw"]:.12g} | {result["after_raw"]:.12g} | {result["before_adjusted"]:.12g} | {result["after_adjusted"]:.12g} | {result["failures"]} |\n')
    lines.append('\n## 归因与证据\n\nC07/C10单网raw逐值评分相同，收益来自收费减少。C08/C14/C15的微小raw差尚不能排除浮点重排漂移，completed_positive只表示初筛分数更低。诊断条目不产生算法after分数，记录为不适用。\n\n')
    for task in queue['tasks']:
        if not task.get('evidence'):
            continue
        lines.append(f'### {task["version"]} {task["key"]} {task["name"]}\n\n')
        lines.append(f'before：{task["before"]}\n\nafter：{task["after"]}\n\n状态：{task["status"]}\n\n')
        result = task.get('result', {})
        if result.get('acceptance'):
            lines.append(f'验收：{result["acceptance"]}。\n\n')
        if 'ratios' in result:
            after = result['after']
            manifest_path = ROOT / result['after_report']
            manifest_path = manifest_path.with_suffix('.manifest.json')
            manifest = json.loads(manifest_path.read_text())
            timing = f'predict={after["wall_time_s"]:.6f}s；residual={after["residual_wall_time_s"]:.6f}s' if 'wall_time_s' in after else '逐网络时间/residual见原始报告，不与单网比较aggregate'
            lines.append(f'{timing}；FLOPs={after["flops_used"]}；peak child RSS={manifest.get("peak_child_rss_kib")}KiB；失败={result["failures"]}。\n\n')
        else:
            lines.append(f'诊断结论：{result.get("conclusion", "见原始证据")}；算法分数前后不适用。\n\n')
        histories = task.get('previous_results', [])
        for previous in histories:
            if 'after' not in previous:
                continue
            lines.append(f'前次快筛：{previous.get("scope")}；raw={previous["after"]["final_layer_mse"]:.12g}；adjusted={previous["after"]["adjusted_final_layer_score"]:.12g}；报告 `{previous["after_report"]}`。\n\n')
        for evidence in task['evidence']:
            lines.append(f'- [{Path(evidence).name}](../../{evidence})\n')
        lines.append('\n')
    (ROOT / 'experiments/phase2/ABLATION_RESULTS.md').write_text(''.join(lines))
    print(dict(counts))


if __name__ == '__main__':
    main()

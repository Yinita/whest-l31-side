"""Build the reviewable, versioned backlog and execution queue; starts no jobs."""

from __future__ import annotations

import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "experiments/phase2"
GROUPS = {
    "compute": "计算重组与资源",
    "highorder": "高阶生产与来源传播",
    "official": "官方生产与图表核查",
    "learning": "残差修正与独立验证",
}
BASELINE = "estimators/k3_v29/estimator.py"
BASELINE_SHA = "86d9ca9b28e6fe2b6c74750a0b6ae4bba4c14f742ddc3f5f56bc7fb4ec9d8e27"
READY = {"code_ready": "已有代码待正式评测", "spec_ready": "规格待实现及验证", "needs_derivation": "先推导再实现"}
KIND = {"algorithm": "算法或计算消融", "diagnostic": "诊断", "contract": "合同或资源验证"}


def build():
    queue_path = DIR / "ablation_queue.json"
    if queue_path.exists():
        old_queue = json.loads(queue_path.read_text())
        if any(
            r.get("status") not in {"pending", "blocked_derivation"}
            or r.get("result") or r.get("evidence")
            for r in old_queue.get("tasks", [])
        ):
            raise RuntimeError("Execution evidence exists; back up and merge by key, never overwrite")
    assert hashlib.sha256((ROOT / BASELINE).read_bytes()).hexdigest() == BASELINE_SHA
    rows = []
    for group in GROUPS:
        for row in json.loads((DIR / f"backlog_{group}.json").read_text()):
            # The same numerical-drift control must not be scheduled twice.
            if row["key"] == "L31":
                continue
            row["group"] = group
            rows.append(row)
    keys = {r["key"] for r in rows}
    assert len(keys) == len(rows)
    for row in rows:
        assert set(row["dependencies"]) <= keys, row
        assert row["readiness"] in READY and row["kind"] in KIND
        for field in ("name", "detail", "before", "after", "cost", "acceptance", "source"):
            assert isinstance(row[field], str) and row[field].strip(), (row["key"], field)
    by_key = {r["key"]: r for r in rows}

    def visit(key, ancestors):
        assert key not in ancestors, f"Dependency cycle: {ancestors}, {key}"
        for dependency in by_key[key]["dependencies"]:
            visit(dependency, ancestors | {key})

    for key in keys:
        visit(key, set())
    # Avoid promoting learning model branches without a useful parent model.
    conditional_learning = {f"L{i:02d}" for i in range(4, 18)}
    for number, row in enumerate(rows, 1):
        row["id"] = f"WB-P2-A{number:03d}"
        row["version"] = f"v30.{number:03d}"
        row["version_state"] = "reserved; not a verified score or released estimator"
        row["parent"] = {
            "default": BASELINE,
            "sha256": BASELINE_SHA,
            "rule": "Default frozen V29; if before names a prerequisite candidate, freeze that candidate explicitly and compare paired, never silently combine mechanisms",
        }
        row["status"] = "pending" if row["readiness"] != "needs_derivation" else "blocked_derivation"
        row["allow_lower_reasoning_execution"] = row["readiness"] != "needs_derivation"
        row["lower_reasoning_boundary"] = (
            "May implement the exact specification and validate; any absent formula/coefficient/API/cost proof changes status to blocked_derivation, no guessing"
            if row["allow_lower_reasoning_execution"] else
            "May collect fixed-source evidence only; no production candidate until formula, repeated-index rules, and complexity are reviewed"
        )
        row["conditional_gate"] = (
            "L03 or named parent shows reliable network-level CV gain; confirmation labels remain sealed"
            if row["key"] in conditional_learning else "All dependencies satisfy the required evidence type and pass mathematical/contract acceptance; a completed_negative score alone cannot unlock a correctness dependency; obey item-specific cost gates"
        )
        row["dependency_evidence_rule"] = (
            "C01 source exists with matching SHA and passed synthetic algebra/FLOP evidence; no competition benchmark completion required"
            if row["key"] == "C29" else
            "Mathematical/contract prerequisites require accepted checks and evidence, not merely completed status; algorithm prerequisites named as parents require frozen source and their explicit result gate"
        )
        row["candidate_path"] = (
            "estimators/k3_v29_symrecursive/estimator.py" if row["key"] == "C01" else
            f"estimators/ablation_v30_{number:03d}/estimator.py" if row["kind"] == "algorithm" else None
        )
        row["output_prefix"] = f"results/phase2/ablation_v30_{number:03d}"
        row["dependency_ids"] = [by_key[k]["id"] for k in row["dependencies"] if "id" in by_key[k]]
    for row in rows:
        row["dependency_ids"] = [by_key[k]["id"] for k in row["dependencies"]]
    order = ["C24", "C25", "C07", "C15", "C14", "C08", "H03", "H01", "H02", "H18", "C10", "C11", "O01", "O02", "O11", "L01", "L02", "L18", "L03"]
    counts = {
        "total": len(rows),
        "by_kind": dict(collections.Counter(r["kind"] for r in rows)),
        "by_readiness": dict(collections.Counter(r["readiness"] for r in rows)),
        "by_group": dict(collections.Counter(r["group"] for r in rows)),
    }
    data = {
        "schema_version": 1,
        "backlog_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reviewable pending ablations for later lower-reasoning continuous execution; not automatically dispatched",
        "baseline": {"path": BASELINE, "sha256": BASELINE_SHA, "mini8_adjusted": 5.778979755766664e-9},
        "stage": 1,
        "current_target_score": 5.779e-10,
        "stage_progression_requires": "Mini8 below target plus unseen-network paired >=10x gain, zero failures, RPC/contracts/resources and trustworthy unchanged references; then target/10",
        "protocol": {"data_commit": "aa99830fdc09fad15407b10e8e3459d3e18bba0a", "flopscope": "0.12.1", "whestbench": "0.16.1", "budget": 2**41, "seed": 0, "threads": 1, "predict_s": 120, "residual_s": 0.4, "setup_s": 5, "lambda": 0},
        "counts": counts,
        "first_batch_keys": order,
        "excluded_duplicate": {"L31": "Merged into C25; same repeated numerical control"},
        "tasks": rows,
    }
    (DIR / "ablation_queue.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    sections = [
        "# WhestBench Phase 2 待验证消融清单\n",
        f"清单版本 1.0，2026-10-12。共 {len(rows)} 项，按独立机制和验证对象拆分；不是 {len(rows)} 个已实现算法，也不承诺十倍收益。所有 v30.xxx 为本项目预留消融版本号，和公开上游 V29 版本无关。before 与 after 描述实现变化，不填写预测分数。\n",
        f"算法与计算消融 {counts['by_kind'].get('algorithm', 0)} 项，诊断 {counts['by_kind'].get('diagnostic', 0)} 项，合同与资源验证 {counts['by_kind'].get('contract', 0)} 项。已有代码 {counts['by_readiness'].get('code_ready', 0)} 项，规格待实现 {counts['by_readiness'].get('spec_ready', 0)} 项，需先推导 {counts['by_readiness'].get('needs_derivation', 0)} 项。诊断和资源结果不得记成精度提升。\n",
        "## 冻结对照与已完成结果\n",
        "原 V29 SHA256 为 `" + BASELINE_SHA + "`。Mini8 adjusted 为 5.778979755766664e-9，raw 为 2.2702818514019896e-8，零失败。第一阶段目标仍为 adjusted < 5.779e-10。当前单网控制 network0 raw 为 2.290466483145792e-8，adjusted 为 6.116832334984139e-9，FLOPs 为 587262754287。单网分数与 Mini8 aggregate 不混用。\n",
        "已测负结果单独保留：pairq3、spilluse、k4square、tree4marg、tree4last、trianglelast、doubleedge-final 均未改善 adjusted；coreproxy、coreproxy-first、年龄及禁用通道的历史结果见 RESEARCH_LOG。V25 对照 raw 接近 V29、FLOPs 高约37.3%。这些不进入默认待跑队列；更广对象必须先通过条目的重新开放门槛。symrecursive 只有合成核验，尚未跑比赛网络，copy 调度可能超 residual 限制。\n",
        "## 执行边界\n",
        "一项只改变一个主要机制。父版本默认冻结 V29；依赖版必须另存代码哈希并说明成对基线。依赖只是实现与证据前置，不表示自动把多个机制合并。独立采样种子、原 reference、网络集合、工具版本、2**41 预算、单线程、120秒 predict、0.4秒 residual、5秒 setup 和 lambda=0 固定。每次使用新输出路径，记录失败和负结果。\n",
        "低推理执行器只能选择已有代码或规格明确、依赖满足的条目；缺少公式、系数、重复指标扣除、合法算子或成本证明时改为需推导，不能猜测后继续。长研究项保留为研究储备，不自动逐项运行。小规模代数通过是实现门槛，不是提分证据。\n",
        "stage 达标须冻结Mini8低于目标、真正未见网络相对上一冻结候选至少十倍收益且零失败，并通过RPC/合同/资源与参考可信度验收。达标才把目标除以10；运行次数不推动stage。仅首次AIcrowd提交已获授权，后续提交必须先展示证据再请求授权；无需新增定时任务、购买算力或向官方HF输入仓库写输出。\n",
        "## 首批执行顺序\n",
        " → ".join(f"{by_key[k]['version']} {k}" for k in order) + "。这是建议顺序，依赖和条件优先于排序；若控制/正确性不通过先定位。C01待copy风险评估，C29先准备调度推导，二者不设循环依赖。\n",
        "## 条目索引\n",
        "| 版本 | 编号 | 名称 | 类型 | 准备度 | 优先级 |\n|---|---|---|---|---|---|\n",
    ]
    for row in rows:
        sections.append(f"| {row['version']} | {row['key']} | {row['name']} | {KIND[row['kind']]} | {READY[row['readiness']]} | {row['priority']} |\n")
    for group, title in GROUPS.items():
        sections.append(f"\n## {title}\n\n")
        for row in rows:
            if row["group"] != group:
                continue
            sections.append(f"### {row['version']} {row['key']} {row['name']}\n\n")
            sections.append(f"编号：{row['id']}。类型：{KIND[row['kind']]}。准备度：{READY[row['readiness']]}。优先级：{row['priority']}。\n\n")
            sections.append(f"**before：** {row['before']}\n\n**after：** {row['after']}\n\n**详情：** {row['detail']}\n\n")
            dep = "、".join(f"{by_key[k]['version']} {k}" for k in row["dependencies"]) or "无条目依赖；仍需通用正确性、数值及资源门槛"
            sections.append(f"**依赖：** {dep}。\n\n**成本：** {row['cost']}\n\n**验收及停止条件：** {row['acceptance']}\n\n**依据：** {row['source']}\n\n")
    sections.append("## 可执行资料\n\n机器队列：`ablation_queue.json`。执行提示词：`LOW_REASONING_HANDOFF.md`。原始分组规格：`backlog_compute.json`、`backlog_highorder.json`、`backlog_official.json`、`backlog_learning.json`。重复的 L31 漂移控制已合并入 C25。\n")
    (DIR / "ABLATION_BACKLOG.md").write_text("".join(sections))
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    build()

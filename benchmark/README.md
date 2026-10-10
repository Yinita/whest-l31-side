# Phase 2 benchmark 与一致性流程

首个候选为 504aldo MIT 公开 V29；V25 保留为相同统计方法、无 Strassen 的诊断对照。代码原样导入，来源、上游 commit 与 SHA-256 见 `estimators/k3_v29/provenance.json`。这次提交建立复现与计量起点，不宣称独立算法贡献或全球最强。

## 本地 benchmark

固定 Phase 2 公共数据 `v2-phase2`、Mini、seed=0、单线程、预算 2**41、lambda=0、120 秒墙钟、0.4 秒 residual、5 秒 setup。工具版本由 `uv.lock` 固定。本地输出文件与 manifest 包含代码哈希、数据 commit、工具版本、运行参数、时间及日志。每次使用新文件名。

```bash
uv run python benchmark/phase2.py run --estimator estimators/k3_v29/estimator.py \
  --n-mlps 8 --runner subprocess --output results/phase2/v29_mini8_subprocess.json
uv run python benchmark/phase2.py run --estimator estimators/k3_v29/estimator.py \
  --n-mlps 8 --runner local --output results/phase2/v29_mini8_local.json
uv run python benchmark/phase2.py compare results/phase2/v29_mini8_local.json \
  results/phase2/v29_mini8_subprocess.json --output results/phase2/v29_runner_comparison.json
```

首次下载完整 Mini 约 6.7GB Arrow。网络受限时先运行 `uv run python benchmark/prepare_public.py`，校验官方首个 Arrow shard 的 SHA-256，保留其前 8 个网络及原始十亿样本 targets，生成可复用缓存。随后给 run 命令添加 `--dataset benchmark/data/mini_first8`。`--streaming` 可供诊断，但当前 whest 数据评分路径仍需随机访问，不作为可靠缓存策略。缓存数据或网络下载问题不能以自生成的低样本 MC 得分代替公共 reference。自生成 64 样本检查仅检查形状、有限值、资源上限与跨调用稳定性。

执行顺序：2 个真实比赛形状检查 → Mini-8 快筛与 runner 对照 → Mini-100 完整公共 benchmark → 冻结独立验证集。后续假设使用相同网络成对比较，原始 per-MLP JSON 为审计依据。

## 线上 benchmark

上传经过本地检查的同一包，保存 SHA-256、清单、描述、提交 ID、Phase 2 round ID 1429、工具版本、线上状态及评分。首次只提交一次。

```bash
uv run whest package --estimator estimators/k3_v29 --yes --output results/phase2/k3_v29.tar.gz
uv run whest validate-package results/phase2/k3_v29.tar.gz
uv run whest submit results/phase2/k3_v29.tar.gz --description 'Phase2 baseline: 504aldo MIT V29 unchanged; reproduction and metering check'
uv run python benchmark/phase2.py status SUBMISSION_ID --output results/phase2/online_SUBMISSION_ID.json
```

提交不是 benchmark 脚本的自动行为。上传后，重复检查现有 ID，不能因评分未完成而重复上传。提交包目录只含 estimator、MIT LICENSE、来源记录；研发依赖、凭据、参考答案不进入提交包。

## 一致性的含义

1. **实现一致**：本地估计器哈希与上传 tar 中的 estimator 哈希一致；参数保持默认，清除 V17–V29 的实验环境覆盖。
2. **规则一致**：数据 revision、形状、预算、评分公式、时间限制、工具版本一致。线程限制不能完全模拟 grader 的物理核限制。
3. **本地 runner 一致**：同网络逐项核对 raw MSE、FLOPs、adjusted score、失败数；浮点运算差异应很小，时间和 residual 允许机器差异。
4. **线上统计一致**：Mini 与线上未必是同一网络集合，不能要求 aggregate 相等。先比较误差量级、用量和失败；本地 Mini-100 对线上 aggregate 的相对差异作为诊断，不作为逐网络数值等价证明。

若线上提供逐网络细节，只有 ID/seed、reference 与规则对齐后才能成对比较。否则明确保留“线上 aggregate 对照”的状态。计量修复导致 FLOPs 变化时，应先对齐工具版本，不能把降费误当作算法精度进步。

## 推进计划

- 10 月 10 日：锁定 V29，完成真实形状检查、首次提交、记录线上结果；开始 Mini 与 runner 对照。
- 10 月 11 日：完成 Mini-100、V25/V29 精度与成本对照、冷启动和连续运行；选择一个来源压缩或计算重组假设。
- 10 月 12 日：48 小时评审。强基线可解释、零失败、成本对齐后才扩展；否则先修环境。
- 后续按项目报告 E03–E10：快筛、独立网络、消融与冻结。最终贡献必须超出原样复制。

官方 Phase 2 公告确认技术报告截止为 UTC 2026-10-24 23:59，即北京时间 2026-10-25 07:59。原项目报告对此列为未确认，此处依据 10 月 10 日再次读取的官方公告补充。

来源：[公开基线](https://github.com/504aldo/whest-p2-cumulant-k3)、[官方 Phase 2 公告](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197)。

## 本次执行记录

2026-10-10 首次提交 [#335146](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/335146) 已接受，平台返回 `challenge_round_id=1429`。上传包与本地源码 SHA-256 已逐字节核对。`submission_335146_manifest.json` 保留包哈希和版本；`online_335146.json` 保存实际评分状态，未 graded 前不记上榜成功。

公开 Mini 数据准备使用官方文件 commit `aa99830fdc09fad15407b10e8e3459d3e18bba0a`，以 LFS SHA-256 验证下载完整性。网络准备脚本的并发下载只用于本地数据缓存，绝不进入比赛提交包。

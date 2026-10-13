# Phase 2 benchmark 与一致性流程

2026-10-13官方Resources覆盖复核见[审计](../experiments/phase2/DYNAMIC_CONTENTS_COVERAGE_2026-10-13.md)。Mini100/Full1000尚未完成。原`run_rpc.py`没有setup硬超时；后续严格RPC工具已验证外部5秒setup与125秒响应截止、predict/residual后验检查和失败后冷恢复，详见下方严格验收记录。本地trusted-server参数尚未确认与官方部署一致。本页早期日程是计划，不是完成记录。

后续新增可信父进程监督工具`run_rpc_strict.py`、`rpc_strict_worker.py`、`rpc_strict_score.py`；原RPC结果与脚本保留。故障注入及候选真实local8失败见[严格时限验证](../experiments/phase2/RPC_STRICT_VALIDATION_2026-10-13.md)。新工具未完成全候选验收前，不将旧RPC报告追认为新约束通过。

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

Mini-8 是前 8 个公共网络，仅用于复现与工程快筛；V29 的部分系数已由作者在公共网络拟合，不把 Mini-8 当作独立泛化验证集。后续确认需使用完整 Mini 及单独冻结的未见网络。


## 已验证结果与后续门槛

首次提交 #335146 已 graded，线上 adjusted **5.4181681e-9**，平台 secondary **2.1326516e-8**。Mini-8 本地 adjusted **5.7789798e-9**，raw **2.2702819e-8**，util **25.4547%**，零失败。线上 adjusted 比 Mini-8 低约 6.24%，与作者公开的不同网络集合偏移同量级，不声称同网络数值等价。

普通 subprocess 在 8GB RLIMIT_AS 下 **4/8 失败**；报告保存具体 MemoryError 和 WORKER_EOF。限制作用于本地 NumPy backing buffers，与线上 RemoteArray client 分离计算的内存归属不同。这个失败报告保留为资源诊断，不能隐藏失败后只报成功样本。

官方 client/server 0.12.1 的本地 RPC 路径 **8/8 成功**，adjusted **5.7837855e-9**、raw **2.2721081e-8**，client 峰值 RSS **296896 KiB（约 290 MiB）**。RPC 与本地每网络 FLOPs 固定差 **16,777,216**，逐算子定位为权重 `astype` 的计量差，约占预算 0.00076294%。逐网络 raw MSE 最大相对差 **0.393%**，aggregate raw 差约 **0.0804%**；当前记录实际差异，不宣称位级相等。

RPC 使用单独 client 环境，无 NumPy；从已安装 whestbench 加载原始 sdk/domain 文件，避免根模块拉入 server-only 导入。trusted 数值 server 单线程；client 受 8GB RLIMIT_AS 约束。为容纳公开 V29 的大 buffer，本地 server 设置 `FLOPSCOPE_MAX_ARRAY_BYTES=1073741824`，默认 100MiB 单数组上限无法运行 V29。此为本地研发部署参数；官方 grader 的具体 trusted-server 上限和生命周期仍未获得独立确认。代码未修改，FLOP 计量未绕过，tokenless 开发 server 不等于完整官方沙箱。

可重复运行 RPC 对照并核对本地报告：

```bash
uv run python benchmark/run_rpc.py --n-mlps 8 --output results/phase2/v29_rpc_repeat_predictions.json
uv run python benchmark/rpc_score.py --predictions results/phase2/v29_rpc_repeat_predictions.json \
  --local results/phase2/v29_mini8_local.json --output results/phase2/v29_rpc_repeat.json
```

`run_rpc.py` 自动导出原始公共权重及 grader seeds、建立独立 client 环境、启动和关闭 server。输入缓存不上传 GitHub；具体版本见 client-requirements.txt。自动工作流的单网络检查已通过。

下一步先扩展 Mini-100 并测长期 client/server 内存及资源尾部；再完成 V25/V29 输出与成本消融。只有这些稳定后，才进入 E03 来源扰动及输出导向压缩。暂不以低样本 MC 或 Mini-8 开发集选择证明泛化。

榜单验证：公开页面的 TeamLeaderboardEntry 显示团队 **Niko**、成员 **nikoo_yu**、提交 **335146**、排名 **111**、Entries=1；动态快照见 `results/phase2/leaderboard_335146.json`。下一步不重复提交相同基线，先完成完整公共 benchmark 和自身改进消融。

## 扩展公共网络集合

`uv run python benchmark/prepare_public.py --shard 1` 缓存固定commit的第二个Mini分片至 `benchmark/data/mini_shard01`，校验官方LFS SHA并记录网络ID。其他分片可用0–13；0保持原冻结Mini8路径。保留原targets，不合并到调参集。本地未调参的公共网络仍不能证明上游未使用；阶段独立验证须额外核查。下载与评测结果分开记录。

2026-10-13新增分片2包含7网（ID16–22），分片行数不能默认都是8。只导出RPC权重与官方派生seed，可以使用：

```bash
uv run --frozen python benchmark/prepare_rpc_inputs.py \
  --dataset benchmark/data/mini_shard02 --n-mlps 7 \
  --output-dir benchmark/data/rpc_inputs/frozen_shard02_r01
```

导出不执行候选、不读取reference以选择参数；新目录保存CRC校验、逐文件SHA、数据commit、SDK版本及seed。后续两候选复用这份固定输入，使用`run_rpc_strict.py`分别运行并用`rpc_strict_score.py`计分。

两个严格RPC评分报告可以使用`compare_strict_cohort.py --before ... --after ... --output ...`生成逐网成对证据；工具核验原预测与候选SHA、输入/数据/seed/版本/监督器/预算的一致性，拒绝不同分片混比。分片2已完成V29与V37b+H18各7网零失败，adjusted5.485889e-9→4.303414e-9、raw下降1.74%；分片3（ID23–29）新增7网双方严格RPC零失败，adjusted5.565246e-9→4.337734e-9（−22.06%）、raw下降2.40%；当前最佳严格RPC总覆盖30网。它们仍是公共数据，Mini100未完成，不能据此宣称最终独立确认。


完整同候选公共覆盖可用 `aggregate_strict_public.py --reports SCORE1 SCORE2 ... --output NEW.json` 合并。脚本核验候选/原预测哈希、版本和协议，拒绝重复网络，按网络等权计分并保留失败。当前30网汇总为adjusted4.480111090883221e-9/raw2.204721927986005e-8，零失败；各分片权重按实际网数，不能简单平均分片均分。[证据](../results/phase2/sprint_public_v37b_h18_strict_public30_r01.json)。

# WhestBench Phase 2 消融执行提示词

本提示词用于用户下一步指定的低推理强度模型。当前只准备交接资料，不表示已经更换模型或启动新线程。清单包含研究储备，执行者按准备度、依赖、证据和成本选择任务，不逐行盲跑。

## 可直接复制的任务

在 `/home/sti/Desktop/projects/whest-l31-side` 接续 Phase 2 消融研究。先阅读 `DEVELOPMENT.md`、`benchmark/README.md`、`experiments/phase2/research_state.json`、`experiments/phase2/RESEARCH_LOG.md`、`experiments/phase2/ablation_queue.json`、`experiments/phase2/ABLATION_BACKLOG.md`。

先检查 state 的 active_experiment、PID、`/proc/PID/stat` 启动 ticks、进程启动时间、锁、原始日志和 manifest 的 finished_at_utc。存在未完成实验则延续和收集结果，不重复启动，不修改运行中的候选，不并行运行大网络。当前交接时 active_experiment 为 null；再次接续时重新核查真实进程，不能沿用本句当实时状态。

冻结原 V29：`estimators/k3_v29/estimator.py`，SHA256 `86d9ca9b28e6fe2b6c74750a0b6ae4bba4c14f742ddc3f5f56bc7fb4ec9d8e27`。Mini8 adjusted 为 `5.778979755766664e-9`，raw 为 `2.2702818514019896e-8`。stage=1，目标 `adjusted < 5.779e-10`；最终须未见网络相对冻结候选至少十倍收益、零失败、RPC/合同/资源和参考不确定性验收。单网结果不当作Mini8 aggregate，运行轮次不推动stage。

优先从队列 first_batch_keys 中选择依赖满足、准备度为 code_ready 或 spec_ready 的一个条目。首批已处理或全部阻塞时，从其余 tasks 按优先级、可辨别收益和成本选择满足条件的独立条目；不要反复重启已完成负结果。每次一个主要机制，版本号写入 provenance 和日志。needs_derivation 只允许整理原始来源、表达式、维度及成本证据；未审定公式不写正式算法。即使标 spec_ready，遇到缺失系数、重复指标扣除、父版本、允许的 API 或数学前提，立即将该项改为 blocked_derivation 并记录问题；选择其他独立任务继续推进，不能猜公式、换参数掩盖失败。

before 是实际父实现，after 是唯一主要变化。默认父版本为冻结 V29；若条目 before 指名其他候选，先冻结该父候选 SHA 并在日志说明。依赖是实现/证据前置，不表示把依赖机制自动叠加。新算法目录按 candidate_path 建立，保留 MIT LICENSE、上游署名、parent hash、修改机制和候选 SHA。代码 hash 一旦正式评测开始不得改变。任何新改动另开 revision 与输出名。

数学与合同依赖必须附验收通过的证据，completed_negative 分数状态不能替代正确性通过。算法父版依赖需要冻结代码及其明确收益门槛。C29 的 C01 前置只要求现成代码和已通过的合成核验，不要求 C01 比赛评测完成；可先做调度审计，再决定真实快筛。

1. 准备最小独立正确性核验；使用明确来源与不同求值路线。允许离线 NumPy、完整小张量或 oracle 作诊断，候选不得依赖 truth、网络 ID 查表或未计量计算。
2. 做必要合同检查。小 shape 因 riders gate 不覆盖 1024×16 新分支时明确记录，以真实 network0 快筛补覆盖。
3. 算成本；明显超过预算或收益不足的高阶对象先否决/留诊断，不能强行扩大实验。资源任务不宣称精度贡献。
4. 使用新结果名执行真实 network0。所有预算、时间、seed、线程、reference 和工具锁保持冻结。保存失败、逐层和逐网指标。
5. 与相同网络、原 reference、同版本规则的父版本成对比较。raw 小差异需用 C25 数值控制判别，不用旧 aggregate 强行比较。增加计算的候选必须降低 adjusted 才值得扩量。
6. 有可信收益才扩大 Mini8；拟合类先按网络级 CV 决策。完整控制通过后对唯一候选进行 public shard01 一次确认，最后用可靠独立 reference 和新种子网络确认。任何已查看标签的确认集进入选择账本，不再称未见。
7. 仅有实质改善的冻结候选进入相同网络 RPC 验收。必要时先更新 run_rpc 的 estimator 接口并验证选择了新候选，不能默认脚本已指向候选。保留普通 subprocess 8GB 失败，不能放宽资源后伪称通过。
8. 写入 state/history、RESEARCH_LOG、队列该项 status/result/evidence；记录代码哈希、数据 commit/网络 IDs、工具版本、命令、PID及启动身份、输出路径、raw/adjusted、FLOPs、时间、内存、失败、结论和下一任务。有完成结果才报新结果；仍运行则明确写仍运行。
9. 有价值检查点做必要验证并 Git commit/push 到已有项目，不 force，不打印或提交凭据。不要自动再次上传 AIcrowd，不购买算力，不重建已取消定时任务，不向官方 HF 输入仓库写输出。

工具参数固定：flopscope `0.12.1`、whestbench `0.16.1`、`uv --frozen`、FLOP预算 `2**41`、lambda `0`、单线程、seed `0`、predict `120s`、residual `0.4s`、setup `5s`。数据 commit `aa99830fdc09fad15407b10e8e3459d3e18bba0a`，保留原十亿样本 targets。换工具/数据版必须重跑冻结 V29，收费变化不记作自身精度改进。

运行模板，先把版本、候选路径和新输出名替换为所选条目实际值：

```bash
uv run --frozen whest validate --estimator estimators/ablation_v30_007/estimator.py
uv run --frozen python benchmark/phase2.py run \
  --estimator estimators/ablation_v30_007/estimator.py \
  --dataset benchmark/data/mini_first8 --n-mlps 1 --runner local \
  --output results/phase2/ablation_v30_007_r01_mini1_local.json
```

模板以 C07 为例，仅示意填参格式；该候选尚需实现，不可直接执行。执行前以队列实际路径和准备度为准。算法任务创建代码；diagnostic/contract 任务使用明确诊断脚本，不创建虚假 estimator。

## 首批任务的执行规格

| key | 版本 | 首要动作 | 门槛 |
|---|---|---|---|
| C24 | v30.024 | 获取合法计费归因或合成 shape 组件账本 | 不修改计量器；明确估计与实际全网差别 |
| C25 | v30.025 | 需要判别小收益时做同源码重复控制 | 不覆盖基线；相同FLOPs；记录浮点漂移 |
| C07 | v30.007 | trim 时删掉最终层未读取的 wk4m 构造 | debug 和非suite分支不破坏 |
| C15 | v30.015 | abs(D21)@ones 改为 row sum | 记录求和次序和 beta 输出微差 |
| C14 | v30.014 | 证明 pk11 逐值对称后删除 assembly 二次平均 | 含 finite/subnormal 数值合同 |
| C08 | v30.008 | 最终层只生产 PK1 mean | 保护所有 pk2/3/4 的后续读取 |
| H03 | v30.033 | 列出现有 p1..4 的截断产品集合 | 只审计，不重复添加已有项 |
| H01 | v30.031 | p2/3/4 补 D3² w(6,p)/72 | Wick 独立核验；p1既有项保持 |
| H02 | v30.032 | p2/3/4 补 D3 g4 w(7,p)/144 | 先补并核验新Wick，不同时加H01 |
| H18 | v30.048 | 只替换边际metric为 diag(WWᵀ) | proxy消融，不宣称完整K4精确运输 |
| C10 | v30.010 | Wick selector GEMM 改明确定义的 indexed pick | 每term核验，真实收费核对 |
| C11 | v30.011 | PK2按输出组独立收缩 | 先确认旧einsum是否已利用稀疏 |
| O01 | v30.063 | K31官方生产逐项独立枚举核验 | 相同截断集合与输入cumulants |
| O02 | v30.064 | capped211生产含排除约定核验 | 不用同一公式的两种实现冒充独立验证 |
| O11 | v30.073 | K111×edge显式公式核验 | 小张量、严格distinct mask |
| L01 | v30.096 | 建原轨迹特征诊断缓存 | 候选禁止truth；诊断预测/FLOPs对照 |
| L02 | v30.097 | 冻结网络级训练和选择账本 | 所有同网络层/神经元同fold |
| L18 | v30.113 | 测合法特征的CV残差可预测性 | 低R²停止大模型/多特征扩张 |
| L03 | v30.098 | 单个13特征final残差ridge模型 | 固定超参，网络级CV后再确认 |

## 队列和证据更新

`backlog_*.json` 是条目规格源；`build_ablation_backlog.py` 生成可读文档和初始机器队列。正式开始执行后，不直接重运行生成器覆盖已有队列中的 status/result。需要重新生成时先备份执行队列，并按 key 合并已有状态、revision、evidence；不丢弃负结果。spec字段修订须增加清单版本号并记录原因。

建议执行状态依次为 pending、implementing、running、completed_negative 或 completed_positive；公式缺口用 blocked_derivation，依赖未满足用 waiting_dependency，成本否决用 rejected_cost。状态不是实验收益结论，必须附实际证据路径。一个阳性只有经过同网络成对确认后才命名 validated；stage 达标另用研究状态的完整门槛判断。

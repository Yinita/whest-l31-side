# WhestBench 当前实验状态

截至 2026-10-12，公开 V29 已完成 Phase 2 基线复现与首次上榜；尚无通过独立网络验证的自身改进。当前研究方向、门槛与实验设计以 [项目报告](ARC_WhestBench_Project_Report_2026-10-10.md) 为准，执行任务见 [开发入口](DEVELOPMENT.md)。

Phase 1 的 `pure-0.2.1s` 成绩属于 256×32 网络，不可用于判断 1024×16 的 Phase 2 表现。完整历史总结已移至 [Phase 1 实验报告](archive/phase1/EXPERIMENTS_REPORT.md)。

已有 [矩链探针](experiments/moment_chain/README.md) 保留作为研究参考，尚需在 Phase 2 网络和当前工具版本下重测。新增结果必须来自日志，并同时报告网络级误差、计算量与失败数。

## 2026-10-10 基线启动

公开 504aldo MIT V29 原样复现已作为 Phase 2 提交 #335146 上传并接受。工具锁定 flopscope 0.12.1 / whestbench 0.16.1；两次 1024×16 子进程资源检查零失败，residual 约 0.25–0.27 秒。该检查的 64 样本 reference 精度不足，不能用于算法性能结论。公共 Mini benchmark 与线上评分以 `results/phase2/` 的实际结果为准。推进流程见 `benchmark/README.md`。

## 首次 Phase 2 结果

#335146 已 graded：线上 adjusted 5.418168137938331e-9，平台 secondary 2.1326515771136202e-8。官方 Mini-8 本地 adjusted 5.778979755766664e-9，raw 2.2702818514019896e-8，零失败；线上 aggregate 比 Mini-8 低约 6.24%，两者网络集合未对齐，不能当作逐网一致性误差。

RPC Mini-8 adjusted 5.783785541907926e-9，raw 2.272108123868577e-8，零失败。普通 subprocess 为 4/8 内存失败，不能用该汇总分数判断算法精度。完整诊断见 `results/phase2/consistency_335146.json` 及 benchmark 文档。

## 2026-10-12 快速候选筛查

在相同冻结Mini network0、原十亿样本reference和原资源规则下完成七项单机制筛查，全部零失败；仅是开发网快筛，没有独立泛化或阶段达标声明。

| 候选 | raw相对原V29 | adjusted相对原V29 | 结论 |
|---|---:|---:|---|
| Gaussian pair q3尾项 | +0.0965% | +0.1291% | 不扩量 |
| WK431 USE精确diagonal spill | −0.8848% | +4.1950% | 成本超过收益 |
| marginal峰度平方 | +1.0915% | +1.0915% | 不扩量 |
| 新生K4 tree无迹残差，一层记忆全链使用 | −3.4928% | +23.0863% | 全链成本太高 |
| 同残差仅用于最后层 | −0.00112% | +1.8349% | 微小raw差不能归因，未改公式早层亦有漂移 |
| covariance triangle rank16 final-only | −0.0404% | +0.2581% | 对当前控制adjusted亦更差；不证明full triangle无价值 |
| doubled covariance edge final-only | −0.0148% | +1.0869% | 对当前控制raw几乎无差，费用更高 |

原V29保持冻结；阶段1目标adjusted<5.779e-10尚未达到。原始报告、逐网络FLOPs/时间/内存/失败、候选SHA及来源见[研究日志](experiments/phase2/RESEARCH_LOG.md)和`results/phase2/*_mini1_local.*`。原V29当前环境控制已完成：raw为2.290466483145792e-8、adjusted为6.116832334984139e-9，零失败，FLOPs与冻结network0完全一致；raw浮点漂移−0.0157%。最后层残差候选相对当前控制反而略差。新控制另存，不覆盖冻结证据。汇总见[快速筛查结果](results/phase2/rapid_screen_20261012.json)。

纠正资料入口：论文实际公开仓库为[mlp_cumulant_propagation](https://github.com/alignment-research-center/mlp_cumulant_propagation)，冻结commit93d091a4c26c042bfffa28f2e76a81bc0aba94bb。官方生成器与V29既有pair项对照：22个新增项中19已存在且系数匹配，剩下3条正是已测q3。不能把其他slice可用的项重复叠加。下一方向是省略的混合K111/高阶对象表示，需先解决生产及合法成本。

公开V25单网控制raw=2.2909681263172388e-8、adjusted=8.400166403374769e-9，零失败；相对当前V29 FLOPs高37.2986%，不改变最强基线。C01全网单网快筛现已完成，adjusted降低0.489%，raw略升，独立/RPC待验证。用户已授权执行129项队列，真实进度见[消融结果](experiments/phase2/ABLATION_RESULTS.md)。

## 129项队列执行结果

H18 v30.048完整Mini-8与冻结V29成对核对通过：adjusted **5.778979755766664e-9 → 5.646816493872825e-9（−2.2870%）**，raw **2.2702818514019896e-8 → 2.218065286996307e-8（−2.3000%）**，总FLOPs增加0.002807%，零失败。8个开发网络全部adjusted下降（0.654%–3.599%）。此结果尚未通过独立网络/RPC确认，不满足阶段1十倍目标。逐项before/after、资源、版本、负结果及待验证项目由[结果文档](experiments/phase2/ABLATION_RESULTS.md)和live queue维护，不能把129项清单描述为全部完成。

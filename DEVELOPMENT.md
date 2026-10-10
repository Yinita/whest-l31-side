# WhestBench Phase 2 开发入口

更新日期：2026-10-10。当前决策依据为 [项目报告](ARC_WhestBench_Project_Report_2026-10-10.md)。先投入 48 小时完成强基线复现与诊断，再按独立网络上的误差和成本决定后续投入。

## 当前状态

- 根目录 `estimator.py` 保留 Phase 1 Hadamard 采样基线，仅作采样对照；尚未验证其 Phase 2 性能与成本，不作为最终候选。
- `examples/03_covariance_propagation.py` 是教学协方差实现，不等同于公开 K=3 强基线。
- `experiments/moment_chain/` 保留已有数学探针；其 2026-08 结果来自旧阶段网络，不是 Phase 2 复现。
- 已原样导入 504aldo MIT V25/V29，来源 commit 与哈希见 `estimators/`。V29 为首次 Phase 2 上榜基线；没有独立算法改进声明。
- V29 基础合同及两次 1024×16 子进程资源检查通过。低样本自生成 reference 仅作运行检查，不报告为精度。
- Phase 2 提交 #335146 已接受，round ID 1429，线上调整后分数 5.4181681e-9，最后层 MSE 2.1326516e-8，状态 graded。
- 官方 Mini-8：本地 runner adjusted 5.7789798e-9、raw 2.2702819e-8、util 25.4547%，零失败。
- 官方 client/server 本地 RPC Mini-8：adjusted 5.7837855e-9、raw 2.2721081e-8，零失败；client 峰值 RSS 约 290MiB。
- 普通子进程 Mini-8 在 8GB RLIMIT_AS 下 4/8 内存失败，保留原始报告。RPC与本地每网络固定 FLOP 差 16,777,216（astype 路径），最大逐网络 raw MSE 相对差约 0.393%；尚不能称所有环境完全等价。
- 工具已升级并锁定 flopscope 0.12.1 / whestbench 0.16.1；Phase 2 benchmark 规则明确固定。
- 可复跑命令与推进计划见 [benchmark/README.md](benchmark/README.md)。

## 当前任务

1. E00：核对报告中的规则及工具版本，运行官方 baseline，记录形状、评分、FLOPs、时间、内存和失败数。
2. E01/E02：按许可与署名要求导入公开 V25/V29，核对权重方向、层编号及最后 ReLU，完成逐层参考对齐。
3. E03/E04：测量来源扰动与组合扰动，构造仅用于诊断的 oracle 压缩上界。
4. E05–E07：选择一个高阶传播假设，做最小候选、合法代理及计算重组消融。
5. E09/E10：连续运行和冻结 holdout，按网络成对比较；满足门槛后再准备最终提交。

普通采样、低阶控制变量和边缘偏度修正保留为对照。原生内核、并发及计量旁路不进入提交路径。提交计算必须通过允许的 flopscope 算子。

## 记录要求

每次实验记录 experiment_id、commit、工具版本、网络集合标识、假设、唯一主要变化、raw MSE、逐网络 adjusted score、FLOPs、时间、内存、失败与结论。开发集与最终 holdout 按网络隔离。

报告建议的内部冻结时间为北京时间 2026-10-17 18:00，最终提交完成时间为当日 23:00 前。外部截止与最终选择流程执行前需再核查。

历史成绩和旧待办见 [Phase 1 归档](archive/phase1/README.md)。

首次上榜已核验：Phase 2 团队 Niko，2026-10-10 榜单快照排名 111，提交 #335146。

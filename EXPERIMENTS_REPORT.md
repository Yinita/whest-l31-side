# WhestBench 当前实验状态

截至 2026-10-10，仓库没有独立复现的 Phase 2 强基线或已验证改进。当前研究方向、门槛与实验设计以 [项目报告](ARC_WhestBench_Project_Report_2026-10-10.md) 为准，执行任务见 [开发入口](DEVELOPMENT.md)。

Phase 1 的 `pure-0.2.1s` 成绩属于 256×32 网络，不可用于判断 1024×16 的 Phase 2 表现。完整历史总结已移至 [Phase 1 实验报告](archive/phase1/EXPERIMENTS_REPORT.md)。

已有 [矩链探针](experiments/moment_chain/README.md) 保留作为研究参考，尚需在 Phase 2 网络和当前工具版本下重测。新增结果必须来自日志，并同时报告网络级误差、计算量与失败数。

## 2026-10-10 基线启动

公开 504aldo MIT V29 原样复现已作为 Phase 2 提交 #335146 上传并接受。工具锁定 flopscope 0.12.1 / whestbench 0.16.1；两次 1024×16 子进程资源检查零失败，residual 约 0.25–0.27 秒。该检查的 64 样本 reference 精度不足，不能用于算法性能结论。公共 Mini benchmark 与线上评分以 `results/phase2/` 的实际结果为准。推进流程见 `benchmark/README.md`。

# WhestBench Phase 2 研究项目

本仓库基于 AIcrowd whest-starterkit，当前目标是复现公开 K=3 高阶统计量传播方法，诊断误差与计算来源，再验证一项可解释的精度—成本改进。项目状态更新于 **2026-10-10**。

完整依据见 [项目调研与推进报告](ARC_WhestBench_Project_Report_2026-10-10.md)，当前任务见 [DEVELOPMENT.md](DEVELOPMENT.md)，实验状态见 [EXPERIMENTS_REPORT.md](EXPERIMENTS_REPORT.md)。报告中的公开结果与路线建议不等于本项目已完成的复现。

## 当前阶段

Phase 2 以宽 1024、深 16 的随机 ReLU MLP 为对象，返回 `(16, 1024)` post-ReLU 均值，按最后一层 MSE 与计算用量评分。报告记录预算为 `2**41` FLOPs，公开数据版本为 `v2-phase2`；执行正式实验前须核对 grader 与工具版本。

Phase 2 当前入口是 `estimators/k3_v29/estimator.py`，公开 V29 原样复现；首次提交 #335146 已接受，评分与记录见 `results/phase2/`。本地与线上 benchmark 流程见 [benchmark/README.md](benchmark/README.md)。

根目录 `estimator.py` 是保留的 Phase 1 Hadamard 采样对照，仍含旧阶段预算标定，且仅填充最后一层预测。它尚未完成 Phase 2 验收。教学例子也不等同于公开 V25/V29 实现。

## 安装与基础检查

```bash
uv sync --frozen --group dev
```

以下小网络检查用于确认已有本地工具可运行：

```bash-test
uv run python estimator.py --width 8 --depth 3
```

```bash
uv run pytest tests/ -q
uv run ruff check .
uv run whest validate --estimator examples/03_covariance_propagation.py
```

Phase 2 数据评测入口如下；应先完成版本核对和候选合同验收。这里使用教学协方差对照，不代表强基线复现：

```bash
uv run whest run --estimator examples/03_covariance_propagation.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 \
    --split mini --runner subprocess
```

提交计算需通过允许的 flopscope 算子。Torch/NumPy 参考程序仅用于研发。打包时显式指定经过验收的 estimator 文件或目录，并检查上传内容。

## 研究顺序

1. E00–E02：官方 baseline、V25/V29 复现、输出约定与成本对齐。
2. E03–E07：来源扰动、输出导向压缩、高阶核心近似与精确计算重组。
3. E09–E10：连续运行、独立网络验证、冻结 holdout 与消融。

先用 48 小时完成可解释复现与诊断。采样与低阶控制变量保留为对照；最终候选需有独立贡献、归因、成本记录与未见网络结果。

## 仓库导航

- `estimator.py`：历史采样对照。
- `local_engine.py`、`examples/`、`tests/`：本地教学、参考算法与检查。
- `experiments/moment_chain/`：保留的旧阶段矩链数学探针，尚需 Phase 2 重测。
- `docs/`：继承的 starter kit 教程及合同参考；涉及旧阶段的内容已标记。
- `archive/phase1/`：旧 L31 版本、日志、失败实验、提交草稿与报告，退出当前工作入口。
- `archive/starterkit/README.md`：上游历史首页。

报告建议北京时间 **2026-10-17 18:00** 内部冻结、当日 **23:00 前**完成提交；报告记录官方截止为 **2026-10-18 07:59**。最终选择界面与外部截止执行前需再次核查。

上游来源：[AIcrowd/whest-starterkit](https://github.com/AIcrowd/whest-starterkit)。许可证见 [LICENSE](LICENSE)。公开参赛代码的许可与署名须在导入时单独记录。

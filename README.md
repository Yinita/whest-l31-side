# WhestBench Phase 2 研究项目

更新于 2026-10-14 15:40 CST。最佳已验证候选是 **V37bH18**：线上 #335678 adjusted **4.293779716792071e-9**、raw **2.1220624581985702e-8**；公共65网严格RPC adjusted **4.459611165100404e-9**、raw **2.1926280100960693e-8**、零失败。用户要求复测的同包 **#335966** 仍 submitted / score null，预计约4.3e-9，实际以平台评分为准。

最新 R61 完成1560唯一有效capture并冻结完整14层模型；R62来源核验、R56实际组件及源代码检查通过。**R57首次新8网完整递推 raw 2.0529841238535947e-8 → 2.09938120285408e-8（+2.259982%，4/8改善、零运行失败）**，paired bootstrap95 ratio[0.9988951937,1.0516581763]，否决固定26特征线性on-policy边缘矩修正。R59整网observer/actualcandidate逐层最大差0，通过实现一致性。云诊断16GiB及residual超0.4秒，不是原资源验收；没有新adjusted成绩，不扩严格RPC/Mini8、不上传此候选。

阶段1仍要求冻结Mini8 adjusted **<5.779e-10**、独立未见网络至少十倍、零失败及严格资源/合同通过，**尚未达标**。原R54 300秒失败及全部负结果保留。下一研究方向需改变联合信息机制，不能继续扫同模型λ/scale/layers。

## 结果与执行入口

|用途|入口|
|---|---|
|最佳已验证提交代码|[V37bH18](estimators/sprint_public_v37b_h18_r01/estimator.py)|
|最新完成实验与before/after|[十倍研究进展](experiments/phase2/TENFOLD_PROGRESS_20261013.md)、[R57结果](results/phase2/onpolicy_fresh8_r57_summary_r01.json)|
|执行协议与下一步|[开发入口](DEVELOPMENT.md)、[状态](experiments/phase2/research_state.json)|
|benchmark复跑|[benchmark/README.md](benchmark/README.md)|
|134项队列（原129项＋论文方法）|[消融结果](experiments/phase2/ABLATION_RESULTS.md)、[机器队列](experiments/phase2/ablation_queue.json)|
|原始证据与实验来源|[研究日志](experiments/phase2/RESEARCH_LOG.md)、[结果索引](results/phase2/README.md)|
|官方资料与规则调研|[项目报告](ARC_WhestBench_Project_Report_2026-10-10.md)|
|仓库目录及发布边界|[仓库整理说明](docs/REPOSITORY_LAYOUT.md)|

## 复现

```bash
uv sync --frozen --group dev
uv run --frozen python benchmark/phase2.py run \
  --estimator estimators/sprint_public_v37b_h18_r01/estimator.py \
  --dataset benchmark/data/mini_first8 --n-mlps 8 --runner local \
  --output results/phase2/v37bh18_new_local.json
```

先按benchmark文档准备公开数据；使用新输出名，保留已有证据。固定flopscope0.12.1、whestbench0.16.1、NumPy2.2.6、seed0、单线程、2**41 FLOP预算、lambda0、predict120秒/residual0.4秒/setup5秒。Math/local用于数值研发，严格RPC用于提交路径和原资源检查。工具升级必须重跑冻结控制；公共开发集与独立确认、raw与adjusted分开报告。

## 历史与许可

根目录 `estimator.py` 是Phase1历史采样对照；当前Phase2提交入口见上表。`examples/`、`local_engine.py`、`tests/` 为教学与合同工具，`experiments/moment_chain/` 为旧阶段研究参考。Phase1报告在[archive/phase1/](archive/phase1/README.md)，starterkit原首页在[archive/starterkit/README.md](archive/starterkit/README.md)。

上游[AIcrowd/whest-starterkit](https://github.com/AIcrowd/whest-starterkit)，许可证[LICENSE](LICENSE)。公开方法源码保留各目录LICENSE/NOTICE。凭据、原始数组、MC labels及许可未明确的学习模型/系数仅本地保存；本仓库发布推理源码、复现工具和可公开的计量证据。

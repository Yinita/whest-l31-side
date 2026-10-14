# WhestBench Phase 2 开发入口

更新于 2026-10-14 15:40 CST。最佳已验证候选是 **V37bH18**：线上 #335678 adjusted **4.293779716792071e-9**、raw **2.1220624581985702e-8**；公共65网严格RPC adjusted **4.459611165100404e-9**、raw **2.1926280100960693e-8**、零失败。用户要求复测的同包 **#335966** 仍 submitted / score null，预计约4.3e-9，实际以平台评分为准。

最新 R61 完成1560唯一有效capture并冻结完整14层模型；R62来源核验、R56实际组件及源代码检查通过。**R57首次新8网完整递推 raw 2.0529841238535947e-8 → 2.09938120285408e-8（+2.259982%，4/8改善、零运行失败）**，paired bootstrap95 ratio[0.9988951937,1.0516581763]，否决固定26特征线性on-policy边缘矩修正。R59整网observer/actualcandidate逐层最大差0，通过实现一致性。云诊断16GiB及residual超0.4秒，不是原资源验收；没有新adjusted成绩，不扩严格RPC/Mini8、不上传此候选。

阶段1仍要求冻结Mini8 adjusted **<5.779e-10**、独立未见网络至少十倍、零失败及严格资源/合同通过，**尚未达标**。原R54 300秒失败及全部负结果保留。下一研究方向需改变联合信息机制，不能继续扫同模型λ/scale/layers。

## 当前执行顺序

1. 查询既有#335966，保存新状态文件；不重复提交同包。
2. R57/R59已完整收齐，固定26线性on-policy机制关闭。研究下一项真实联合方向/图context机制，先核查R58准备、原import失败及输入/代码哈希； prepared不等于执行完成。
3. 新候选先做单机制正确性与合法成本门槛，再完整新网络成对rollout；可靠raw收益后再同周期严格RPC和原冻结Mini8。
4. 满足阶段门槛才冻结候选并将目标除10。没有达标保持当前阈值。

## 核心约束

冻结Mini8是开发/复现集，不称独立泛化。使用原始reference、固定网络、seed0、线程1及锁定工具；预算2**41 FLOPs、lambda0、predict120秒、residual0.4秒、setup5秒。推理只用允许的flopscope算子；oracle/MC只离线诊断。16GiB云研究、局部矩CV、组件一致性或降费不能代替最终资源验收和精度改善。

实验开始先检查PID/startticks、锁、日志与完成标志；不重复启动、不修改运行源。每项记录唯一主要变化、版本和源码哈希、数据版本/网络ID、命令/PID、raw/adjusted、FLOPs、时间、内存、失败、结论。新输出路径保存原始证据和负结果。

## 文档与证据

- [逐版本before/after](experiments/phase2/TENFOLD_PROGRESS_20261013.md)
- [机器状态](experiments/phase2/research_state.json)与[研究日志](experiments/phase2/RESEARCH_LOG.md)
- [benchmark流程](benchmark/README.md)与[结果索引](results/phase2/README.md)
- [消融结果](experiments/phase2/ABLATION_RESULTS.md)与[原队列](experiments/phase2/ablation_queue.json)
- [官方调研报告](ARC_WhestBench_Project_Report_2026-10-10.md)
- [截至本次整理的旧开发日志](archive/research/DEVELOPMENT_CHECKPOINT_20261014.md)

历史快照保留当时状态；当前结论以上方及最新completion为准。学习系数/标签和凭据不进入Git；生成日志与快照留在本地，公开完成证据显式提交。下一轮接续读取state及完成报告，不以文档中的旧running行重新启动实验。

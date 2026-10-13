# 项目报告历史检查点（原文归档）

以下原报告顶部快照按当时记录保留，当前状态以开发入口与research_state为准；进行中条目不是现在仍在运行的证明。

# ARC White-Box Estimation Challenge 2026 项目调研与推进报告

2026-10-13 19:33：r47全部24TRAIN actualpartialdiag oracle raw2.212115e-8→4.454531e-9（约4.97倍），仅目标依赖诊断。r48 localjoint26训练网络CV改善K3/K4，模型已冻结；r49新8网640..752正在无真值完整递推，结果尚未收齐。此前totalstate模型均负，最佳仍V37bH18、十倍未达。

2026-10-13 19:28：实际partialdiag oracle两TRAIN网raw平均2.442906e-8→4.490815e-9（约5.44倍），仅目标依赖诊断，十倍未达/最佳不变。r47剩余22TRAIN网络正在采集localjointclosure输入，随后固定网络CV与无真值freshlearnedrollout；先前r38/r39/r41/r42完整学习候选均否决。参见[本轮真实结果](experiments/phase2/TENFOLD_PROGRESS_20261013.md)。

2026-10-13 19:20：r41二次交互及r42分层联合状态闭合在新8网完整完成，raw分别+8.44%/+10.61%，均零失败、均否决。train24静态CV改善没有迁移到闭环，不重复扩量/线上提交。r43正在仅24train上用固定Phi/W剥离继承均值误差，下一CV检验局部innovation预测；这是待验证假设，非已有提分。十倍目标未达、最佳仍V37bH18。

2026-10-13 19:16最新：独立corpus24train/8held的完整闭环已完成，线性mean-only raw+5.52%、joint4diag+19.43%，均0/8数值失败，否决；首轮4GiB32SIGKILL另保留。固定二次交互及perlayer机制通过train24网络CV，已冻结并启动新8网idx512..624完整闭环，尚无成绩。此为N1e8离线诊断，非Mini8/严格资源或十倍验收；最佳仍V37bH18。见[本轮before/after](results/phase2/corpus_joint_rollout_before_after_r38_r39.json)。

2026-10-13 19:11：独立corpus32完整baseline零失败、首层解析/MC噪声ratio0.99324通过；24train固定22feature ridge已冻结，静态held8误差比mean0.9934/var0.9858/K3 1.1707/K4 1.2568，无强信号。完整closed8的mean+variancefloor、formal4diag及真正K1only r39分别运行，未提前报提分。首版4GiB 32SIGKILL失败全部保留，r02是16GiB离线诊断，不比赛资源验收。最佳及十倍目标不变。

2026-10-13 19:00最新：r25/r34/r37真实四阶残差历史及r29完整pairgrade4严格单网均零失败，全部adjusted未改善，否决。r37纠正总迹后消除灾难退化，但raw仍+2.05%；停止该分支。下一P0为独立corpus网络的联合状态闭环训练pilot。十倍目标5.779e-10未达，线上最佳V37bH18未变。见[实际before/after](experiments/phase2/TENFOLD_PROGRESS_20261013.md)。


2026-10-13 18:44最新：r25真实D4-only全历史严格单网零失败但raw恶化19.415倍；r29完整pair grade4 raw−0.4766%但adjusted+3.1703%，均否决。下一r34保留harmoniccross并补signedD4残差历史；最佳V37bH18与阶段1目标5.779e-10不变，十倍未达。详见[实际before/after](experiments/phase2/TENFOLD_PROGRESS_20261013.md)。


2026-10-13最新：r20严格单网raw−3.7406%但adjusted+114.03%，未晋升；双流离线诊断1015成功/9timeout收齐，真实状态联合闭合仍是重点。最佳V37bH18与阶段1目标5.779e-10不变，十倍未达。最新候选、失败、before/after与局限见[专项进展](experiments/phase2/TENFOLD_PROGRESS_20261013.md)。

2026-10-13严格新结果：r14完整heavyhypertree全历史raw+0.029%/adjusted+239.7%；r16tail2真metric完整USE/feed raw+0.167%/adjusted+5.99%，均零失败、均否决。转向官方舍弃nonhypertree与真正四阶历史联合闭合，十倍尚未达标。
2026-10-13当前：r15固定二阶边际闭合严格单网零失败，但raw/adjusted均+1.08%，否决；完整三阶反馈新表示正在实现。双流离线oracle残余已接近原参考噪声，只作机制定位，不能称候选十倍收益。最佳与阶段1阈值不变。
2026-10-13后续：r12解析球面径向版本严格RPC单网raw仅−0.145%、adjusted+13.30%，零失败，未晋升。云独立新seed两流初步复现边际K2/K3/K4缺口与late局部mean误差的关联；后续出现基础设施失败并完整保留；oracle归因不可当作候选分数，正在完整采集与设计新的高阶表示。

2026-10-13最新执行进展：最佳仍为V37bH18（线上#335678 adjusted4.2937797168e-9；公共65网严格RPC零失败adjusted4.4596111651e-9）。阶段1目标5.779e-10未达。完整当前represented source/K4 proxy的retained非对角K31/K211生产已做整网负结果，不再写为尚未集成；它不等于完整真实K4。新rank1 GH2条件分布在两公共网络raw分别改善11.91%/7.37%，费用近翻倍导致adjusted恶化，未晋升。cheapdiag响应r10 raw恶化33.52%；fullcov响应r11 raw改善7.00%但adjusted恶化105.61%，均严格RPC零失败。当前转向解析径向剥离和离线逐层矩诊断，详见[十倍记录](experiments/phase2/TENFOLD_PROGRESS_20261013.md)。

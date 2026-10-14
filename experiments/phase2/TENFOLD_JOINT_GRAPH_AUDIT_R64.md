# R64/R65 联合方向与graph-context证据审计

2026-10-14。R57新8网raw恶化2.259982%，固定26线性on-policy边缘矩纠正已否决，最佳仍V37bH18。以下审计重新利用已有数组，不拟合、不重复R57、无新adjusted或十倍结论。

## 已有数据能回答什么

R51 TRAIN24的完整传输曾含同轨迹local32，但落盘仅保留`idx,graph_features(15,1024,6)`。旧R46/R47的26features/states/scales来自另一次浮点轨迹；固定拼接门槛复验仍14/24通过。不能通过放宽门槛、筛掉10网、或用传输hash假装恢复丢弃数组来拟合R53。

旧covariance_causal_r02有1015成功/9失败，只存每个神经元pre/post的count/mean/M2/M3/M4，没有样本或联合张量。它仍能支撑既有边缘矩误差和噪声归因，却不能恢复真实联合方向。R64明确反例：两种二维分布逐维前四阶矩完全相同，但(1,1)方向方差分别2和4。完整网络样本的来源不等于联合信息已被归档。

## 数学机制

R58目标为`Jp = κp(X@W[:,j]) − Σi W[i,j]^p κp(Xi)`，p2/3/4。p2含全部cross-covariance方向收缩；p3/p4含全部mixed cumulants。以前只改善四个边缘post矩，不能保证这些cross对象相符。R57的K3/K4局部CV改善而整网恶化，与此解释一致，但尚不能证明这是唯一根因。

R50六个graph特征来自当前prestate的C_off、D21、wk4m，均不读取真值，包括rho二/三阶行和、triangle及rho×mixed slice。它们是可计费的有限context，不能自动视为完整联合闭合。

## R65实际1024宽组件复用

使用已冻结同次H18 layer14的pre_state.npz，禁止再次predict或读取reference。独立dense公式最大差1.301e-18，原始到内部置换最大差1.388e-17，输入数组不变。合法fnp单层成本4,343,347,200 FLOPs；14层外推60,806,860,800，约H18的12.65%额外成本，外推不是整网计量。候选需要实际raw收益足以支付费用，并严格资源通过。

## R58 r02固定最小诊断

来源、weights、原reference箭头和64个方向hash全部复验。保留原ModuleNotFoundError，原运行返回0个MC样本。新身份r02在远端模块内冻结configuration，先四个唯一smoke，然后同worker两网络capture/control。smoke复用，不重新采样；只有预设数值/数据门槛通过才接续固定剩余252项。

网络是已经使用的public global0/16，不是corpus idx0/16或新holdout。layers8/14/15，canonical columns0,16,…1008，two streams×64shards×65536，批256，1CPU/2GiB/300s，最大96worker。当前MC与baseline纯离线，最终候选只能使用允许的fnp。两个开发网络不做学习泛化结论。

信号门槛在pre14固定jointK2：每网two-stream crossenergy>0、平均噪声proxy/crossenergy<.25、八whole-direction MCchunk bootstrap下界>0；无失败/不缺分片。K3/K4信号不能代替K2门槛，负cross估计不是负MSE。需要保留正物理残差、原失败、float32/64误差及所有层/方向。

后续最小算法实验必须按网络冻结训练/确认分割，并对joint target而不是另一次post marginal重拟合。先看R58固定信号能否过门槛，再预注册唯一新机制、数据协议与合法成本；不扫λ/scale/layer，不拿二网拟合结果当十倍收益。

## 一手资料

- [Pébay SAND2008-6212](https://www.sandia.gov/research/publications/details/formulas-for-robust-one-pass-parallel-computation-of-covariances-and-arbitr-2008-09-01/)：本轮重新浏览，讨论centered arbitrary-order moment pairwise/incremental merge与raw-power数值不稳。R58的merge独立离散分布合同复验通过。
- [官方factor_k3源码，commit93d091a](https://raw.githubusercontent.com/alignment-research-center/mlp_cumulant_propagation/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py)：lines271–278明确factorized augment舍弃covariance triangle及κ3×edge，影响相同领先MSE阶。该声明是上游算法限制；不能据此断言我们新特征已复现完整方法或保证提分。

公开证据：[R64](../../results/phase2/joint_graph_reuse_preflight_r64_r01.json)、[旧graph重验](../../results/phase2/graph_context_reuse_audit_r64_r01.json)、[merge合同重验](../../results/phase2/joint_direction_contract_reaudit_r64_r01.json)、[R65](../../results/phase2/graph_actual_state_component_r65_r01.json)。private数组保留本地。

## R67真实中断与兼容修复

R58r02PID2516325退出，host datasets不能解码Arrow metadata的List类型；journal第6行572NUL，原5条有效（4smoke+network0baseline）。network16NPZ存在且ZIP/shape/finite/输出控制核验通过，但原云metadata未持久化，不补造elapsed/cost/audit/trace/archive。R67直接pyarrow.ipc.open_stream读同SHA reference、最后层与final_means核对一致；两baseline原raw分别2.23275658e-8/1.81156760e-8，capture/control完全相同；smoke第一层校准1.12493/1.06401通过。全部4+2载荷复用，原文件不改，无重采。完整metadata门槛仍缺network16原回执，剩余252未启动。下一步查可恢复的原云回执/归档证据；不能将数组数值核验代替遗失的源审计。

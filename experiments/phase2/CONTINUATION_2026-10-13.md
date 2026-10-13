# 2026-10-13 冻结评测扩展与方法级 pilot

本轮从12:12（北京时间）接续。当前最佳线上仍是 V37b+H18，#335678 adjusted **4.293779716792071e-9**。第一阶段目标5.779e-10未达成，不递进阈值。

## V38-S4-H18：资源余量与分数代价

唯一主要变化为Strassen递归层数默认上限5→4，依赖该上限的hub/shared-basis默认值相应变为4。保持合法算子、原始数据与120/0.4/5秒约束，不修改计量SDK。候选SHA `669f9c1ac1b191ca8adacfeee9aa01cf48336c490735ed7686c3631f91ca2350`。

| 指标 | before V37b+H18 local8 | after V38-S4-H18 local8 |
|---|---:|---:|
| adjusted（包含全部失败） | 0.34217436909220283 | 4.876618851087354e-9 |
| raw MSE（包含全部失败） | 0.34217438065589345 | 2.2660965326437577e-8 |
| 失败网络 | 3/8 | 0/8 |
| 最高residual秒 | 0.4171564950 | 0.3809135860 |
| 平均FLOPs | 见原报告 | 473328782033 |

before的5/6/7网是超时后的零输出fallback，不能把其巨大误差下降解读为算法精度改进。V38暂为资源备选；线上最佳不变，未上传V38。

12:38完整严格RPC8确认完成零失败，adjusted **4.876791740532289e-9**、raw **2.2660965326437577e-8**。同候选local8绑定源码/manifest/数据/seed/约束后，逐网raw MSE完全一致，RPC每网多16,777,216 FLOPs为已知astype差。对同runner、同输入的V37b+H18严格RPC8，adjusted **4.6043530010738635e-9→4.876791740532289e-9（+5.91698%）**，raw变化-0.08276%；不能称新最好算法。见[严格RPC与local一致性](../../results/phase2/v38_strassen4_h18_r02_strict_rpc8_score.json)、[同runner成对对照](../../results/phase2/v38_strassen4_h18_r02_vs_parent_strict.json)。

证据：[逐网前后](../../results/phase2/v38_strassen4_h18_r01_before_after.json)、[原始报告](../../results/phase2/v38_strassen4_h18_r01_local8.json)、[manifest](../../results/phase2/v38_strassen4_h18_r01_local8.manifest.json)。Mini8已经反复研究，不作为未见泛化。

cProfile单网只用于定位调用结构：`_bufh`1841次，内部0.024秒、累计0.162秒。剖析增加耗时，实际residual0.807秒失败；保留但不作资源验收。计量wrapper累计时间不等于residual，尚未分离每个结构的因果贡献。[诊断摘要](../../results/phase2/v37b_h18_local_profile_r01_summary.json)。

## M04：已知期望的首层ReLU代理

一手来源：[Art Owen，Monte Carlo theory, methods and examples，第8章](https://artowen.su.domains/mc/Ch-var-basic.pdf)，§8.9公式8.30–8.34及独立pilot段落。原文要求代理期望可知；用与最终估计流独立的pilot拟合系数可保留无偏性前提。PDF只保存在ignored研究缓存，不再分发，SHA `26b03406aefdcf64fcd2e19532a9c6fd4d60947273bdbafcc3aa7bd67f683f9e`。

本轮代理是 `g(x)=(ReLU(xW0)-m0)J`，其中`m0_j=||W0[:,j]||/sqrt(2π)`，J由解析首层均值处的后续硬门切线固定，理论上E[g]=0。每个输出beta在4096个训练样本拟合，再用8192个独立样本测残差方差，重复2个seed。此为offline NumPy诊断，未进入比赛计算。

| seed组 | before采样方差均值 | after残差方差均值 | 方差比 | 同2**41预算预计MSE before→after |
|---|---:|---:|---:|---:|
| 61013000/1 | 0.0788432693 | 0.0766723025 | 0.9724648 | 1.2036405e-6 → 1.3534626e-6 |
| 61013002/3 | 0.0781340909 | 0.0759863350 | 0.9725119 | 1.1928140e-6 → 1.3413535e-6 |

包括代理构建与训练样本成本后，两次预计MSE都恶化约12.45%。否决这一个硬门切线代理的直接移植；M04仍为partial，未否决其他代理。表中是经验方差和解析成本外推，不是benchmark分数或已实现参考精度。float32中心化可能有数值偏差；一网两seed也不构成未见网络确认。

后续单因素pilot r02已完成：保持首层特征、解析期望、4096/8192样本数与独立seed不变，只用训练流的ridge回归替代硬门切线传输。固定ridge系数1e-3，不搜索held最优参数；把回归矩阵和solve构建成本加入同预算核验。方差比降到0.954749/0.961338，但同预算预计MSE比仍为1.093508/1.101054，恶化9.35%/10.11%，不移植此代理。实现为`m04_first_relu_regression_pilot.py`，结果见[原始r02](../../results/phase2/m04_first_relu_regression_pilot_r02.json)。不是benchmark分数，M04仍未完成。

[实现](m04_first_relu_control_pilot.py)、[完整逐神经元方差与成本](../../results/phase2/m04_first_relu_control_pilot_r01.json)。

## 新公共分片的冻结协议

官方commit `aa99830fdc09fad15407b10e8e3459d3e18bba0a` 的分片2只有7网（ID16–22），Arrow SHA `f4ef83bd0d40bfcd7b72c7ad6e0a2fd65b92988fe4b391b67d4be6ac7c31ac6a`。在新分片评分前冻结V29与V37b+H18，不据此修改候选。两者使用同一CRC/SHA验证输入缓存、官方派生seed、原始reference、SDK版本及严格RPC约束，串行运行。

准备结果见[数据证据](../../results/phase2/mini_shard02_dataset.json)和[冻结协议](DEVELOPMENT_PROTOCOL_20261013.json)。本项目没有提前用这7网进行调参；上游作者是否使用未知，不能称最终未见holdout。数据准备不是评测完成，也不能把7网当8网或完整Mini100。

本轮工具回归24测试通过；新增脚本ruff检查通过。后续保存两候选逐网raw/adjusted/FLOPs/时间/内存/失败，再补本页结果。

12:33：新7网双方严格RPC完成零失败。

| 指标 | before V29 | after V37b+H18 | 变化 |
|---|---:|---:|---:|
| adjusted | 5.485889416202945e-9 | 4.303414283711552e-9 | -21.55485% |
| raw MSE | 2.1561328976206695e-8 | 2.1186933858124187e-8 | -1.73642% |
| 平均预算使用率 | 25.48386% | 20.35752% | 主要降费 |
| 失败 | 0/7 | 0/7 | — |

after最高wall58.752507秒、residual0.185619秒、client峰值299632KiB，7网adjusted均改善。逐网、数据/工具/输入哈希与原始报告绑定见[成对证据](../../results/phase2/sprint_public_v37b_h18_r08_shard02_before_after.json)。最佳候选严格RPC覆盖为23个公共网络，Mini100/最终独立确认仍未完成。12:33:38开始V38同候选local/RPC Mini8确认；当前最佳线上不变。

## M05一手方法与成本核验

补读[Monchot等ICML2023原论文](https://proceedings.mlr.press/v202/monchot23a/monchot23a.pdf)§3.2–3.4与Algorithm1，以及作者[WGMprop实现](https://github.com/PaulMcht/WGMprop)，固定commit `915eaacea0ae3906536b8ffc6662a84b43b8487c`。论文实际先分裂输入高斯混合，比较完整网络非线性与局部Taylor的sigma-point结果，使用Wasserstein上界决定继续分裂；本文实验不使用merge步骤。不能把普通逐层Gaussian闭合或任意固定混合称为完整复现。

原实现UT点数为2r+1，按输入协方差最大特征值选子空间。比赛输入I1024特征值相同，截掉维度会主动改变分布，必须另行证明误差，不能直接复用该降维依据。全维每component需2049次dense网络forward，解析下界约68.753GFLOPs，因此2**41最多31个component，尚未包括Taylor、矩、criterion、分裂与eigh成本。这是成本外推，非flopscope实测。公开实现含TensorFlow/自动微分，需要重写为允许算子后完整验收。

M05由pending变为partial_mapping；下一步是小shape完整split/UT/Wasserstein核验及有限混合偏差诊断，暂无benchmark after分数。[来源SHA与成本证据](../../results/phase2/m05_source_audit_r01.json)。

进一步复现原实现的未分裂UT边界：在本项目一网的第一层，原`κ=n-3=1021`的均值为`sum_i abs(Wij)/(2sqrt(n+κ))`，对解析ReLU均值MSE0.0271809151，平均偏差-29.2148%。κ=0诊断对照MSE2.16877e-5，仍远高于累积量方法的量级。显式sigma points在宽3/5/9与简化式误差<1e-12。这个结果解释为何必须核验自适应分裂；不能用它否决完整WGMprop。[诊断](../../results/phase2/m05_unsplit_relu_gate_r02.json)。

用公开三分量模板的rounded参数计算标准一维高斯的解析矩，分裂后variance0.9978235063，ReLU期望0.3982864984（原高斯0.3989422804），偏差-0.0006557820；分裂不是精确保分布，不能把弱混合reference作为比赛真值。静态代码还显示Taylor项用了`reduce_mean(weight*squared_error)`，而论文§3.4.2是归一化权重的求和，前者根号项多出`1/sqrt(2r+1)`缩放；记录为论文/实现核对问题，尚未运行完整上游系统，不直接宣称作者方法错误或复现成功。

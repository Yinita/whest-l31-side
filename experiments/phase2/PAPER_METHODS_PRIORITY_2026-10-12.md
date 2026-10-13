# 论文方法级对照与优先级升级

2026-10-12：用户明确要求纳入消融文档并提高优先级。本文件是当前129项细粒度消融之上的执行优先级补充，不表示原任务完成，不重置队列、运行状态或证据。当前实验安全结束并收集结果后，优先推进以下方法级工作；不并行运行大网络，不中断或修改已冻结候选。

## 优先顺序

P0：M01论文算法映射与复现规格 → M02精确协方差方法对照 → M03累积量算法阶数/表示对照。P1：M04解析控制变量与残差采样 → M05条件积分/高斯混合传播。已具备可信收益的冻结候选继续完成必要验收；其余重复局部调参、纯计算微优化和学习特征扩展让位于这些方法级任务。准备度不足先读论文、推导和成本核验，不能猜公式。

## 方法与验收

| ID | 优先级 | 来源与方法 | 当前覆盖 | 下一动作和验收 |
|---|---|---|---|---|
| M01 | P0 | Wu等2026：累积量、Hermite、basic/augmented/factorized | V29主干、H/O来源诊断；并非论文全变体复现 | 以原论文第4/5节和补充材料建立论文算法—官方commit—本地实现—证据映射；分别标明精确恒等式、截断假设、理论适用条件及经验结果 |
| M02 | P0 | Wright等2024：解析均值/协方差传播 | 教学基线与H04 pair核诊断；完整方法未确认复现 | 独立复现论文的ReLU均值/协方差更新，区分精确高斯矩和深层高斯闭合误差；建立完整K2方法及V29仅替换pair核两个独立对照。先核对正在执行的discussion exactpair实验，复用证据，勿重复启动 |
| M03 | P0 | Wu等2026：高阶累积量与因子化 | K3、K4来源和K5/K6部分探针 | 官方basic/augmented/factorized选择成本可行的完整变体，分别冻结K2/K3及可行K4；统一预算测精度—成本曲线。不可将K5边缘探针称为完整K5算法，不把小shape生成器核验称为端到端复现 |
| M04 | P1 | 控制变量与Rao–Blackwell标准估计理论 | L28只有方差/成本门槛，已知期望代理仍待实现 | 构造有可计算期望的g，估计E[g]+mean(f-g)；独立训练/评测网络及采样流，测残差方差、总成本和无偏性前提。学习型输出修正不等于无偏控制变量 |
| M05 | P1 | Monchot等2023：输入不确定性传播；条件积分 | L29固定替代分布负结果；尚无完整条件方法 | 先核对原文实际算法及分布分解，构造有限分量混合或低维条件方法的独立规格与成本上限；比较同预算纯采样。L29负结果只否决该替代家族，不能否决整条路线 |

所有方法先做最小独立数学核验和成本筛选，再按冻结协议做network0/Mini8成对评测；超预算方法保留诊断，不放宽比赛约束。方法不同不强求同一父算法，但必须同网络、同reference、同预算规则，并明确各自完整算法与单因素消融的区别。记录逐层误差、raw/adjusted、FLOPs、时间、内存、失败、代码/数据哈希。沿用现有独立确认与选择偏差账本。

## 理论问题优先于无解释的局部消融

1. 在成本可控的小/中网络上预定宽度、深度、截断阶数网格，检验误差与成本尺度；O28第一层结果不足以验证深层定理。
2. 分离非线性展开截断、高阶状态丢弃、表示压缩及数值误差。单个来源恢复的负结果不证明完整方法无效。
3. 研究局部误差如何传至最后层，关联L24/L25，避免只看最终分数。
4. Wu等的渐近定理有固定深度、多项式激活及精度尺度条件；ReLU竞赛结果单独报告，不声称已获得其完整理论保证。

## 一手来源

- Wu等，2026，Estimating the expected output of wide random MLPs more efficiently than sampling：https://arxiv.org/abs/2605.05179 ，全文 https://arxiv.org/html/2605.05179v2 。官方代码 https://github.com/alignment-research-center/mlp_cumulant_propagation ，沿用项目冻结commit，另行变更须记来源。
- Wright、Nakahira、Moura，AISTATS 2024，An Analytic Solution to Covariance Propagation in Neural Networks：https://proceedings.mlr.press/v238/wright24a.html 。公式需从论文核对，不以教学gain近似冒充完整实现。
- Monchot等，ICML 2023，Input uncertainty propagation through trained neural networks：https://proceedings.mlr.press/v202/monchot23a.html 。属于相关输入不确定性研究，不是本比赛专门解法。

## 执行协调

执行agent下一轮先读取本文件，把M01–M05按现有队列schema合并为独立方法级任务并保存优先级；保留全部129项状态、result、evidence和active_experiment。不要运行build_ablation_backlog.py覆盖已有执行证据。方法级任务使用M编号，暂不抢占v30编号。同步更新结果生成器后才将方法任务纳入自动汇总，未完成项不填虚假after分数。


2026-10-13执行更新：M04两种已知期望代理含费用后未通过门槛。M05已完成完整小对象、精确分裂/UT误差拆分、真实1024宽float32有限分裂诊断；9叶/13评估raw0.07479，占预算90.86%、未收敛，当前实现降P2，完整论文路线仍partial。O05继续优先：三次ReLU Wick k1..5独立Stein积分通过、合法K31/K211中心化合同及无需n³输入的CP capped中心化已验证；实际来源映射与整网替换仍待。最新前后分数和优先级以[消融结果](ABLATION_RESULTS.md)及[优先级表](PROMISING_PRIORITY.md)为准。

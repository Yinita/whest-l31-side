# WhestBench Phase 2 开发入口

更新日期：2026-10-13。当前决策依据为 [项目报告](ARC_WhestBench_Project_Report_2026-10-10.md)。先投入 48 小时完成强基线复现与诊断，再按独立网络上的误差和成本决定后续投入。

官方Resources逐项核验见[页面覆盖审计](experiments/phase2/DYNAMIC_CONTENTS_COVERAGE_2026-10-13.md)：工程已实际使用，公共数据仅16网，Mini100/Full1000、完整论文比赛移植与独立验收仍未完成。V37b+H18原RPC Mini8 adjusted **4.6070113621e-9**，8网零失败，主要是成本收益；阶段目标未达。

后续核验：同候选local8已完成但3网residual超过400ms，失败保留；新严格RPC执行器故障验收已完成，完整候选运行中。线上335678已graded **4.2937797168e-9**，本队当前最佳、rank67；335677原样V37b为4.3922571600e-9。最新证据与执行状态见[严格RPC验证](experiments/phase2/RPC_STRICT_VALIDATION_2026-10-13.md)。

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

## 连续研究进展

取消定时触发后持续推进。阶段1 adjusted目标仍为5.779e-10，未达标。2026-10-12已完成Gaussian pair三阶尾项、USE-only diagonal spill、边缘峰度平方、单层新生四阶树无迹残差等真实network0快筛；成本与误差成对记录，负结果保留，不将小规模代数核验当作提分。最后层残差时间消融也无收益；原V29当前环境单网控制已完成：FLOPs完全一致，raw相对冻结报告漂移−0.0157%；最后层残差候选相对当前控制也无收益。后续triangle rank16和doubleedge final-only快筛亦未改善adjusted。

论文代码实际仓库是[mlp_cumulant_propagation](https://github.com/alignment-research-center/mlp_cumulant_propagation)，已冻结commit93d091a4c26c042bfffa28f2e76a81bc0aba94bb；此前误查模块名mlp_kprop的404已纠正。官方组合生成器与V29现有pair表精确对照发现22项中19项已有，剩余3项即本轮pairq3，不重复添加已有系数。详见[研究日志](experiments/phase2/RESEARCH_LOG.md)。

## 2026-10-12 用户优先级升级

论文方法级对照提升为P0/P1，当前运行安全完成后优先于其余局部消融。执行顺序、论文来源、已有覆盖和验收见[论文方法级对照](experiments/phase2/PAPER_METHODS_PRIORITY_2026-10-12.md)。保留原129项状态与证据；不得重建队列覆盖结果。

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

## 消融执行队列

用户要求先准备清单再交低推理模型：已整理[129项待验证消融](experiments/phase2/ABLATION_BACKLOG.md)，每项含预留v30.xxx版本、before/after、详情、依赖、成本和验收。机器队列见`experiments/phase2/ablation_queue.json`，执行提示见[交接文件](experiments/phase2/LOW_REASONING_HANDOFF.md)。71项算法/计算、43项诊断、15项合同；59项需先推导，不允许执行模型猜公式后正式评测。用户已授权执行全部队列。真实结果持续写入[消融结果](experiments/phase2/ABLATION_RESULTS.md)，运行身份和进程以research_state.json为准。

当前H18完整Mini8 adjusted 5.646816493872825e-9（相对冻结基线−2.287%，零失败）；本地RPC完整Mini8 adjusted 5.647856469248522e-9，零失败；独立确认仍未完成，目标不递进。C03/C05增费未通过条目门槛，C06微小差待重复控制归因。逐项真实结果见消融结果。

C16 r03固定16行调度Mini8 adjusted 5.750992775997765e-9（−0.4843%），8/8零失败，最大residual0.2527s；r02原2/8超residual失败保留。RPC及独立确认仍待。

C16 r05完整Mini8 RPC adjusted 5.7525138595226505e-9、8/8零失败，最大residual0.1502s；r04输入CRC失败单独保留。H22未来权重sketch单网adjusted−2.5917%，完整Mini8 r03运行中；H19/H20 dense metric slices分别adjusted+15.791%/+17.750%，否决当前高费实现。

H22 r03完整Mini8 adjusted 5.764482499205733e-9（−0.2509%），零失败，单网收益在全集明显缩小；RPC复验运行中。部分推导合同不代表任务完成，独立确认与十倍目标均未通过。

2026-10-12连续执行补充：H22完整Mini8 RPC adjusted5.761097169765404e-9、raw2.263240195254923e-8，8/8零失败。C01r02完整Mini8 local adjusted5.750454989509922e-9、raw2.2708409819216513e-8（计算量−0.5180%，raw+0.02463%），同hashRPC adjusted5.748034693185937e-9、raw2.269812404698257e-8，8/8零失败，local最大residual0.3221s。两者均未获独立确认，当前最强Mini8仍为H18，阶段阈值不递进。

L20预注册8个1024×16新seed网络权重已实体生成，无reference labels。L19实测单线程约3700samples/s，1e9×2seed×8network约50CPU天的吞吐外推；4096样本pilot只是弱噪声诊断，不能替代强reference或验证10倍收益。O27冻结特征后正在进行公共reserved shard无label状态采集，原8GB subprocess C01资源对照已串行安排。

O27公共reserved shard状态分布对照已完成：冻结variance/alpha/标准化D3/K4代理/Phi特征，8/8网络无target访问；最大平均分位点偏移/developmentIQR0.266。该结论为描述性状态诊断，不能用于声称公共分片仍未见或独立分数提高。C01原8GB subprocess资源对照继续运行。

L03 r04完整合法计费networkLOO完成：每fold只评测训练排除的一个网络，当前成对V29 adjusted6.061515659222411e-9→candidate6.096439781475735e-9（+0.5762%），raw2.269752985561979e-8→2.282830258160118e-8，8fold零失败。该固定13feature线性模型否决，不能将fold0微幅改善当整体收益。singleton逐网络控制因pool生命周期不同于连续Mini8，必须用本次成对控制，不能替换冻结Mini8阶段门槛。

L21已冻结H18 hash da7154bb8170c9deda98ed82b97c826bafbea7740054e8b04567dcf2837a7891，串行执行公共8–15当前V29与H18配对确认。O27已查看该集合状态，上游公共调参重叠未知，不称独立验收。接续实验为L19 antithetic reference弱pilot和C26实际predict shape序列8GiB资源审计。

H18公共8–15一次冻结配对确认完成：当前V29 adjusted5.896492018734529e-9/raw2.316187108775125e-8→H18 adjusted5.757025246568847e-9/raw2.2611129413263598e-8，8/8网络adjusted均改善，aggregate−2.36525%，双方零失败。仅公共确认；状态先前已查看、上游调参重叠未知，不代替独立reference验证，阶段1目标不递进。

L19 antithetic参考两个独立4096pair种子相同forward次数finalvariance ratios0.91035/0.90858，约9%降方差；弱pilot不是benchmark labels，不能声明十倍目标。C26记录器tuple处理错误已保留原log并修复重跑。

2026-10-12 H29真实来源子集接入：r07误用半系数，r08/r09导数阶次为替代轨道w2*w1*w1，不能否决目标w1*w2*w2；保留所有报告。目标轨道r10已通过独立对称CP显式tensor小shape核验，单网正式计费筛选中。C26 RPC同进程64×4→1024×16→64×4→1024×16四次无异常，client peakRSS959580KiB；逐次VSZ补测待完成，不能提前标记全部验收。

O12 exact表示CP成本诊断、C26真实local/RPC形状生命周期诊断、C28最后层deadLA/LP复用合同已补齐。C28 local单网adjusted6.1167127894412344e-9，RPC6.120838343176817e-9，均零失败；local FLOPs与控制相同，RPC固定astype增费保留，不宣称精度提升。H29正确目标阶次first1子集adjusted6.116832334984139e-9→6.251611877719655e-9，负结果，完整来源仍待。剩余74项partial，十倍门槛未通过。

K5 Gaussianbirth对角final子集Mini8 local5.7773073237326585e-9，RPC5.777750550380405e-9均8/8零失败；同周期V29控制5.777513736227004e-9，差仅−0.00357%，无法认定可靠收益。fullK5/cross/nonGaussian门槛仍未覆盖。L18/L34/L35合法计费解释力、冻结模型推理及证据账本已补齐；当前60项相应快筛/诊断完成，69项partial，不代表十倍达标。L01同周期trace/repeated control首网输出差超过重复包络，保留partial，原始证据不覆盖。

H31真实layer14固定4output方差gate否决（optimistic250probes SE/signal0.344–1.464），完成negative。H06历史90条未剪枝重建合法评测adjusted6.2350e-9，劣于控制；历史依赖未确认，仍partial。H08/H09/H11/H12固定layer14first16出生子集均零失败但增费后adjusted恶化（6.141–6.144e-9 vs6.1168e-9），完整来源不宣称验证。当前61项快筛/诊断完成，68项partial。

H13完整B2固定第14层出生、1024全列单次运输零失败：初版adjusted6.2450e-9，identity精确重组降为6.1516e-9；raw约−0.53%仍不足支付费用，完成negative。H07/H08全列单次出生分别adjusted6.1823e-9/6.1892e-9，负；历史来源覆盖仍partial。当前62项相应快筛/诊断完成，67项partial。

H17完整Gaussian二阶矩阵残差（全出生层）去重合同通过，同期单网V29 adjusted6.116832334984139e-9→7.150011123478984e-9，raw2.290466483145792e-8→2.2978715819022e-8，零失败；增加费用且无收益，完成negative，不扩量。当前63项相应验收完成、66项partial，129项尚未全部完成。

H06历史依赖与90条未剪枝表逐项合同补齐；此前r04单网adjusted6.23498e-9（同期V29为6.11683e-9）负结果正式验收。上游94条注释过时，不猜补4条。当前64项相应验收完成、65项partial。

讨论全文97主题301回复、29附件已存档索引。新leaf8完整Mini8 adjusted5.573952118067259e-9（−3.5478%）、8/8零失败，当前本地最强；费用收益，精度未改善，RPC待验。

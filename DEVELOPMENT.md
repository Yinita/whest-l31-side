# WhestBench Phase 2 开发入口

2026-10-13十倍精度接续：两网GH2条件积分raw−11.91%/−7.37%但费用翻倍，adjusted未改善；r10cheapdiag否决（raw+33.52%），r11fullcov响应raw−7.00%但adjusted+105.61%，均零失败。下一单链解析径向分布版本正在做实际source合同；云逐层矩采集首版大量timeout，保留全部失败，不记候选收益。阶段1目标5.779e-10、当前线上最佳V37bH18均未改变。见[真实before/after](experiments/phase2/TENFOLD_PROGRESS_20261013.md)。

最新检查点（2026-10-13）：V37bH18严格RPC累计65公共网零失败，adjusted **4.4596111651e-9**，raw **2.1926280101e-8**。同runner成对49网，V29→V37bH18 adjusted下降21.54%、raw下降1.76%、FLOPs下降20.12%；49/49 adjusted改善、34/49 raw改善，主要收益来自降费。[65网汇总](results/phase2/sprint_public_v37b_h18_strict_public65_r06.json)与[49网成对归因](results/phase2/frozen_strict_public49_paired_r04.json)保留逐网证据。阶段1目标5.779e-10未达，Mini100未完成，公共集不称最终独立holdout；线上最佳仍#335678的4.2937797168e-9，未新上传。

本轮新增真实高阶组件：r21利用重复P与rank-one feed精确收缩，全14来源成本下降33.06%；r22年龄裁剪及r23列抽样偏差较大，不接入候选。r25 float32完整K31中心化精度失败，r26仅改float64后独立核验通过，1024宽15类图项成本2.678亿FLOPs。均无整网分数，完整K211和proxy去重仍待。版本、before/after、失败与成本见[真实来源记录](experiments/phase2/REAL_SOURCE_K211_2026-10-13.md)。

十倍专项已进入整网验证：[本轮before/after](experiments/phase2/TENFOLD_PROGRESS_20261013.md)。新完整retained非对角core→LS候选单层仅raw改善0.09%，全1–14层raw恶化13.02%；均严格RPC单网零失败，成本使adjusted变差。已否决这些版本，不称十倍。当前单因素beta归因与局部新增误差模型继续推进；V37bH18仍最佳。

更新日期：2026-10-13。当前决策依据为 [项目报告](ARC_WhestBench_Project_Report_2026-10-10.md)。先投入 48 小时完成强基线复现与诊断，再按独立网络上的误差和成本决定后续投入。

官方Resources逐项核验见[页面覆盖审计](experiments/phase2/DYNAMIC_CONTENTS_COVERAGE_2026-10-13.md)：工程已实际使用，最佳候选严格RPC累计覆盖58个公共网络；Mini100/Full1000、完整论文比赛移植与独立验收仍未完成。V37b+H18原RPC Mini8 adjusted **4.6070113621e-9**，8网零失败，主要是成本收益；阶段目标未达。

后续核验：同候选local8已完成但3网residual超过400ms，失败保留；新严格RPC执行器故障验收已完成，Mini8 **4.6043530011e-9**、扩展公共8网 **4.6350586013e-9**，各8/8零失败。扩展集较V29 adjusted下降21.39%、raw下降1.55%，主要降费；这些公共网曾用于研究，不能称最终未见holdout。线上335678已graded **4.2937797168e-9**，本队当前最佳；上次榜单快照rank67；335677原样V37b为4.3922571600e-9。最新证据与执行状态见[严格RPC验证](experiments/phase2/RPC_STRICT_VALIDATION_2026-10-13.md)。

12:33接续：V38-S4-H18仅将Strassen递归上限5→4，local8零失败，最高residual0.380914秒，adjusted4.876619e-9；这是成本略高的资源备选，未替代线上最佳。同候选严格RPC8也零失败，raw逐网与local完全一致；同runner相比parent adjusted+5.92%，不提升为线上最佳。新公共分片2（7网、ID16–22）双方严格RPC零失败，V29 adjusted5.485889e-9→V37b+H18 4.303414e-9（-21.55%），raw改善1.74%，最佳候选覆盖23公共网；Mini100未完成。M04两个已知期望代理pilot含成本后预计MSE分别恶化12.45%和9–10%，M05完成原论文/代码映射及未分裂UT偏差诊断；完整方法仍未完成。详见[本轮记录](experiments/phase2/CONTINUATION_2026-10-13.md)。

13:39接续：分片3（ID23–29）V29/最佳候选严格RPC双方7网零失败，adjusted **5.565246e-9→4.337734e-9（−22.06%）**，raw **2.184545e-8→2.132122e-8（−2.40%）**，逐网adjusted均改善。最佳候选累计严格RPC覆盖44公共网，合并adjusted4.532173e-9/raw2.228391e-8、零失败；阶段1目标未达。M05小对象分裂/UT/Taylor/W2正确性及非对角W2独立合同通过，1024宽float64单分量完整实测307371364362 FLOPs（预算13.98%，约7次评估），尚非官方资源验收。精确一维深度5→12的总误差降低包含UT与分裂偏差抵消，逐分量阈值通过不等于总误差界。见[M05小流程](experiments/phase2/M05_SMALL_PIPELINE_2026-10-13.md)。M05真实1024宽9叶/13评估进一步诊断raw0.07479、占预算90.86%、阈值未达，当前实现降到P2，未提交。已[冻结公共验收队列](experiments/phase2/PUBLIC_ACCEPTANCE_QUEUE_20261013_r03.json)，分片4已完成V29adjusted5.712536e-9→最佳4.459808e-9（−21.93%）、raw下降2.20%、双方7网零失败；分片5已完成V29adjusted6.107346e-9→最佳4.827658e-9（−20.95%）、raw下降1.13%、双方7网零失败；分片6（ID44–50）14:54已启动同候选严格RPC串行验收并启用状态同步，复用原始reference；不重复Mini8或上传原包。

14:51成对归因：新公共分片2–5共28网，双方同严格RPC/原reference，V29 adjusted5.717754e-9→最佳4.482154e-9（−21.61%），raw下降1.85%、FLOPs下降20.12%；28/28 adjusted改善，19/28 raw改善，双方零失败。见[冻结28网成对汇总](results/phase2/frozen_strict_public28_paired_r01.json)。主要仍是计算收益，不能把所有adjusted收益解释为精度提升。

15:25接续：分片6（ID44–50）已完成，全部6阶段输出及候选/工具哈希核验通过；V29 adjusted **5.557312e-9→V37b+H18 4.352294e-9（−21.68%）**，raw **2.180149e-8→2.138083e-8（−1.93%）**，双方7网零失败。最佳候选51公共网去重汇总adjusted **4.507483e-9**、raw **2.215995e-8**，零失败；同runner新公共35网成对adjusted下降21.62%、raw下降1.87%、FLOPs下降20.12%。见[51网汇总](results/phase2/sprint_public_v37b_h18_strict_public51_r04.json)和[35网归因](results/phase2/frozen_strict_public35_paired_r02.json)。分片7（ID51–57）已冻结原候选并启动串行严格RPC，不提前计入完成网络。阶段1目标仍5.779e-10。

高阶来源接入新增[真实B1组件验证](experiments/phase2/REAL_SOURCE_K211_2026-10-13.md)：从人工CP推进到真实layer14来源的全部1024列、实际pk1v及严格重复指标剔除。1024宽固定条目最大误差7.61e-10，完整组件计费12,907,085,824 FLOPs；实际均值与Gaussian均值代理使该组件改变1.32%。当前只有一个来源的K111 Wick/中心化轨道，其他来源、feedback/feed/M、其他K211轨道及完整整网接入未覆盖，不能报告candidate提分。

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

15:37接续：分片7双方7网严格RPC零失败，adjusted5.3961065e-9→4.2720295e-9（−20.83%）、raw下降0.95%；全部58网汇总adjusted4.47906655e-9/raw2.20170747e-8、零失败。同runner公共42网成对adjusted下降21.50%、raw下降1.72%、FLOPs下降20.12%，42/42adjusted改善、28/42raw改善。见[58网汇总](results/phase2/sprint_public_v37b_h18_strict_public58_r05.json)及[42网归因](results/phase2/frozen_strict_public42_paired_r03.json)。当前高阶真实完整来源r14/r15串行工作流接续，阶段阈值不变，无新线上上传。

15:40结构进展：真实layer14单来源的B1+feedback/M/feed完整CP映射通过全D3/D21方程对照；随后仅做精确收缩重组，6个矩阵乘法减为3个，相同输入组件FLOPs38,839,413,760→19,519,403,008（−49.74%），输出相对差3.22e-7、独立条目核验通过。见[来源证据](experiments/phase2/REAL_SOURCE_K211_2026-10-13.md)。只有一个来源和一个K211轨道，完整K31/K211生产与整网提分仍待。分片8原始数据准备中，继续冻结同候选，不重跑Mini8。

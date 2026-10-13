# 十倍目标整网推进：2026-10-13

阶段1仍要求冻结Mini8 adjusted <5.779e-10，并独立网络成对10倍、零失败、严格资源验收。当前未达标。评分最低乘数0.1，因此raw必须<5.779e-9，仅降费不充分。

## 新完成的实际网络结果

以下均同网络0、原始reference、immutable RPC inputs SHA9e53442b88a0da22e8b9dafa26aa92318b38af95ab284776fa0e972b991854b4，同工具flopscope0.12.1/whestbench0.16.1，预算2**41、120秒predict、0.4秒residual、5秒setup，线程1。不是Mini8均分或独立泛化证据。

| 版本/证据stem | raw MSE | adjusted | FLOPs | wall / residual秒 | 失败 |
|---|---:|---:|---:|---:|---:|
| tenfold_parent_h18_samecycle_r01 | 2.2281179568e-08 | 4.8714160494e-09 | 480,780,523,639 | 46.567 / 0.193766 | 0 |
| tenfold_corelambda_li14_r02 | 2.2261087196e-08 | 6.5716805352e-09 | 649,172,171,955 | 51.174 / 0.241808 | 0 |
| tenfold_corelambda_li14_r03 | 2.2261087196e-08 | 5.9590167706e-09 | 588,651,234,483 | 52.924 / 0.224646 | 0 |
| tenfold_corelambda_alllayers_r04 | 2.5181893903e-08 | 1.9380980907e-08 | 1,692,455,217,747 | 80.000 / 0.394487 | 0 |

r01完整矩阵source/core版本local资源失败：residual0.439074>0.4，fallback分数0.943476不能当算法raw；保存原始JSON与log。r02采用28图项3GEMM融合、source direct scalar LS numerator；r03再共享old-source HQ/HQU，raw与r02逐位一致，cost降低9.32%。r02/r03都只改li14 base lambda；微小raw收益0.09%被成本抵消。r04只改介入层为li1..14，raw恶化13.02%，否决全层标量压缩改善假设。

r05单因素检查关闭新在线LS之后的经验beta（li>=2，初始li1不改），实际严格RPC资源失败：residual0.404978>0.4，FLOPs1692425828653、wall81.205109s、client294476KiB。fallback0.943476不称算法raw，不放宽规则。

## 组件及限定

已完成实际K4三指标power/当前meanfeed、K31、28lower-support、postCovCov、tree出生center、全部14来源1024出生列jointcentering。官方固定93d091a4，所有candidate单文件合法计费算子，无truth/reference依赖；完整仅指给定parent represented source与K4 proxy的retained非对角生产，仍有source confinement、平均metric和K4再生近似。g_prev对角与父K4/K22一致；不把offdiag行和误加到diag。

来源scalar投影独立n8/32、float32/64、masked sharedbasis和wholehelper合同通过。mixedfloat32 CP→float64 precision difference已列，不称位级精确。新候选ruff排除父代码既有F401/F841/E402/I001/E702后通过；实验helper完整ruff通过。

## 另一条精度机制

末层约80%平方误差来自继承mean transport（reference oracle离线定位）。新local innovation回归先按网络LOO：target−prediction误差减Phi*W前层误差作为local label，再固定状态递推；不直接回归累积误差。cached线性pilot提升1.52%，5/8改善；固定HistGradientBoosting深度3/100迭代/lr0.05/minleaf128对照raw2.270165994e-8→2.218988583e-8（−2.25%），仍未通过预注册rawratio<0.25必要门槛。没有解封public或改生产候选。Angular真实整网分布pilot只prepared/selfcheck，不宣称收益。

现最佳仍V37bH18线上335678 adjusted4.2937797168e-9。新负候选不上传，阶段目标与冻结基线不改。

## 更改下一实验对象

父LAM表拟合的是下一层preactivation，当前LS拟合postcore不是同一目标。新r06先保存li14完整Goff，经最后层W真实运输到dG以诊断压缩损失；保持父metric近似，末层trim无需新增sourcefeed持久表示。此是新机制测试，不宣称十倍充分，因为早层误差仍继承。后续应采用transport-aware LS或更丰富core。

真实n1024 lower-support融合已完成：121106763776→13396717568 FLOPs（−88.94%），相对输出差4.50e-16；两网络均核验。这仍是费用合同不是新score。

r06完整矩阵真实末层运输已严格RPC零失败：raw2.215712413545e-08（相对同cycleparent下降0.557%）、adjusted8.651172836984e-09，FLOPs858601059417，wall48.500416/residual0.199326s。成本抵消精度收益，不提升最佳。

r07下一preactivation目标整链已严格RPC零失败：raw2.322639502950e-08、adjusted2.041687781313e-08，FLOPs1933024434477，wall76.952829/residual0.372440s。相对parent raw恶化4.24%，不推广。该完整retained-offdiag LS系列未证实十倍原理，停止细调。

多层已知均值Gaussian-linear控制变量：两独立heldstream variance ratio0.66484/0.67292，约3倍sample成本。乐观全预算float32误差约2.5e-6，不通过预注册5e-4variance ratio门槛，停止该路线。不混成benchmark分数。

下一新分布机制r08：rank1输入条件积分，4固定Gaussian quadrature节点，残余1023维完整Gaussian K3母链；方向仅weight-only Gaussian diagonal skeleton的outputmean sensitivity，不跑额外parent全链、无reference读取。第一层mean=tq/Cpre=Gram−qqT，其余保持parent。四链同一计费context，不重置预算；高阶接口/conditional分布校准/4nodequadraturebias与资源均待实际验证。

r09 GH2 rank1条件链实际strict1零失败：raw1.962775897368e-08，相对parent改善11.908%，adjusted8.339351717318e-09（更差），FLOPs934310859798，wall99.503029/residual0.381472s。有raw信号，但新候选未达到adjusted改善门槛，不成为最佳、不得上传。

8公共网络的首层解析quadrature gate（无targets）：GH2 mean MSE2.9204e-12、GH4 5.7782e-21、GH8/16约3e-32。首层单方向mean variance占比0.20%，说明明显firstlayerquadraturebias不是当前深链残余主因；深层积分误差仍未知。

下一定位云CPU单publicnetwork0两独立32m streams，全16层pre/post mean/variance/fourth及分片不确定性，不更改原benchmarkreference、不进入提交。方差truth仅oracle诊断，用于检验meanlocalinnovation是否由DeltaVar解释；只有零泄露合法替代才能进生产。用户已授权Modal费用，按短片64worker保存journal失败而非900smonolith。

## 2026-10-13 接续：第二公共网络复验与方差诊断

冻结network16（mini_shard02首行、原reference），parent/r09严格RPC同输入/seed/工具串行配对，双方零失败。before raw 1.810610328334e-08、adjusted 3.958603800869e-09；after raw 1.677138428136e-08、adjusted 7.125748410506e-09；raw变化-7.3717%、adjusted变化80.0066%。parent与candidate FLOPs分别480780523639、934310859798，wall/residual分别43.920273/0.191472、102.763497/0.388126秒。证据：`results/phase2/tenfold_conditioning2node_network16_r01_paired.json`；workflow `results/phase2/tenfold_conditioning2node_network16_r01_workflow.json`，保留command/PID/hash。两网GH2均raw改善但费用使adjusted更差，不晋升、不上传、不称十倍或独立最终验收。

方差诊断Modal `covariance.causal.r01` 64×1CPU、256短片×262144、两stream运行中；第一wave1成功/63 timeout，原300秒失败全部保留，禁止放宽后伪称资源通过。这是离线truth诊断非候选资源评测；下一版拟仅缩为65536/片、交错stream，原自然完成后执行。母链canonical trace另执行一次，不读取reference选参数。

下一候选r10为一次t=0条件母链加GH8廉价diagonal-Gaussian response差值，数学上是有偏近似，不是已知均值无偏CV。固定三toy中diag近似有一个恶化，保留风险；先严格单网否决/确认，再按证据选择低成本variance机制。阶段1阈值5.779e-10不变，最佳仍V37bH18线上4.2937797168e-9。

r10 strict1同network0新结果：raw2.974875101813268e-8、adjusted6.525045206060913e-9，零失败；同cycleparent raw2.2281179568039988e-8、adjusted4.8714160493622575e-9。raw+33.515%、adjusted+33.946%，cheapdiag条件响应否决。r11仅cheapdiag→fullcovfirstchaos，GH8及t0母链冻结，严格RPC单网正在运行；不得提前报成功。

r11 fullcovcheap response严格单网完成：raw2.072104088313e-08（相对parent-7.0020%），adjusted1.001602295015e-08（105.6080%），零失败、FLOPs1062951785084、wall50.432606/residual0.226421s、clientRSS287840KiB。全协方差响应比diag正确但其成本仍抵消收益，停止该CV路线节点/参数扫描。下一项r12球面输入分布的解析径向剥离正在做实际representedK3/core合同，不以理论合同称整网提分。

r12径向sphere单链严格RPC完成：raw2.224884809721e-08（-0.1451%），adjusted5.519223882747e-09（13.2981%），FLOPs545506968169，wall/residual46.880127/0.209110s，零失败。没有实质精度收益，成本增加，否决该首层径向表示变换（不扫scale，不上传）。实际representedK3/core n8/32合同通过不等于真tensor；NO_CORR默认1且strict清V17env，本次firstmeanrider无删除造成的额外因素。

云离线高阶归因初值：13successshards（stream0=1.049m/stream1=2.359m）分4whole-neuronvector独立chunks，以固定理论dK2,dK3,dK4系数估计末层localinnovation4.419e-9→oracle残差2.658e-10，4chunkSE1.20e-10。只诊断边际累积量闭合可能误差源，不是合法候选/整链raw/adjusted或十倍验收；早层采样噪声和成功片选择必须更可信r02确认。记录`results/phase2/modal_covariance_causal_r01/highorder_partial198_r01.json`。

Modal covariance.causal.r01完整终止：256返回、13成功、243失败，completion.complete=false。失败含300s timeout及preemption/runnerSIGKILL，不能把成功子集称完整采样。r02唯一数值工作量改变262144→65536每片，总67m保持，同时offlineworker内存1→2GiB、交错stream/新seed修复早期诊断与资源稳定；这些是离线诊断资源，不是候选120s/8GB验收放宽。PID1575604，64×1CPU，journal逐条fsync，不重复启动。原source/rawseed错命名在独立审计更正：dataset mlp_seed6319981554997072999，grader setupseed3909487726。

## 固定二阶边际闭合 r15：真实负结果

从同一H18父版本独立生成r15，只补齐边际rawmoments的固定二阶Edgeworth项：均值已有κ3²/72与κ3κ4/144，故均值只补κ4²/1152；p2/p3/p4补三种二阶项。不拟合真值，不改变pair或资源。独立256点Gaussian积分核验实际候选block，maxerror1.4477e-15，合法计费2212FLOPs（7个toy neuron），不是提分。

严格同network0，before raw2.228117956804e-8 / adjusted4.871416049362e-9；after raw2.252167519146e-8 / adjusted4.924006086041e-9，分别+1.07937%/+1.07956%。FLOPs480781460599，wall45.866670s，residual.210547197s，client293448KiB，零失败。候选SHA5e6f92cb89e32a7fa888a19dc59a02e54bea17fe78e7a3a023a90acc2c40e9e3；score与samecyclecomparison为results/phase2/tenfold_marginal_edgeworth2_r15_*。否决此固定二阶marginal-only机制；不扫增益、不扩量、不上传。不能由这项负结果推出真实K2/K3/K4闭合误差无关。

最新离线双流诊断约13–15m/stream，r02至少3个基础设施失败，保留原journal。末层局部innovation4.419e-9；两stream平均理论effect后的真实positiveMSE约1.8e-10，独立流交叉消噪约8.4e-11，接近官方10亿样本参考的nominal约7.9e-11噪声。此前小样本/8chunk曾出现负残余，后续样本已回到与噪声相容；不能当负MSE或精确CI。oracle线索不是候选/整网成绩。当前完整三阶feedback与fullmetric四阶组件并行开发，原十倍阈值不变。

## 完整heavyhypertree历史 r14：真实负结果

固定新完整来源机制：关闭parentthinFB，恢复官方cost3的K21/K12/K22/双covariance与lightedge所有hypertreepaths；合并成H腿并用M=-6(A*H).T精确剔除出生重指标，4腿全宽全历史运输，无新增rank/age/row近似。独立generator n4/7/12/32合同max2.49e-14，n32/64depth4wholepredictfinite；FP32Strassen相对误差≤3.90e-7。n8depth16nonfinite原失败保留，实际1024严格验证独立进行。

network0严格RPC零失败：before raw2.228117956804e-8 / adjusted4.871416049362e-9；after raw2.228754070188e-08 / adjusted1.654884106169e-08，raw+0.02855%、adjusted+239.713%。FLOPs1632808519965，wall99.421335/residual0.286602129s，client287916KiB。候选SHA832bb894ca238fa2014d963ab1f81c3ca08bb6d27de7434bef9eec70d9e4d7ad。没有精度收益，停止该历史扩展及leg共享降费支线；不扩量、不上传。它只否决这些retainedhypertree缺项，未验证官方本身过滤的triangle/K111×edge等nonhypertree图项。r16fullmetric两层候选开始串行strict，完成前不报成绩。

## 真metric两层 r16：真实负结果

仅倒数第二层将parent近似G与实际B=W W.T配对，D4/D31/D22完整替换USE，跳过该层旧averagefeed；外部保存官方三feedCP及严格重指标扣除，rawW运输一次到final。早层与parentG生产freeze。actualn8/32depth3对象/CP/actualfinalreadout合同通过（float32≤1.17e-7），generic表nonregen未完全读D31的范围限制已注明，不扩大成全部图项parity。

同原network0严格RPC，before raw2.228117956804e-8 / adjusted4.871416049362e-9；after raw2.231846352174e-08 / adjusted5.163226797216e-09（raw+0.16733%、adjusted+5.99027%），零失败。FLOPs508729276534，wall44.609014/residual0.193475329s、client288208KiB。候选SHAa1672cc0d2dedaa65eb668213612c1b47855ec46102bb0cb35721c6df1b7cfce。否决tail2currentGmetric机制，不扫层、不扩量、无raw信号故不做final收缩降费支线。真正全历史K4 birth/metric闭合仍未实现，不可从局部负结果全盘否决。

更大双流全链离线oracle以固定K2/K3/K4系数和PhiW递推：parentfinalraw2.22819e-8；两流平均effect后的真实positive residualK2=1.275e-8，K3=1.866e-8，K4=1.609e-8，K2+3=8.328e-9，K2+3+4=2.019e-9。组合oracle约11倍不是合法候选成绩；整链早层MC噪声大，8chunkcross残余1.336e-9±1.348e-9，不能据此认证30倍。下一重点仍联合真实state生产，source非树项成本门槛和真K4历史闭合。

## 低秩残差丢失补回 r20：真实raw小信号、adjusted负结果

只将出生Rres−QBm恢复为独立CP(3(Rres−QBm).T,I,I)，2腿全宽全历史，父已有thinM保持，diagremainder也保留以取消近似QBm的伪对角。不是改rank扫参数，不依赖truth；独立n4/8/32三年龄tensor/helper合同max1.55e-15，n32depth4wholepredictfinite；只读生产审计alias/newbornrawW/olderWD及去重通过。后续parentP有confine而remainderP全宽，故称birthdiscarded恢复，不能称完整fullM。

network0原reference严格RPC零失败：before raw2.228117956804e-8 / adjusted4.871416049362e-9；after raw2.144773247892e-08 / adjusted1.042629715438e-08（raw−3.74059%、adjusted+114.0301%）。FLOPs1069002046455，wall74.028162/residual0.264957639s，client287904KiB；候选SHAc50fd5d81ca67fdfd6b5163b82c5970b7f371c185cb6cd396ba5ee5722b57de0。残差丢失有小raw信号，但不足支付费用且距tenfold远，未晋升、不上传、不扫rank/scale。

非树K111×edge r18固定outputs0–3/all14sources有信号，但r21实际matrixfreeprobe每output每probe4,227,952,640FLOPs；output0达到预定10%SE需要4503probes，扣父费用仅418probe（SE/signal.3282）。1024output单probe约4.329e12，超总预算，此具体Hutch实现否决，非所有方法下界。r18fixed4outputs与V29同hashMC方向3/4反向，保留noise/旧capture局限不据4点断言全方法无效。继续真正K4 birth/history最小producer合同。

2026-10-13T10:26:50.187209+00:00

## 双流诊断完成与 r20 状态归因 r24

云r02自然终止：1024返回、1015成功、9基础设施timeout；两stream有效样本33,226,752/33,292,288，总66,519,040，原失败保留、不重启。最终冻结报告和脚本身份见`results/phase2/modal_covariance_causal_r02/frozen_final_commit_manifest_r03.json`。固定联合K2/K3/K4的离线全链oracle正残余2.22819e-8→1.48439e-9，wholecross8.28073e-10，8chunkcross1.05231e-9±6.313e-10；仅指真值诊断，不能当候选或十倍验收。

r24对r20使用新target-free canonical capture及该call自身prediction/gain，保持原reference与成功双流moments。严格分数仍采用原strict报告，instrumentedtrace不能当资源验收。r20 trace与strict最大差2.861e-6，finalrawcapture2.14668e-8。末层localinnovation4.41939e-9→4.43811e-9；K2missingcross6.07788e-8→5.85025e-8，K3missingcross6.52633e-5→6.51589e-5，K4missingcross5.39773e-7→5.39747e-7。恢复birthdiscarded M只填补小部分状态缺口，联合oraclelocalresidual约1.294e-10未改变；停止rank/scale追扫，重点继续真正K4producer与K3非树闭合。证据`results/phase2/tenfold_completed_state_attribution_r24.json`，无candidate真值访问或经验拟合。

r22/r23 target-free V29旧capture组件（不同于H18canonical）完成：全14来源M的diag/ssep/rank16/feed逐来源年龄分解与普通functional精确重组；四输出ordinary组件478,705,248FLOPs，逐来源与独立materializedM误差<2e-12。这些M已被parent表示，不再次叠加，不称新分数。真实四阶producer n4/n8完整合同已完成，但fixedR2仍丢失约54–84%birthFrobenius，通用stream O(n4)不可据此宣称1024生产可行；继续repeat-support lift新组件。阶段1与最佳不变。

## 非高斯三阶历史激活增益 r24：严格负结果

唯一机制为旧K3来源期望ReLU导数Phi→Phi+D3(α²−1)φ/(6σ³)+g4(3α−α³)φ/(24σ⁴)，birth重复D3/D21扣除与历史gain一致，Gaussianbirth/covariancecarrier冻结；不裁gain、不拟合oracle。独立Hermiteindicator积分与CP/algebra合同通过，实际candidate接入审计另记录。

network0原reference严格RPC零失败：before raw2.2281179568040e-08/adjusted4.8714160493623e-09；after raw2.2659300213945e-08/adjusted4.9540952601586e-09，分别+1.69704%/+1.69723%。FLOPs480781426807，wall44.847881s/residual0.199365116s，client293512KiB。SHA7ff47e2a24fae92837b0229a8040318c1e2ab979e5582809f27e5eca6ed86a94。否决此fixedgainmechanism，不扫增益/κ3only、不扩量或上传，最佳与十倍阈值不变。下一项：paircost4完整grade一致闭合与真K4重复support持久history，先共享H18真实state合同/成本门槛。

## r25 / r29 整网严格对照：2026-10-13 18:44 CST

同冻结network0、原reference、输入SHA9e53442b88a0da22e8b9dafa26aa92318b38af95ab284776fa0e972b991854b4、数据commit aa99830fdc09fad15407b10e8e3459d3e18bba0a与同严格runner；工具flopscope0.12.1/whestbench0.16.1/numpy2.2.6，单线程、2**41FLOPs/120s/.4s/5s/lambda0。小合同不替代此整网评分。

| 版本 | 唯一机制 | raw before→after | adjusted before→after | FLOPs | wall/residual秒 | 失败/结论 |
|---|---|---:|---:|---:|---:|---|
| r25 sparse D4 | 用真实CP4全历史+共享完整CP3feed替代harmonicproxy；只保留D4新出生 | 2.228117956804e-8→4.325905251790e-7 | 4.871416049362e-9→3.743006846401e-7 | 1,902,713,679,991 | 55.289197/.201143 | 0；raw19.415倍、adjusted76.836倍更差，否决 |
| r29 consistent pair grade4 | mode1全部marginal/pair cost4固定图补齐、一致float64中心化后nativef32history | 2.228117956804e-8→2.217498007440e-8 | 4.871416049362e-9→5.025853859795e-9 | 498,398,171,255 | 58.744976/.318560 | 0；raw−.47663%、adjusted+3.17029%，否决 |

r25 SHA b6fe6506fe37061a9efe48326349100bae18dead466ff6d80657b8e920447438；r29 SHA73099422205a37b7ce0f41fcb599292a04c30dd0e3d00509ee706ad7e3f9034d。client RSS287712/287676KiB，manifest.completed均true。逐网证据：`results/phase2/tenfold_sparse_d4_r25_samecycle_comparison.json`、`results/phase2/tenfold_pair_grade4_consistent_r29_samecycle_comparison.json`。prediction、score、合同、失败与执行snapshot全部保留；不扩Mini8、不重复线上上传、不改十倍阈值。

r25targetfree逐层delta从li1已出现mean+2.875e-5，li15 RMS6.55e-4，不是尾层突变；genericactualledger独立PASS排除已测gain/rawW与重复feed，但未假称actual1024trace。D4only删除parent交叉出生，尤其早Gaussian K22，可能解释链偏差，仍需归因。fullGaussianK22并非仅Cov²：含rho1..3 power cumulant及pK11²的rho2..6中心化项。新exact31历史组件n4/8/16合同<2.78e-17，但完整allhistory naive总约3.77T超预算，否决直接candidate，不age/rank扫。

下一固定机制r34：保留parent harmonic cross producer/USE/feed，同时精确携带signed univariate CP4 residualhistory及一致feedback，按postmetric合同去掉已有D4再出生。它仍是crossapprox，不称fullK4；先独立tensor与actualwholecall合同，再单网strict。另推导Gaussian22 structured transport成本。最佳V37bH18/线上335678仍不变，阶段1<5.779e-10未达。

## r34 signed D4 residual + parentcross：严格否决

冻结SHA83d81b99c7bd38db184686d63ddbe5b39402c3bff0597289ab10c34aa6665799；postidentitymetric使proxyD4=g_prev，残差出生K4v−g_prev−oldEpostD4。parentcrossUSE/原近似feed完整保留，extra E completefeed独立去重。独立jointtensor、identitynorm、实际bodydiagnostic小合同与ledger通过；forcedriders小16层NaN失败、traceharness失败保留，不将snapshot当真实suite验收。

真实同network0 strict1 zero failures，before raw2.228117956804e-8/adjusted4.871416049362e-9；after raw4.779058713211e-7/adjusted4.135419140055e-7（raw21.44886倍、adjusted84.89152倍更差）。资源限制通过，但不是算法改进；逐网/FLOPs/wall/residual/hash见`results/phase2/tenfold_harmonic_d4_residual_r34_samecycle_comparison.json`。不扩Mini8、不上传。

独立joint约束推导指出signedD4补回虽然匹配D4，却改变trace；`tenfold_joint_diagonal_trace_projection_contract.json` n4/8/16同时对densefulltensor与独立2n线性方程解合同<1e-16，并检测r34broken反例。下一r37固定projection：s_eff=K22row−(traceDiag(Eoldpost)−D4(Eoldpost))，g=6s_eff/(n−2)−3sum(s_eff)/[(n−2)(n−1)]，cnew=K4v−g−D4(Eoldpost)。这是模型内精确两个约束，不oracle、不rank/layer/系数搜索；cross仍近似。后续actualbody合同/strict才能判断是否提分。

新增r36对已完成strict预测和原reference逐层只诊断，第二层r25 MSE1.99837e-9/bias2.94005e-5，而parent7.40447e-10/bias6.51e-7；最终r25 centerederrorvariance3.74358e-7，不能靠一个meanoffset解释或修复。两个新脚本loader失败及executedsource保留，最终显式split=mini读取原prepared目标，不改reference；结果`tenfold_completed_layer_scores_r36.json`。

## r37 joint D4+trace：消除错误，仍无分数改进

SHA637212f82ca4906a3b6524623c64bb0bb0eff5712ae592e25708db93fd48a469，仅替换r34出生projection为同时匹配ordinaryD4和K22rowtrace的固定affine解（扣oldEcross）；无拟合、层/rank/系数扫描。独立dense+2n方程解及actualbodyledger通过，saveddiagnostic onlyforcedriders与原source哈希分别记录。

同原network0 strict零失败：parent raw2.228117956804e-8→candidate2.273770505212e-8（+2.04893%）；adjusted4.871416049362e-9→1.968337971666e-8（+304.0587%，4.04059倍）。严格FLOPs/时间/残差/内存逐网见`results/phase2/tenfold_joint_trace_d4_r37_samecycle_comparison.json`。相比r34raw4.779058713211e-7明显恢复：sameinput targetfree第二层deltaRMS9.80e-5→5.11e-6，最终deltaRMS约6.6e-4→1.66e-5；证据`tenfold_diagonal_joint_targetfree_layerdiff_r01.json`。这解释了一种状态不一致，但没有超越parent，不能将修错恢复说成自身提分。停止CP4/D4projection邻近候选、降费、rank/layer/scale扫描，不扩量、不上传。

下一P0为真正不同的jointstate rollout校正pilot：公开独立随机He网络corpus14048，实际shard00 idx0,16,…2032；前32row固定前24train/后8validation。仅用离线独立网络高精度MC目标训练，禁止Mini8或公共比赛reference作训练，网络划分与工具/数据哈希预先冻结。先验证torchseed生成与首层解析均值/MCnoise匹配，再捕获原H18完整actualpost μ/K2/K3/K4，固定理论features/一项ridge模型；后续闭环注入必须同步Cdiag、ordinaryM出生、G和所有centering，不能在未校正状态上teacher-force。云pilot不是受测候选/最终holdout，若失败/噪声/资源门槛不过原样保留；不保证十倍可行。阶段1阈值未改，最佳仍V37bH18。

## r38/r39 全闭环已完成：否决线性联合模型

2026-10-13 19:14 CST，完整32baseline、三个paired8closed模式均零数值失败；原reference/权重配对hash不变。此为N1e8独立corpus诊断，非比赛adjusted/资源验收；首轮32SIGKILL另保留。

|版本/机制|raw before|raw after|变化|结论|
|---|---:|---:|---:|---|
|mean_and_varfloor|2.267771679777e-08|2.393295327818e-08|+5.5351%|否决，0/8失败|
|four_diag|2.267771679777e-08|2.708309470173e-08|+19.4260%|否决，0/8失败|
|exact_mean|2.267771679777e-08|2.392908622113e-08|+5.5181%|否决，0/8失败|

不扩原Mini8、不线上上传、不扫λ/scale。模型48909c1ab79be77b30699dd1b66385a4a50d048718cc31f70a7fd1f604a795d0仅训练24网；当前共同线性模型对高阶误差没有迁移收益。训练24内另固定degree2交互、λ.01、4个18train/6held网络fold，quad/linear ratio μ.98816/var.97669/K3.87129/K4.76683，后两通过预设门槛；这只CV预测性，不闭环提分。冻结新模型SHA0610032b01bb3538a5ca6bea09079c92601222e7155775d1ea4ff0b292231e3b，接下来新8网idx512..624 step16闭环。不把已用384..496验证网当最终未见holdout。最佳仍V37bH18，十倍未达。

## r41/r42 新8网完整闭环负结果

2026-10-13 19:20：r41二次交互及r42分层联合状态闭合在新8网完整完成，raw分别+8.44%/+10.61%，均零失败、均否决。train24静态CV改善没有迁移到闭环，不重复扩量/线上提交。r43正在仅24train上用固定Phi/W剥离继承均值误差，下一CV检验局部innovation预测；这是待验证假设，非已有提分。十倍目标未达、最佳仍V37bH18。

|版本|唯一机制|raw before|raw after|变化|失败|
|---|---|---:|---:|---:|---:|
|r41|22feature→253quadratic共享jointmodel，layer0zero|2.223778510747e-8|2.411489836404e-8|+8.4411%|0/8|
|r42|固定22feature ridge分层模型，layer0zero|2.223778510747e-8|2.459668120859e-8|+10.6076%|0/8|

两版相同新8网512..624/原MC N1e8 reference/权重sha配对，严格提交资源未测，诊断内存16GiB、sys.settrace/NumPy不进入提交。lawfulfnp模板及exactmodelhelperparity只组件，不称提分。r43 e_l−Phi_l*(W_l.T@e_l−1)为firstorder局部error target，忽略协方差和source响应；canonicalCDF/features/weights严格匹配，不能冒称完整真实innovation。目标训练仅24旧train网络，已用16验证网络不再称未见最终验收。

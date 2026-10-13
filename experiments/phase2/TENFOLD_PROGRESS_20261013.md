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

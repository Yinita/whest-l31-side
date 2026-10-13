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

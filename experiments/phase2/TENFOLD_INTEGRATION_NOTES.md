# 十倍目标：完整四阶核心的最小接入审计

2026-10-13。只读审计冻结 `sprint_public_v37b_h18_r01/estimator.py`、固定官方 commit `93d091a4c26c042bfffa28f2e76a81bc0aba94bb` 和已有实验摘要；未运行新重实验、未修改候选或研究状态。本文是实现建议，没有新增整网分数。

当前 adjusted 的计量下限为 `0.1 * raw_MSE`。阶段目标 `5.779e-10` 因而要求 raw `<5.779e-9`，即使降到最低计量档也比目前约 `2.2e-8` raw 至少好 3.8 倍。已确认的 V37bH18 改善主要来自计算成本，单纯继续降费不能完成本阶段。

## 当前闭合路径及代码锚点

以下行号对应冻结 V37bH18 文件。

| 位置 | 当前作用 | 完整核心接入的影响 |
|---|---|---|
| 970–972 | 仅保存 `g_prev`、`var_prev`、`lam_prev` | 当前没有持续 dense core 状态 |
| 1454–1463 | `dG=(W²)@(g_prev-lambda*var_prev)+lambda*var`，并执行 V25 自适应 | 仅为 `diag(g)+lambda*C_off` 的 transported diagonal；不能称完整 core 输运 |
| 1464–1467 | `g4row=dG*sum(W²)`，`wk4m=m(dG_i+dG_j)/6`，`wk431=m*lambda*C_off/2` | 边际项用真实 metric 对角；矩阵项仍用平均 metric `mI`，混合近似必须保留标注 |
| 1524–1528 | 四阶 slices 进入既有 nonlin term program | 此处直接加 partial K211 会重复代理内容，且不是官方 power→ordinary 转换 |
| 1668–1687 | feed 的 `X3=A*c1+P*c2`，`c1=lambda*w1`、`c2=w1²*dG`，`y=m*w2/4` | 标量 lambda 可沿用此表示；一般 dense core 无法只靠 A/P 列缩放表示 |
| 1731–1758 | 将 post-ReLU `pk11/pk21/pk22` 转为 K11/K21/K22 | CovCov 转换必须使用这里的 post-ReLU K11，而不是 pre-ReLU C_off |
| 1780–1817 | newborn 残差和低秩 M/feed 腿加入来源 | 完整 core 若新增 feed leg，必须处理 repeated 部分并与这里的 marginal 残差去重 |
| 1825–1833 | `g_prev=cA*(K4v+K22@1)+cI*trace`；base lambda 查 LAM 表 | 最小替换入口；保持对角闭合与 V25，单改 base lambda |
| 1907–2070 | `_dslices`、tier1/tier2 shared basis | 可利用既有 old-source 因子降低 K211 收缩费，不能抽样源列冒充 exact |

官方 `factor_k3.py` 330–355 固定 retained diagram 过滤与输出 Sym；395–453 列出 K4→K3 的三个 feed 图项；459–521 进行 ordinary cumulant 转换和 K111/K211 额外贡献；524–533 对四阶整体投影。`kprop_harmonic.py` 276–313 明确 linear core transport 与 metric transport/平均 metric 选择，578–586 明确每层采用平均或真实 metric。

## 可立即接入的最小候选

建议版本 `tenfold.corelambda.layer14.r01`：仅在 `li==14` 的 post-ReLU stage 用完整生产得到的非对角 core 投影出 base lambda，替换 LAM 的这一个值；其余机制冻结为 V37bH18。层14修改会在末层15读出，最小化成本并避开多层循环归因。

定义 `S211[j,k]=sum_{i distinct j,k} K211[i,j,k]`、对角置零，K31 也置零对角，则

```
Goff = (6/(n+4)) * zero_diag(K31 + K31.T + S211)
Cout = zero_diag((K11 + K11.T)/2)
lambda_base = sum(Goff*Cout) / max(sum(Cout*Cout), 1e-20)
```

全部使用合法计费算子与当前模型状态，不读 reference。符号 `S211` 指完整 centered 贡献，不能把 r33 的 partial 或已经乘过 `6/(n+4)` 的 source component 再当 S211。使用 Sym(K11) 是为了当前 float32 非对称舍入，不改变理论投影。对角不参与 LS。`g_prev`、`var_prev`、V25 的自适应 multiplier/REF_R、feed/use 逻辑全部冻结。

实际调用要分两步：在 `pk1v/pk2v/pk21/pk11` 刚得到、旧块wick缩放之前（1550–1592）计算并保存非对角core；到1731–1758得到post K11，再1825附近执行LS替换。1603–1604会原地缩放年轻A/P腿，若等1825才读取 A_st/P_st，收缩使用的便不是原pre-ReLU来源。中心化使用pk1v；既有经验修正delta是独立机制，不能改成之后的mu=pk1v+delta。

这仍是完整生产向 `lambda*Cout` 的压缩近似，且输入 K4 继承旧代理；不是完整四阶传播。不得加 oracle lambda，不基于测试真值选择层、符号或 clip。

`corrected core diagonal` 不是这一步的可用替代方案：K31 与 capped K211 在本投影中只贡献非对角，`g_prev` 的对角依旧由 K4/K22 决定。同状态下仅保存新 core 的对角会原样丢掉这两类新信息。

## 完整生产需要同时闭合的部分

1. 低支持 pK211 matrix 图项（r30）、post CovCov 抵消（r31）、出生 hypertree 的 `-2 mean*pK111`（r32），需同一 Sym/过滤/中心化约定联合计算。
2. 输入来源 K111 的 `w(1,2)` 与 `-2 mean*w(1,1)` 贡献在收缩前合权重。r21保留完整来源的B1+feedback/M/feed，但只证明这一轨道。
3. 三指标 K4 的 pK211 和出生 pK111 的 mean-feed 仍需补齐并与上述来源贡献去重；r35 capture正在收取，不能提前称完成。
4. K31 的 15 类 augmented 图项和中心化已在 r26/r28通过，输入四阶仍为 proxy。K31/capped K211 不能先任意加到 `wk431/wk4m`，它们属于 post-ReLU 输出投影，USE 的 `wk431/wk4m` 属于 pre-ReLU 输入。

2026-10-13补项助手 `tenfold_k4_component.py` 的调用返回未乘谐投影的当层K4 power+K4feed mean-centering。联合raw cap为 `lower_support_power -2CovCov(post pk11) + tree_birth_mean_center + input_source_joint_center + current_K4_component`，最后统一 `6/(n+4)`。输入历史source内已含过去层K4feed，与当层新增K4feed是不同时间对象；两者需要保留，但不能另行把历史feed再加一次。

这些项存在大抵消：r30 power 范数137.75/140.24，经CovCov剩36.29/37.12，加出生tree mean-centering剩2.85/3.71。只加一个 orbit 会改变抵消和尺度，可能比原proxy更差。完整生产得到的 `Goff` 应替换旧 `lambda*Cout`，不能在旧proxy上叠加全部贡献。

## 成本门槛与可用的精确降费

已有 r21 全14来源 costs `182,943,179,762` FLOPs；r33 matrix/CovCov/tree组合 `147,196,184,576`；r26 K31 `267,828,224`。单层合计至少 `330,407,192,562`（15.0%预算），尚未含新增 K4项和生产集成转换。约4500亿FLOPs的父候选加3300亿后，raw需降低约42%才能仅使 adjusted 持平；新增 allsource 若全float64，门槛更高。预算未超限不等于 score改善，必须报告二者。

先采用完整 source 正确性门槛；不要做年轻来源裁剪或固定列抽样来降低成本：r22仅年轻8/4/1源组件误差37.30%/75.25%/97.72%；r23固定抽512/256/128列误差49.67%/84.95%/127.97%。

对旧来源可保持全部出生列，利用已有 shared basis 精确改乘法次序。r17中的一个乘积

```
X = [b*(sum(u*a)-2*u*a)*v] @ (c*v).T
```

若右腿已表示为 `c=Qc*F`，精确改写为

```
X = v[:,None] * ([b*(sum(u*a)-2*u*a)] @ F.T) @ (v[:,None]*Qc).T
```

从一个 n³乘法变为两个 n²r乘法，出生列数和源数完全不变。tier2用 `Qc@U` 与其既有 FA2/FP2；feedback右腿Y1可用 `[Qc,F2]` 与 `[FA*w2; R2.T]`；M右腿用 `[Qc,Z]` 与 `[FP*s+3FA*e; ell.T]`；A/P右腿直接用 Qc与FA/FP。行内逐元素混合可能不再位于Qc空间，因此只把确实有因子表示的右腿拉出，不假称整个Hadamard项低秩。最终 diagonal mask可能令结果满秩。

## 进一步的 dense core 候选

若 LS 压缩丢失主要 residual，应另立版本，保存 `Gpost` 并在下一层完整计算 `Gpre=W@Gpost@W.T`。可以首先继续采用官方允许的平均 metric `mI`，单独研究 core记忆；无需同时更换成完整 `W@W.T`。这要明确保留平均metric近似，不能称真实metric完整法。

在平均metric下，`dG=diag(Gpre)`、`wk431=m*Gpre_off/2`、`wk4m=m*(dG_i+dG_j)/6`，USE对应一致。一般 `X3=diag(w1)@Gpre@diag(w1)` 不能从现有 A/P及c1/c2列缩放恢复；需要额外dense transported X3腿、经过核验的薄 residual 分解，或扩展已有 source表示。持久新增腿成本随来源年龄累积，不能只核算一次 WGWᵀ。

如另切换真实metric M，K4边际为 `Gii*Mii`，K31为 `(Gii*Mij+Gij*Mii)/2`，K22为 `(Gii*Mjj+Gjj*Mii+4Gij*Mij)/6`，distinct K_aabc为 `(Gaa*Mbc+Gbc*Maa+2Gab*Mac+2Gac*Mab)/6`。这会同时激活官方第二个 feed图项（系数1）；不能仅用改过的dG搭配旧lambdaCoff其他slice。

## 现有整网结果筛选

只检查 ablation_queue 的 completed_positive 数值及最近摘要，没有扫描巨大诊断数组。

- 已确认主要精度改进 H18 的 Mini8 raw `2.2702818514e-8→2.2180652870e-8`，已进最佳版。
- 未整合 H22 输出敏感 feedback sketch 的 Mini8 raw `2.2702818514e-8→2.2646484465e-8`（约0.248%），并非十倍路线；可在结构候选之后单独组合归因。
- H01/H02只在单网有约0.10%/0.008% raw改善；不足以解释十倍缺口，仍需mini8而不能从completed_positive标记推断泛化。
- M03 residual diagonal correction Mini8 raw `2.2737487670e-8`，比冻结基线更差；不推广其单网0.53%收益。
- 旧 Gaussian lambda proxy 全链单网 raw `5.4152209827e-8`（恶化2.36倍）；仅首层 raw `2.2967631352e-8`（控制2.2904664831e-8），也未改善。不重复旧代理或盲扫clip。
- 官方小网络完整 K3aug 在同dtype/common reference的实测误差显著好于 generic V29（约3.6倍/4.3倍），但 width64/depth4不是赛形。最有潜力的机制方向仍是把完整 augmented/core闭合的缺项接入实际1024×16；它目前没有新整网分数。

推荐顺序：完整 K4三指标/mean-feed正确性 → layer14完整core→LS标量替换整网pilot → old-source共享因子精确降费 → 若LS表示上限明显，独立版本dense core记忆。结果无论正负都保留 raw/adjusted/FLOPs/失败，避免只做组件文档而不进入整网候选。

## r02精确LS分子组件已核验

新增 `tenfold_source_scalar_projection.py` 直接计算source capped core与live covariance的内积，不形成每来源capped matrix。令 `H=diag(v) Cout diag(v)`，则一块Sym(a,b,c)的分子为

```
( sum[b*(sum(u*a)-2*u*a)*(H@c)]
 +sum[c*(sum(u*b)-2*u*b)*(H@a)]
 +sum[a*(sum(u*c)-2*u*c)*(H@b)] ) / 3
```

这里Cout必须Sym且zero-diagonal，重复坐标剔除包含在-2项中。所有B1、M和feed共享 `H@A/H@P`，余薄腿直接 H@Z/H@F1/H@F2，所有出生列保留。旧源采用 `(H@Qc)@FA/FP` 或tier2因子；Qc必须具有与现有A/P一致的行mask。li14的S_ON=0、ro=0，tail928:为零；若推广到li15的128饱和头行，不能用未masked Qc替代故意zero的formed A/P。

新 `tenfold_source_scalar_projection_contract_r02.json` 在n8/32、float32/64、dense/shared masked basis共8组，独立完整六置换tensor+distinct sum通过，float64最大绝对误差7.28e-12，float32最大相对误差小于3e-7。没有实际1024成本或分数。

另 `tenfold_source_scalar_projection_mixed_r03.json` 核验r01候选实际mixed-dtype操作：父source float32、中心化和LS权重float64。n8/32 fullcap→scalar重组相对差4.20e-8/5.86e-7；n32组件876,828→486,210 FLOPs。由于先float32构成Xt/M/X3与共享线性后运算的舍入顺序不同，不能宣称bit-identical；同公式在float64独立合同通过。后续实际1024同capture需确认误差与资源门槛。

## r03 sharedHQ 与 r04 连续全层候选

隔离候选 `estimators/tenfold_corelambda_li14_r03/estimator.py` 从冻结r02改一个主要机制：旧来源的 HA/HP 以共享 `H@Qc`、`H@QU` 再乘所有源FA/FP得到，young来源仍dense。tier1本地因子索引为 `j-kb`，tier2为 `j`。源码保留source生产float64、LS结果float32，不修改父候选/state。

QU是scratch，它在当前层只写ro:ns，尾/头可能陈旧；候选合法astype到独立float64复制，按ro:nr剔除，再HQ，原父数组不改。Qc采用相同mask以适配推广到有饱和首行的层。全部原represented A/P及出生列仍保留。原float32/Strassen形成的A/P与新HQF只在舍入意义等价，不能说逐位相同。

`tenfold_shared_basis_contract_r01.json` 共8组：n8/32、无/有首行mask、父Strassen0/1，各含2tier2+2tier1+1young来源。最大scalar numerator相对差2.0624e-6，通过固定 `absdelta <= 2e-5*max(abs(before),1)+1e-8` 门槛。n32 five-source component计费2,650,580→2,099,764（−20.78%）。这是小shape合同，无1024或RPC通过声明。

`tenfold_shared_lambda_contract_r03.json` 进一步直接调用r02/r03完整 `tenfold_fullcore`，检查参数传递、tier索引、普通图项组合与最终lambda dtype。n8/32合成状态、2tier2/2tier1/1young，两个版本返回的float32 lambda逐位相同；n32完整helper计费5,017,910→4,467,094（−10.98%）。数据为合成且无reference，不代表实际整网提分。

`estimators/tenfold_corelambda_alllayers_r04/estimator.py` 仅将r03的 `li==14` 改为 `1<=li<=14`，验证“连续完整生产反馈闭合”假设；没有根据分数选层，mode0/layer0及last保持父路径。

`tenfold_alllayers_analytic_cost_r01.json` 使用shape-only主GEMM模型：3fused lower-support、1CovCov、5tree每层；HA/HP young4来源上限；tier1rank384、tier2rank224及薄rank8/18。14层新增主GEMM约1.17785e12 FLOPs；加约4.5e11父成本约1.62785e12（预算2.19902e12）。这排除elementwise、Wick、astype、K31与其他输出/资源费，不能称完整预算或120s/.4s验收。实际模型接近预算/时间门槛，需root按原限制串行试验并保留失败，不以增预算解释达标。

## r07：修正 LS 闭合所在的坐标系

审计发现冻结 parent 的 LAM 表注释明确拟合 transported `G_pre_off = lambda*C_pre_off`，不是 post covariance。r01–r05 的 post→post LS 与此闭合对象不一致。新隔离 `tenfold_precorelambda_alllayers_r07` SHA256 `b20fd6a1044c462914acabb1c01b0ccd98c337e28583bc66585f9fbd9b1ce005` 在层1–14使用实际下一层权重做 white-box lookahead，先形成 `D=off(W*Cpost*W.T)` 与 `H=W.T*D*W`，再由 adjoint contraction 计算完整 represented post core 与 H 的内积，包括 `dot(g_prev,diag(H))`。source helper只接收 `off(H)`，保持zero-diagonal合同，最终除以 `sum(D*D)`。

权重存储是input-by-output；当前input permutation作用在其第一轴，之后转置。未来output permutation不会改变Frobenius LS。post covariance对角用float64的 `pk2v-pk1v**2`，非对角用Sym(pk11)。新lambda所在层停止旧V25 beta，该beta原来针对LAM表拟合；没有新lambda的首层仍继承parent。此版本保留后继scalar regeneration与average metric近似，未切换dense G记忆。

独立 `tenfold_prels_transport_contract_r01.json` 用六置换三阶tensor、distinct cap、任意对称残差及对角，直接比较dense `W*Gpost*W.T` 与adjoint/source scalar，n8/n32绝对误差为1.35525e-20/1.11173e-21，权重方向与两种permutation均通过。无reference、无整网score或资源通过声明。Lookahead额外4个dense float64 GEMM每层，shape-only估计约2.40518e11 FLOPs，叠加r04实测1.69246e12约1.93298e12；实际elementwise/astype及资源必须严格评测，不能提前称通过。

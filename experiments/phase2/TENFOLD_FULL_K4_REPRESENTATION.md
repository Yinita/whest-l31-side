# 多 metric 四阶表示：组件合同、预算与实施路径

2026-10-13。本轮只添加独立组件、合同和设计，没有改候选、shared state/queue，没有读取 benchmark reference，没有报告新的整网 raw/adjusted 分数。与 alternative 分工：本文件处理 K4 多矩阵表示及完整 USE/feed；K3 非 hypertree 新来源由 alternative 另行设计。

## 现在具体推进了什么

`tenfold_full_k4_component_contract.py` 合法计费 helper 已实现任意 symmetric dense A/B 的 D4、D31、D22、weighted distinct D211 三项、官方全部三个 K4→K3 feed groups、birth D3/D21、feed 对 K211 的 mean centering、真实 W/WD 运输。n4/n8、dense/diagonal metric 独立 dense tensor 合同全部通过，float64 最大绝对误差 `4.27e-14`。证据为 `results/phase2/tenfold_full_k4_component_contract_r02.json`；脚本固定快照 `tenfold_full_k4_component_contract_r02_snapshot.py`。r01 是前一组件版本的证据，保留不覆盖。

`tenfold_full_k4_real_component.py` 在冻结网络0、layer14、width1024 内部 proxy capture 与同网络公开 weights 上检查同样算子，12 对坐标独立逐项 scalar sum 的 raw cap 误差 `2.78e-17`、feed centering `2.08e-17`。完整组件计费 `17,488,199,672` FLOPs。它还实际测量固定 seed、rank32、精确 diagonal 校正的 metric QR 压缩误差，不能只凭 D4 保真推断全部 feed 保真。

| width1024 metric 压缩指标 | 实测 |
|---|---:|
| metric relative Frobenius error | 69.14% |
| metric diagonal max absolute error | 4.44e-16 |
| D4 relative error | 1.64e-17 |
| D31 relative error | 15.24% |
| D22 relative error | 1.17% |
| raw K4 power D211 cap relative error | 11.43% |
| birth feed D3 relative error | 1.77e-17 |
| birth feed D21 relative error | 4.82% |
| feed mean-centering D211 cap relative error | 3.36% |

压缩采用 `Q=qr(B Ω)`、`Bhat=Q(Q.T B Q)Q.T+diag(B−low)`，Ω 固定 seed261013952，rank32；没有 score/truth 选择。实测构造 metric+Wicks+symmetrized proxy G `4,303,833,090` FLOPs，压缩 `419,148,460` FLOPs；压缩后的 metric 被 materialize 成 dense 以做同算子比较，组件 FLOPs 仍为 `17,488,199,672`，**没有测得低秩运行加速**。报告 `results/phase2/tenfold_full_k4_real_component_r01.json` 保存 capture/weights/script/helper hash、数据 commit `aa99830fdc09fad15407b10e8e3459d3e18bba0a` 及工具版本。

真实组件的 A 是旧 capture 中 dG/offG proxy 的重建及对称化，B 是当层 `W14.T W14`；不是整条历史精确运输的 K4。先前捕获的 asymmetry 约7.65e-7，通过 `(A+A.T)/2` 按 symmetric tensor 定义处理；未通过检查不允许放宽容差。该结果只有组件意义，不能写成 candidate 分数提升。

## 单 G 与平均 metric 缺失的内容

当前 `K4≈Sym(G⊗mI)` 将多种四阶来源合成一个 core，再将实际 metric 重置成平均值。即使 core 保留完整 dense G，其运输应同时作用于 G 和 metric。真实权重导致的 metric offdiagonal、不同出生时间的 gate/weight history 与不同 signed 四阶贡献不能由单个 g/λ 表确定。

径向 sphere 输入法的代数转换是精确的，但其后复用的 nonlinear cumulant closure、K4 proxy、Gaussian 学得的 LAM/REF_R 与源压缩仍近似；r12 strict1 negative 表明输入 homogeneity 转换不能自动闭合所有后层误差。不以 radial 合同替代整网十倍证据。此前 dense-G/early-LS/r07 negative 也没有验证下面完整 actual metric +全部 USE/feed/history 一致路线，不机械重跑同版本。

[官方 harmonic.py](https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/harmonic.py) 的 `contract_W` 明确默认同时 `G→W G W.T`、`metric→W metric W.T`；`set_metric` 的平均替换是有条件的另一选择。[官方 factor_k3.py](https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py#L410-L454) 的 dense metric feed 有全部三个组，bridge 第二组不能沿用 diagonal metric 路径省掉。

## 固定 R 个 matrix pairs 的明确表示

```
H = sum_(r=1..R) alpha_r * S(A_r,B_r)
S(A,B)_(i,j,k,l) = [Aij Bkl + Aik Bjl + Ail Bjk
                    +Bij Akl +Bik Ajl +Bil Ajk]/6
```

A/B symmetric、可以 signed，S 是24置换平均，等于上述六项。交换 A/B 不改变 tensor。每项存两个 n×n 矩阵，float32 每pair约8MiB（n1024）；R2为16MiB，不包括 scratch、feed history、parent sources。

固定 small R **不表示任意 true K4**。通用 fully symmetric 四阶张量有 O(n4) 自由度，而此表示只有 O(R n2)。Gaussian covariance 的 nonlinear 四阶星形/树形 birth 也通常不能精确缩为 small R；必须引入 CP4/图项 producer，再明确投影损失。这里的“full”仅指各 pair 表示的四阶 tensor 的全部 index support 和所需切片，不能称完整 true cumulant algorithm。

### 精确 slices

下式每pair乘alpha后累加。未mask 的矩阵条目便于合同，production 按 index partition 去掉重复坐标：

```
D4_i = Aii Bii
D31_ij = (Aii Bij + Bii Aij)/2
D22_ij = (Aii Bjj + Bii Ajj + 4 Aij Bij)/6
D211_ijk = (Aii Bjk + Bii Ajk + 2Aij Bik + 2Aik Bij)/6
D1111_ijkl = S(A,B)_ijkl
```

D211 不构建 n3 array。令 F(z)_(j,k)=sum_(i distinct j,k) z_i H_iijk，P(z)_(j,k)=sum_(i distinct j,k) z_i H_ijjk：

```
F(z) = [<z,diagA>B + <z,diagB>A
        +2(A diag(z) B + B diag(z) A)]/6
       −z_j D31_jk−z_k D31_kj
P(z) = [diagA_j (Bz)_k + diagB_j (Az)_k
        +2Bjk (Az)_j +2Ajk (Bz)_j]/6
       −z_j D31_jk−z_k D22_jk
```

输出 j=k 清零。对 symmetric A/B，F 里的两个 GEMM 互为转置，可一乘积加其转置；现组件用双乘积以先核验清晰表达。

完整 retained single-K4 power K211 cap 为

```
1/2 [F(w22)*w1_j*w1_k
      +P(w12)*w2_j*w1_k + P(w12).T*w1_j*w2_k]
```

Wick 约定 `w1=w(1,1)`、`w2=w(2,1)`、`w12=w(1,2)`、`w22=w(2,2)`；不能将 w12 与 w2 混淆。K31/K22 的其他图项仍需同一冻结 retained diagram table eval；此 helper 给的是其中 K4 input tensor 完整 slices，不能删除 K2/K3/covariance-product 图项。

### actual 运输及 nonlinear 边界

旧 inherited pair 从 post层运输为 `A'=WD A WD.T`、`B'=WD B WD.T`，WD=`W diag(w1_prev)`；刚完成 nonlinear birth 的 source 下次先用 raw W，避免重复 gain。每个 H 的 leading nonlinear inherited part 用同一 diagonal gain 左右缩放两矩阵，等价于 tensor 四腿乘 w1。新的 nonlinear birth 必须用同样 pK→K 中心化定义产生，不能直接把 post core proxy 当 birth delta。

输入 K4=0 时首 Gaussian ReLU 后的 K4 不为0；新方案需 nonlinear production。可先保留父版出生机制，改变 metric 的 USE/feed 一致性做最小受控候选，再将更完整 birth producer 作为独立下一项。多pair递归方案应以 `birth = producer − inherited_full_tensor` 定义 remainder，只投影 remainder；若对整个 producer重新投影又另加 history，会重复 inherited部分。

## K4→K3 完整 feed 与去重

每pair产生三个 CP组，`Diag(w)` 的 row scaling 记作 Dw，I为identity，1为全ones向量：

```
g1 = CP((w2*diagA) 1.T, Dw1 B, Dw1 I/4)
g2 = CP(Dw1 A, Dw2 I, Dw1 B)
g3 = CP(Dw1 A, Dw1 I, (w2*diagB) 1.T/4)
```

CP采用3腿同column相乘后 average-Sym3，每pair乘alpha，三组都保留。它们等价于

```
feed_ijk = 1/2 [w2_i w1_j w1_k H_iijk
                +w1_i w2_j w1_k H_ijjk
                +w1_i w1_j w2_k H_ijkk]
```

因而 birth slices 可直接低成本算：

```
feedD3_i = 1.5*w2_i*w1_i²*D4_i
feedD21_ij = w2_i*w1_i*w1_j*D31_ij
           +0.5*w1_i²*w2_j*D22_ij
```

当层 K211 中 mean-centering 还须完整 `−2 sum distinct i mu_i feed_ijk`；可复用 F(mu*w2) 和 P(mu*w1)，无需当层建立三组denseCP再收缩。small合同直接同官方3groups比对，已验证这两个便宜公式。

**当层便宜 contracted readout不代替后层 feed history。** g1/g3可写 matrix×vector特殊 tensor，g2是 dense桥接 hub。每出生一次都要持久携带，下一层全部三腿同 map 运输；做D3/D21的newborn residual时先减它们的 repeated，再把 residual add回。pK111→K211 mean centering也应使用同完整 source；历史feed不再次当本层新feed重复添加。原parent singlefeed的2thin/hub编码不能直接装进g2。

若 B=diag+Q H Q.T，bridge g2 并不会自动从 rank n 降为 q：CP仍按中心神经元 n列，直接展开 Q可形成 nq。需要专用 hub 的 shared-basis运输或合法 genericCP压缩；没有这一步，宣称只有两额外GEMM/层是遗漏feed历史成本。

## 非 oracle 压缩与 exact-diagonal 的界限

优先在 matrix空间保留 fixed R histories，按贡献的模型内部 Frobenius norm / 当前 USE/feed operator norm排序，没有reference输入；不能按真实误差或validation每网挑R/层。loss必需记录同状态、压缩前后张量或随机 contractions。以下是可核验的全tensor inner product：

```
<S(A,B),S(C,D)> = [<A,C><B,D> + <A,D><B,C>
                    +2tr(A C B D)+2tr(A D B C)]/6
L_B S(C,D) = [<B,D>C+<B,C>D+2(C B D+D B C)]/6
```

令 production H不是dense张量，`L_B H` 可按它的CP4/graph producer合法收缩得到。给定当前固定 B_r，可用normal equations迭代拟合A_r：`sum_r L_(B_s) S(A_r,B_r) = L_(B_s) H`；fixed counts/seed、regularization仅按模型矩阵条件数确定。每步dense矩阵乘法真实收费，不免费离线求factor。Gram矩阵求解可以只对已有pairs的scalarweights做廉价R×R系数压缩，但只保留R已有pairs时无法增加新metric方向。

一种便宜metric投影是rankq QR加exactdiag，实测rank32有上述15% D31/11% D211损失。不能从D4exact称full一致；首先保留full B的component为判别器，比较operator残差。它与fullmetric candidate是不同候选，不同时改core/birth等机制。参考误差不能用于投影目标。

## 算子与完整历史成本

float32 naive GEMM成本近似2n3；实际flopscope symmetry/Strassen按允许算子计。n1024一普通GEMM约2.147e9 FLOPs。

| 机制 | 每pair每层成本（主项） | 必须保留的额外工作 |
|---|---|---|
| full A与B实际运输 | 4 denseGEMM，约8.59e9 | gains、symmetrize、与重排一致 |
| D4/D31/D22 USE | O(n2) | 其他K2/K3图项沿用同冻结table |
| rawD211 K4cap | 现helper2GEMM+2GEMV，可1GEMM+transpose | distinct支持排除 |
| currentfeed mean center | 同上2GEMM+2GEMV，可1GEMM+transpose | 同完整feed，不能用旧averageproxy |
| generic3CP birthdslices | 9GEMM，O(n3) | 可用已验证slice公式降至O(n2) |
| each历史feed transport | naive最多9GEMM/每出生pair每后层 | 特殊g1/g3可matrix×vector压缩，g2generichub需实际共基方案 |
| matrixpair压缩 | 依据固定迭代数，多次GEMM | 投影误差报告/原始证据 |

R2×15层运输 naive约257.7e9 FLOPs，R2×15层现组件4GEMMcap约257.7e9。可是若每层新增2pair的全部3CP feeds、不压缩历史，15×14/2=105对出生→后层关系×R2×9GEMM约4.06e12，已超2**41，尚未加parent预算。**单两矩阵运输可行不等于完整算法可行。** g1/g3专用通道与g2年龄共享压缩是必要实现环节；也可先只在最后非线性使用真实metric+完整当层readout，无后层feed运输，但只能称局部归因pilot。

## 最小 production 行动顺序

| 待验证版本 | before | after | 作用与验收 |
|---|---|---|---|
| K4-M1 component（已做） | identity平均metric | actual dense metric全部USE/currentfeed，独立dense合同 | 组件通过；没有score |
| K4-M2 last-layer pilot | 父final g4row与proxy | frozenparent G，对final mean用真实metricD4完整替换 | 无后层feed，仅局部必要信号；不能当全metric算法或重复H19/H20解释 |
| K4-M3 two-step state contract | 一层parentbirth及下一层 | 完整g1/g2/g3CP、D3/D21去重、currentcenter、真正WD | 在n8/32捕获actualparent state，与独立tensor逐项比；不能只用standalonerandommatrix |
| K4-M4 full-metric integration | single平均metric proxy | 每层一个新的frozenparentG/actualmetric pair；全部USE和feed历史共同保存/共基 | 先合法1024资源组件测出全部history成本，再strict1；一次只改变metric一致性 |
| K4-M5 R2 multi-history | 仅当层或单history | 旧pair leadinggain运输+完整birthremainder production，fixed2压缩 | 明确producer retained图项与projectionerror；不得把parentG直接add为newbirth |
| K4-M6 q32 metric compression | M4fullmetric | 固定QR+diagactualmetric，完整feed同一compressedB | 当前compression残差大，应先资源/内部operator门槛；不靠truth挑seed/q |

root若选择M4，具体接入点是pre-Wick b1/b2的 `g4row=ΣD4`、`wk431=zero_diag(ΣD31)`、`wk4m=zero_diag(ΣD22)`，同current A/B计算K211 raw与feed center；newborn用上文feedD3/21减去实际feedrepeated，再保存三组generic/specialchannels。generation/metric/beta的norm约定必须统一，禁止保留旧averagefeed同时add新feed。若旧G仍只diag(dG)+λC，明确freeze这一原近似；后续history才另改birth。

必要合同还包括：A/B互换不变、signed与nonidentitymetric、permutation equivariance、special newborn rawW与oldWD一次gain、所有indexcollision masks、sumR linearity、全tensor/contract投影残差、actualparentpool alias/age/covcarrier、pruning/confine一致、seed/线程/算子/真实资源。组件通过之后才做samecycle strict1；足够raw信号才扩大Mini8和独立成对确认。十倍阶段不因组件进展改变阈值。

## 已实现最小候选 R16

路径 `estimators/tenfold_fullmetric_tail2_r16/estimator.py`，冻结 SHA256 `a1672cc0d2dedaa65eb668213612c1b47855ec46102bb0cb35721c6df1b7cfce`。这采用倒数第二层完整 metric USE、去掉母版该层 average feed，并保存新完整 feed 的 **all-distinct** tensor到final一次；其出生 repeated通过第四组`−CP(M,I,I)`在压缩前精确去掉，M=`diag(feedD3)+3feedD21.T`。母版ordinary M residual独立提供相同nonlinear K3 marginal/pair出生，不把externalfeed repeated经rank16减去后再加回，避免双计残差误差。

final-only metric gate无效果：H18已有`g4row=dG*sum(W²)`，actualmetricD4同式。因此R16用了有意义的penultimate→final两步；它不是all-layer多historyfullK4实现，未改parent K4 core/birth。该最小机制无需后层D211/fullG生产，相关完整组件合同继续为未来路线提供证据。

`tenfold_fullmetric_tail2_production_contract.py` 捕获实际候选n8/32、depth3的G/B、四CP组与actualfinal externalD3；独立denseH4/CP3及rawW运输合同通过。报告 `results/phase2/tenfold_fullmetric_tail2_production_contract_r02.json`：USE最大`1.17e-7`，birthrepeated≤`6.51e-8`，actualtransport≤`5.30e-8`（float32）。r01最初只在离线重新收缩groups，r02改为比较actualfinaladdition，两个原始报告均保留。1024strict/RPC/resource/score由root串行管理，尚无通过证据。

smallshape走non-regeneration fallback `G=diag(g4row/METRIC_C)`，用于实际代码algebra合同，不声称重构genericparent原独立D22proxy；1024 suite走真正`diag(dG)+λCoff`，不除metric。G显式对称化去掉父float32 transport微反对称浮点项，实际B由同一reordered全W计算、externalfeed不受parentblockpruning，SOURCE均记录。

generic nonregen表未必在b2读取wk431；small合同核验捕获的D31对象和source，并非全部retaineddiagrams输出一致性。实际1024regen路径将wk431加入b2，读取fullD31对象。不能扩大small合同scope为真实suite完整nonlinear输出parity。

### R16 有 raw 信号后才执行的精确 final 收缩优化

当前R16直接运输四组CP的十二腿。final仅需D3，可代数融合为3个GEMM而不改tensor：

```
V = W diag(w1)
XA = V A; XB = V B; XM = W M
extraD3 = (W*(w2*diagA)) .* rowsum(V .* XB)/4
         + rowsum(XA .* (W diag(w2)) .* XB)
         + (W*(w2*diagB)) .* rowsum(V .* XA)/4
         − rowsum(XM .* (W.*W))
```

其中`W*(vector)`在第1/3行表示GEMV，`.*`为逐元素。XA/XB各是普通matrixmultiply，XM为repeatedcorrection；其signed M不能擅自symmetric-tag。组件合同r03已对实际候选capturedfeed核验此式，n8/n32 float32 denseoracle最大误差分别`3.98e-8`/`3.18e-8`，允许3GEMM+2GEMV收费均完整保留；精确数值见`results/phase2/tenfold_fullmetric_tail2_production_contract_r03.json`。这仅是离线合法组件，不改candidate。float32operationorder会改变舍入，不将少收费当精度贡献。R16 strict raw无收益，故停止此机制的降费实施，不将这条融合组件当新candidate。

多层若每出生都保留genericfeed，finalreadout融合不能消除中间D21及所有历史成本。g1/g3的matrix×vector张量可用r12式专用state，g2桥接需generichub/commonbasis；固定R metric histories与fixedK3feedhistorycompression必须同时记录各自模型内部operatorresidual，不能据当前D3零birth宣称以后内容为零。

### R16 严格负结果

root串行strict1报告：零失败；raw `2.2281179568e-8 → 2.2318463522e-8`（+0.16733%），adjusted `4.8714160494e-9 → 5.1632267972e-9`（+5.9903%）；FLOPs `508729276534`，predict wall `44.6090s`，residual `0.193475329s`。该两层当前parentG/actualmetric完整USE与feed没有解释十倍误差，停止tail层扫描、停止融合降费候选。完整tensorhistoryproducer/projection尚未验证，不能用r16负结果否决它；也不能跳过production可行性，把该未实现路线说成预期收益。

## 接下来真正不同的可执行组件

必须改变的是K4的producer或history，不能继续保留相同parentG而只扫描哪一尾层替metric。建议下一组件从**目标无关smallshape局部非线性算子**开始，先决定已有K2/K3闭合是否支持准确的K4出生：

1. 固定n4/8、mean/C、独立Gaussian解析ReLU scalar moments，生成官方budget3/4的完整四阶retained diagram table，包括(1111)；为每个四阶项定义matrix-free`apply_H(B)`，不是先生成denseH后在predict免费投影。denseH只作为offline合同oracle；smallshape项目可以独立多维quadrature/随机有限分布做moment验证，但不拿这些truth拟合production。
2. 验证producer对输入H4=ΣS(A,B)的线性part，K3 terms及纯K2 nonlinear birth分开记录；同一pK→K中心化与CovCov抵消，先对arbitrary signed symmetric inputs做显式scalargraph枚举，不能只验证leadingWickH4 inheritedpart。
3. 从producer减去`ΣS(Dw1 A Dw1,Dw1 B Dw1)`定义newbirthremainder。用固定R2的birthbasis（例如当前C与I的已知矩阵方向）或固定seedQ basis求normal equations，记录`<H,H>`及随机polarization contractions残差；若生产`apply_H(B)`需要O(n4)，先在n32测cost并判定此producer不可扩至1024，不能先整网盲跑。
4. 先将完整K4历史transport→所有USE slices→全3groupfeed组成**两次非线性局部productioncontract**，只需n8/32，不依赖冻结benchmarkreference；用上文组件验证 feedbirthrepeateddedup。目标是实际新增source与tensorhistory都parity，避免“fullmetric组件正确但真实producer还是父proxy”。
5. 合法1024模型内部capture只计费`apply_H(B)`和一次R2projection/压缩，而不执行整候选。budget预测必须包含所有历史feed；若tensorprojection大残差或历史state显著超budget，保存负结果并转向alternative的完整K3nonhypertree闭合，不扫lambda/scale。

这个组件的主要变化是**保留真实retainedK4production/history**，它与“用任意thin采样拟合K4 proxy”不同。官方K3augment会投影掉四阶tracefree部分，固定R2能保留一部分多metric历史；仍不能保证trueK4精度。若fullproducer不具可行factor结构，可先精确保留`D4/D31/D22`及它们诱导的liftedDSTensor传播，单列另一机制；这需要各partition支持完整lift和重复去重，不能将D31/D22array当四阶matrixpair本身。

## R04 完整 retained producer：实际组件与成本门槛

`tenfold_true_k4_producer.py` 实际生成官方 k4-basic 的全部 retained 图项（k_max4,d_max4,augment=False），含 29 个1111原图/4类，全部通过官方 factoredfilter。各 pK→ordinary polynomial 来自集合分割展开，并通过任意19样本分布独立验证（不是以分数拟合）。独立 scalar rawmoment cumulant oracle 与 charged mean/C/K3及K4 apply逐项相符。完整实际 producer 不创建 charged n4 tensor，却使用 **n3 slab** temporary，O(n4)work；先前注释的n2temporary是旧row路径描述，不能沿用。冻结R04 source快照保留。

| report | 宽度 | 两步合法 FLOPs | 总时长 | R1→R2出生 Frobenius residual（两步） |
|---|---:|---:|---:|---|
| `tenfold_true_k4_contract_r03b_w4.json` | 4 | 7.21M / 13.27M | 66.09s | 70.03%→66.92% / 70.11%→53.83% |
| `tenfold_true_k4_contract_r04_w8.json` | 8 | 77.90M / 125.09M | 86.88s | 83.95%→83.91% / 85.36%→78.53% |
| `tenfold_true_k4_contract_r04_w16.json` | 16 | 1.008B / 1.476B | 176.89s | 91.09%→91.08% / 92.46%→89.17% |

n8 streaming/source parity≤1.78e-15，fullhistory pairtransport≤6.27e-15；完整 retained birth 明显无法被固定identity+Cov²两个方向表示。n16成本门槛与n4work阶已经否决直接1024 streamingprojection；不扫R/rank或radial系数。

## 新机制：exact repeat-support lift

`tenfold_true_k4_support_lift.py` 从真实retained producer减掉actualleading历史后，保留D4/D31/D22。4、31、22的支持 masks 互斥，出生211/1111为零。所有source entries、实际W运输后的D4/D31/D22、weightedD211 F/P cap以及完整三项feed D3/D21/centering与独立dense合同n4/n8误差≤1.11e-15。

| 目标无关 toy width | R2出生残差 | exact重复支持残差 | R2+exact重复 residual | source合法 FLOPs | transport+allUSE/feed FLOPs |
|---|---:|---:|---:|---:|---:|
| 4 | 65.14% | 0.3133% | 13.58% | 118494 | 19312 |
| 8 | 82.54% | 0.2285% | 10.30% | 508434 | 153312 |

这是near-independent固定toy的模型内部强信号，不是raw/adjusted分数，也不是1024泛化。R2+支持残差较高，因为R2在all-distinct支持引入不真实内容。任意source B收缩 `W apply_H(W.T B W) W.T` 以及D4/D31/Pcaps均O(n3)，但**所有transportedD22对任意densebirthD22通常O(n4)**：22项含`2∑ij D22ij Wa_i Wb_i Wa_j Wb_j`，不能称只需两个GEMM。完整supportlift还无可提交候选。

## Actual H18 内部切片门槛

共享一次H18li14 capture：`results/phase2/h18_joint_li14_r25/completion.json`。`tenfold_true_k4_h18_gate_r01_w32.json` 固定内部leading32neuron、保留14source全部1024columns；不是32随机网络或完整1024。sourceK3与parent已prune的D3/D21 max差3.95e-9/1.25e-6，需记录source含被parentreadout禁用内容。

真实模型内部retained birth norm：D4=2.146e-6，D31=3.211e-6，D22=2.094e-7。D22 targetfree offline singular residual rank1/2/4/8/16 =58.47%/30.30%/16.11%/6.26%/1.26%；简单Cov²形状最佳scalar投影残差92.31%。该谱只诊断结构，未实施或按benchmark选rank。完整retainedproducer输入K4仍是当层实际parentG surrogate，不能宣称已有truefullK4history。

## 实际提交候选 R25：独立 pure D4 history

`estimators/tenfold_sparse_d4_r25/estimator.py` SHA256 `b6fe6506fe37061a9efe48326349100bae18dead466ff6d80657b8e920447438`。替换全部层parentharmonicK4USE与approxfeed，只保存普通univariate出生 `newc=K4v−Σ c (w1*qpre)^4`，完整先前history为CP4共享leg。全新31/22/211/1111出生被该机制明确遗漏，因而与parent保存crossborntrace不同。

K4→K3 fullfeed为`Sym(X,q,q)`，Xnew=`1.5*c*w2*qpre²`、qpost=`w1*qpre`、Xpost=`w1*Xpre+Xnew`，每个源共享一X，下一层rawW一次运输q/X。新完整feed repeated加入parentD3new/21new并从普通M residual扣掉，已有externalfeed加到preD3/21；不保留oldregenfeed。birthD4不叠加parentG或完整oldunivariate，避免双计。external histories保留完整rows，范围与parentprunedlegs不同。

`tenfold_sparse_d4_history_contract_r01.json` signed n4/8/16两步组件≤2.22e-16；`tenfold_sparse_d4_production_contract_r25_r03.json` actualcandidate n8/32depth4 USE≤6.94e-7、完整feed和dslice≤6.90e-8、birthD4≤5.88e-8。r01首次实际candidate运行segfault无diagnostic，保留log；r02/r03完成，不将重跑宣称完整稳定认证。最终省去unusedfinalD21；birthfeed slices精确利用已有D4/31/22，省420n³，不改张量。分析新增成本1320n³≈1.417e12，加changedparent约0.48e12；whole1024strict由root串行验证，当前无新score或资源通过声明。完整非linear闭合仍为parenttables，不是所有K211图的production。

### R25 整网严格负结果，停止扩量

root同cycle strict1零失败，合法资源通过：FLOPs1,902,713,679,991，predict55.289s，residual0.20114s。但raw `2.2281179568e-8 → 4.32590525179e-7`（19.415倍更差），adjusted `4.8714160494e-9 → 3.7430068464e-7`（76.836倍更差）。因此它不是最佳版，不扩Mini8、不上传；sharedCP4/feed组织资源可行并不意味着纯D4出生足够。

无truth逐层比较旧同input strict预测 `tenfold_sparse_d4_r25_targetfree_layerdiff_r01.json`：li0完全一致；li1RMS3.58e-5/mean+2.875e-5，li2RMS9.99e-5，li5RMS2.60e-4，li10RMS4.75e-4，li15RMS6.55e-4/mean+2.374e-4，呈逐层积累而非尾层突然失效。独立actualledgerpassed已排除已测oldgain/rawW和重复feed问题；crossbirth删掉是待归因因素，不能凭delta认定唯一原因。尤其li0GaussianReLU已有K22 birth，母版将其投影入harmonictrace，pureD4删除它后li1即改变。晚层D31大不能证明只补D31即可恢复早层；下一完整来源应先解析/保留GaussianleadingK22结构，避免再盲纯subset替换。

## Gaussian cross22 门槛与 R34 hybrid（已实现待strict）

`tenfold_gaussian_cross22_gate_r02.json` 在actualnetwork0第一层Gaussianinput、1024fullwidth，原权重canonical顺序计算完整 budget3 ordinaryD22，不读reference。解析式

```
a_di=w(d,2)_i−2postmean_i*w(d,1)_i
pK11_ij=Σd=1..3 Cij^d/d! w(d,1)_i w(d,1)_j
D22off_ij=Σd=1..3 Cij^d/d! a_di a_dj−2pK11_ij²
```

它不仅是简单Cov²项：含ρ¹/ρ²/ρ³以及pK11平方带ρ²..6。独立scalarrawmoment四阶n8≤3.82e-16。D4norm35.96、D22offnorm20.43、D22rowsumnorm23.49；D4mean1.1195、D22rowmean.40627。repeat-trace中22row占norm44.56%，radialGdiagwith22norm.18393/without22.10635。该证据仅归因内部cross content重要，不证明r25失败唯一原因。

R34 `estimators/tenfold_harmonic_d4_residual_r34/estimator.py` SHA `83d81b99c7bd38db184686d63ddbe5b39402c3bff0597289ab10c34aa6665799` 保留全部parentharmoniccross USE/feed，在其上加signedCP4univariate residual及完整共享CP3反馈。**postproxy是S(G,I)，D4=g_prev；不是METRIC_C*g_prev**。每birth `newc=K4v−g_prev−EoldpostD4` 去重；只有pretransport metric平均才是2。所有feed重叠由parentM residual账本扣除，oldgain/rawW一次。

production只原suite-shape riders分支激活，generic原fallback未改。独立joint和identitynorm合同通过；实际bodyr05 n32depth4诊断copy只literalforce ridersTrue，反向replacementassert主体不变，报告original/snapshothash；残差USE≤7.71e-8，fullfeed/marginal≤1.08e-8，combinedpostD4≤5.83e-8。r01旧genericdraftwk431undefined、r02/03forcedsmall16suiteNaN、r04合同wrongtrace trigger失败原log均保留。不将诊断copy当true1024资源；root管理whole strict。候选冻结不在运行时改。R34不是oraclecorrector，不保留任意newcrosssupport真值；hypothesis是恢复parentcross同时精确保留D4被radialprojection丢失的分量。

### R34 严格负结果；R37 同时保持 D4 与 trace

R34 strict1资源通过/零失败，FLOPs1,902,860,669,047、predict55.1714s、residual0.200424s。但raw4.77905871321127e-7，为parent21.44886倍；adjusted4.135419140055134e-7，为parent84.89152倍（comparison字段relative值存增量83.89152，绝对ratio需+1）。停止扩量/上传。

固定oldharmonicG再补diagresidual保D4，却同时改变trace：原harmonicprojection已把D4部分投影进crosssupport。R37只改变这个代数split，必须把Eoldpost的crossrow从producedK22row扣掉：

```
q2=qpost²
oldCross=q2@(c*sum(q2,axis=0))−sum(c*q2²,axis=1)
s=ordinaryK22@ones−oldCross
g_prev=6/(n−2)*s−3/((n−2)(n−1))*sum(s)
newc=ordinaryK4v−g_prev−D4(Eoldpost)
```

新G是joint affine约束解，不再oldorthogonalharmonicprojection，但postmetric仍I。D22crossrow=`((n−2)diagG+trG)/6`；combinedpostH同时匹配普通D4和K22rowtrace，不能称完整K22/211/1111tensor匹配。parentlambda/REF_R仍冻结配置，adaptiveformula会响应新G，不把自适应记录当新fit。

冻结candidate `estimators/tenfold_joint_trace_d4_r37/estimator.py` SHA637212f82ca4906a3b6524623c64bb0bb0eff5712ae592e25708db93fd48a469。`tenfold_joint_trace_d4_production_contract_r37_r01.json`仅ridersTrue诊断copy、原candidatebody反向assert一致，n32depth4 USE≤7.51e-8，完整feed≤1.85e-8，postD4≤3.07e-8，crossrowtrace≤2.05e-7。root管理整网strict归因；未接受十倍stage，不扩量预期收益。若失败不能继续将相同表示缩放或盲扫rank/layers。

### R37 归因成功、精度机制否决

R37 strict1零失败/资源通过：FLOPs1,903,631,419,511，raw2.2737705052122692e-8（parent+2.04893%），adjusted1.9683379716659492e-8（parent+304.059%、ratio4.04059）。这从R34raw4.779e-7恢复到parent附近，却没有新precision收益；不晋升、扩量、上传、costopt或缩放/rank扫描。

无truth sameinput层差 `tenfold_diagonal_joint_targetfree_layerdiff_r01.json`：li1R25RMS3.58e-5/mean+2.87e-5，R34RMS9.80e-5/mean−8.00e-5，R37RMS5.11e-6/mean−1.24e-7；finalR37RMS1.66e-5/mean+3.30e-6，对比R25/R34约6.6e-4。这验证双约束确实消除了trace不一致造成的大累计偏差，但精确D4残差对分数无改善。**CP4 D4 projection这条分支停止**；其组件、合法全history预算与独立账本证据保存，后续仅科学上不同的完整distinct coupled-state来源可重新开启，不能从这里再盲扫附近版本。

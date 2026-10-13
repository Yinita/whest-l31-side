# 条件 GH2 提前合链设计

给定 r09 两个条件分支，提前合并可以减少后续成对计算，但必须合并分布的累积量和实际来源对象。仅均值平均会丢掉已测精度收益的潜在来源。本文不改候选，不读 reference，不报告新整网分数。

## 精确双分支混合公式

两分支等权，定义 `mu=(mu+ + mu-)/2`、`d=(mu+ - mu-)/2`、`Cbar=(C+ + C-)/2`、`deltaC=(C+ - C-)/2`、`deltaK3=(K3+ - K3-)/2`。`Sym` 表示完整置换平均。

```
merged C = Cbar + d d.T
merged K3 = (K3+ + K3-)/2 + 3 Sym(d tensor deltaC)
merged K4 = (K4+ + K4-)/2
          + 4 Sym(d tensor deltaK3)
          + 3 Sym(deltaC tensor deltaC) - 2 d tensor d tensor d tensor d
```

`d³` 因等权的正负偏移取消，`d² Cbar` 项与混合 covariance 的扣除抵消；仍不能丢掉 `d·deltaC` 或四阶的三类额外项。四阶公式还依赖两条件 K3 的差，单个 g/λ 状态不足以确定它。

[tenfold_conditional_merge_contract.py](./tenfold_conditional_merge_contract.py) 在 n4/8、每分支53/61支持点的独立任意非高斯有限分布上，直接算混合 centered moments 对照以上公式；K3误差≤2.67e-15，K4≤2.54e-14。另用合法计费操作验证下面 parent CP 表示，误差≤4.45e-16。当前脚本证据 [tenfold_conditional_merge_contract_r02.json](../../results/phase2/tenfold_conditional_merge_contract_r02.json)。这是代数合同，不证明实际条件模型误差或完整资源通过。

## 父来源格式的可表示性

父 full source 形式是三块：

```
Sym(3A+F1 R1.T, P, A*diag(w2)+F2 R2.T)
+ Sym(P*diag(s)+3A*diag(e)+Z ell.T, P, P)
+ Sym(A*diag(c1)+P*diag(c2), P, y*ones.T)
```

混合产生的额外 `3Sym(d tensor deltaC)` 可以独立作为 rank-n 第三块：`A=deltaC, P=I, c1=3, c2=0, y=d`；`w2/e/s/feedback/Zell` 全零。Cdiff 对称，因此 `X3 P.T=3deltaC` 的 parent feed 对称前提满足。D3、D21 与全异 K111 都必须一起计入，不能只塞入 D3 向量。

原两个分支的 K3 **tensor** 各乘 1/2 后求和。不可对 A/P/Z 或 CP 腿逐项平均，因为 tensor 是多线性函数。保留全部来源时有一个直接精确缩放：令 `gamma=(1/2)^(1/3)`，每个 source 的动态输运腿 `A/P/Z/F1/F2/y` 都乘 gamma，静态 `ell/w2/s/e/c1/c2/R1/R2` 保持。每块三腿因此共同乘 `gamma³=1/2`，包括Z内携带y的特殊列；feed对称前提仍保持。共享表示可保留Qc基，缩放FA/FP/Z/feedback，不把物理merged covariance carrier一起缩放。这里source的P只是第三阶tensor因子，缩放后不再解释为单位出生传播矩阵，但线性运输与来源dslices仍代数自洽；年龄、出生范围finder及covariance槽必须解耦，不能误把scaled P用作真实W或merged C。

真正最小的可靠表示改动是允许一般 CP source（或一类专用 merge source），显式保存其加权三腿，来源运输、dslices、年龄和共享 basis 都接受多来源同一层。另一方案将旧source的全部raw CP合法构成后合并并计费重压缩；这保持表示可控但通常增加全层成本，不能免费用离线 NumPy 生成因子。

## 现有循环不支持直接暂停/续跑

每次 `_predict_one_condition` 当前从零重建状态，结束只返回均值矩阵。早合链至少要拆为“run prefix→snapshot→resume suffix”：

1. 每分支 snapshot 必须复制当时已有均值、C、所有 source 参数、thin feedback、shared basis、年龄/slots、w1_prev、新born、g_prev/var_prev/lam_prev 和重排行列 identity；不能只保留池的 view。下一条件分支会复写整个持久 pool。
2. 两分支 pruning 的 perm 不同，需要逆置换到共同 neuron坐标，之后一次重排；conditional mean rider修正后的mu与实际 cumulants中心化采用的pk1v不是同一对象。若用修正mu混合“估计分布”，必须记录其近似，不能把它当精确有限分布合同。
3. 父栈容量 `(L-1)`、每层一个 newborn、`ka/kb` 为leading slots和固定年龄关口假设，合并后的来源数量约为前缀两倍再加merge source；需扩容与逐source出生年龄，不能复用原容量导致来源被覆盖。
4. 旧source可能已w1缩放，而当层newborn尚未缩放；snapshot边界要选post-ReLU更新全部结束，在resume时明确一次运输的gain，避免重复乘gate。
5. 初次suffix C 槽原来随唯一newborn骑运输family，两个branch newborn和merge source并存，需单独指定covariance carrier；它只有一份merged C。

这不是一次 wrapper 小改，直接推进会把吞吐问题变成未核验的状态迁移与tensor表示错误。

## K4代理不能平均结束

平均 g/λ 无法复现 `4Sym(d·deltaK3)+3Sym(deltaC²)-2d⁴` 的对角、K22、K31或capped K211。最小数学闭合需至少合法收缩混合 K4 与 parent harmonic projection：

```
T[j,j]=K4[j]+sum_i K22[i,j]
T[j,k]=K31[j,k]+K31[k,j]+sum_i K211[i,j,k]
G=cA*T+cI*trace(T)*I
```

再选择压缩成g/λ、完整densecore或现有proxy；任何一种都应单独承认表示误差。新增 K4 feed若只沿用平均旧λ，会与混合后来源和边际不一致。把全部 K4 设零可以成为独立受控降阶候选，但不能称保留原GH2改善的精确merge。

## 优先更小、保留合法分布的策略

在搭建完整迁移前，推荐**先做前缀分支后按层混合均值的诊断**，保存两分支每层误差与分歧，但不用最终reference来挑prefix；它可观察分歧是否在早层集中，却不能从均值分歧证明准确率。

另一项可快速实现的合法候选是 **一完整条件链 + 精确全分布控制变量**，而不是两个完整父链：用随机标准normal t（有合法seed）或固定quadrature时明确积分偏差，完整预测一支，再用便宜已知积分的Gaussian skeleton去减差异。只要控制变量的积分有独立解析证明，就保持目标分布；已有条件父bias仍存在，且单点variance可能很高。不能把条件幅度调小而仍宣称采样 `N(0,I)`；不合法的幅度变更改变输入 covariance。

更直接的费用机制是 **GH2父链共享计算**：两条件输入的 covariance完全相同，第一层Cpre以及只依赖sigma的部分重复；可缓存这些合法已计费对象。但非零mu导致gates、conditionalpostCov和全部source随后不同，不能共享第二层以上C或AP乘法。只有前层大头能共用，不会自动把2×45秒降成1×45秒。可提取firstCpre只计算一次，合同与资源后再扩展，不能假设lowercost足以完成十倍目标。

GH4没有可保证不增积分误差的两节点替代。降低父链精度/秩/Strassen层次必须另立单因素候选及原budget资源验收；不能只在某条件分支删除来源或采用固定测试网络自适应节点。root已测GH2 raw改善但费用拖累adjusted，这仍需成本与误差联合评估。

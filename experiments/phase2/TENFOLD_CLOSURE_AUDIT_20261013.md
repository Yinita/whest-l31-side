# 多层四阶闭合审计

2026-10-13；只读审查 r01/r02 候选、固定官方 commit `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`，没有访问 reference、没有改候选、没有新增整网分数。

## `g_prev` 的对角闭合正确

令 `T[j,k]=sum_i K4[i,i,j,k]`，所有低支持 slices 在各自不同坐标时计算。完整 trace 的精确分拆为：

```
T[j,j] = K4[j] + sum_{i != j} K22[i,j]
T[j,k] = K31[j,k] + K31[k,j] + sum_{i distinct j,k} K211[i,j,k]  (j != k)
G_identity = 6/(n+4)*T - 3/((n+2)*(n+4))*trace(T)*I
```

官方 `harmonic.py` 的 `lap`、`proj_coef`、`DS_harmonic_proj` 正是此投影。K31 和全异 K211 部分仅产生非对角 T；它们的对角严格为零，因此不产生 `trace(T)` 修正。没有“非对角行和应再反馈到 g_prev”的漏项。现有

```
g_prev = cA*(K4v+K22@ones) + cI*(sum(K4v)+sum(K22@ones))
```

是父模型已计算 slices 下的完整对角；同状态下改 offdiag→LS lambda 并保持 g_prev，投影闭合自洽。把新 K31/K211 行和加到 g_prev 反而会混淆两类指标支持。

一手源码：https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/harmonic.py 。既有独立小 tensor 合同 `probe_harmonic_core.py`/`results/phase2/harmonic_core_algebra.json` 验证此分拆及 metric 1/2 规范。

## r01/r02 完整性标签的准确范围

给定父模型已经表示的 K3 来源和已有四阶输入 slices，当前新增路径覆盖官方 factored-kept 的全部非对角 K31/K211：15 类 K31 图项、28 个低支持 K211 图项、K3 全异来源轨道、当层 hypertree 出生中心化、当前 K4 单轨道及 feed 中心化、post CovCov 抵消和统一谐投影。历史来源中的旧 K4 feed 与当层新 feed 不同，不额外重复旧 feed。

因此可以称“给定父模型输入表示的完整 retained 非对角 core 生产”。不能称“完整官方 K3augment 算法”或“真实四阶 cumulant”：父模型的旧源压缩、残差低秩、裁剪、经验均值修正、K4 再生与平均 metric 仍存在。原官方 `factor_k3.py` 本身还明确过滤某些非 hypertree 的 top K111 图项；保留它的过滤也是有条件的完整性。

K4/K22 父路径也保留原近似。官方对照中 `[11]^3` 的 K22 项曾由 pairq3 消融测试为负；它不是本轮新发现，不重复实验、不把该项遗漏解释为新增 offdiag 合同错误。

## metric 与尺度

官方 post-ReLU 投影返回 identity metric 的 core `A`。若把同一 tensor 改写为 `Sym(G tensor mI)`，必须取 `G=A/m`。V29 延续的再生规范在 pre-ReLU 固定使用平均 `m=2` 并沿用 post core 的 g/λ，属于原有 metric-reset/再生近似；r01/r02 保持 g 和 λ 相同尺度可与父模型做单因素比较，但不能声称真实 transported metric。

只给新 λ 除 2、保留 g 原值，会破坏已有两部分的相同规范。若要研究 metric 规范，需另立候选，同时调整 diagonal/offdiagonal、feed/use 及运输一致性，不能在本轮隐式修正。最佳 V37b 的边际 g4row 已使用真实 metric 对角 `sum(W²)`，其余矩阵 slices 仍采用平均 `2I`，这也是既有混合近似。

## LS 与下一层 beta 的限制

新的 post-ReLU LS 解决当前 Frobenius 最优标量：

```
lambda_base = <Goff, Cout> / <Cout, Cout>
```

设 `R=Goff-lambda_base*Cout`，它只保证 `<R,Cout>=0`。下一层权重使 residual 变为 `W R W.T`，可能含重要的对角和非对角；其最优 λ 未必仍等于当前值。diag(g)+λC 压缩会删除它，而一般 full matrix core 记忆可以保留。这是表示限制，不是 K31/K211 系数合同失败。

V25 的 multiplier 是依据旧冻结 λ 表和 `REF_R` 校准。新 λ 已依赖当前网络，再乘下一层 `clip(rr/REF_R)^BETA` 属于第二次经验校准。保留 beta 的首个多层候选有明确归因：仅换 base λ。若首次实测 poor raw，下一单因素可只关闭新 LS 层之后的 multiplier；不能同时更改 core/metric/dtype，并不能事先断言 beta 造成失败。

r01/r02 当前 LS 路径只替换 λ，成本和 raw 都需整网证据；数学组件更完整或组件合同通过，不意味着分数更高。

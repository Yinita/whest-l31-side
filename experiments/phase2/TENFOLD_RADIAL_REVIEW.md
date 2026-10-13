# R12 球面链径向转换审计

2026-10-13，只读审计；没有运行整网实验，没有修改候选、队列或研究状态。

审计候选：`estimators/tenfold_radial_sphere_r12/estimator.py`，SHA256 `cf0d3c001b0f542a3f6ffc23ab0c1169290ba023d951a72ea35a36d5a22699a2`。冻结母版：`sprint_public_v37b_h18_r01`，SHA256 `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`。

结论：首层 power cumulant 转换、represented K3 线性系数及新增 matrix/vector 通道的代数正确，可以进入严格单网筛选。小尺寸合同仅验证所表示对象的转换；没有证明真实球面整网 cumulant 精确，也没有十倍分数证据。

## 公式依据及本项目推导

[原论文 S.3.3](https://arxiv.org/html/2605.05179v2#S3.SS3) 定义 power cumulants 为不同坐标幂的 cumulants，并给出 slicewise 转为 ordinary cumulants 的 partition formula；其 S.1.4 明示 `K3(i,i,j)=Cov(Y_i²,Y_j)−2*mu_i*Cov(Y_i,Y_j)`。本次复核浏览了原论文。官方实现固定为 [factor_k3.py](https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py)，commit `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`，本地固定文件 SHA256 `e408aa1d8544e7d77b95dad5a548560098e7265845b7f713846ff8b165007918`。

[NIST chi-square 分布](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm) 给出独立标准正态平方和的密度及 Gamma 积分。由径向高斯密度与球坐标分解，可推导标准高斯 `XG=(R/sqrt(n))*XS`，`R²~chi-square(n)`，均匀方向 `XS/sqrt(n)` 与 R 独立。无 bias 的线性/ReLU 网络正齐次，故每层 raw moment 的总次数 p 满足

```
a_p = 2^(p/2)*n^(−p/2)*Gamma((n+p)/2)/Gamma(n/2)
M_G(p) = a_p*M_S(p)
a2 = 1; a3 = a1*(1+1/n); a4 = 1+2/n
```

以下径向逆变换是本项目从 raw moments 与中心化定义推导，并通过独立 chi4 积分复核的公式；不是宣称原论文发布了这个 R12 算法。令 `mg, cg, TG` 为 Gaussian first-post mean/cov/represented K3，则

```
ms = mg/a1
cs = cg+(1−1/a1²)*mg*mg.T
TS = TG/a3
   + (1/a3−1/a1)*sum3(cg_ij*mg_k)
   + (1/a3−3/a1+2/a1³)*mg³
```

`sum3` 是三个不同位置项相加，不能混用库中的 average-Sym。

## 逐项代码复核

| 对象 | before | R12 after | 判断 |
|---|---|---|---|
| pair 输入 | `pk11=Cov(Yi,Yj)`、`pk21=Cov(Yi²,Yj)`、`pk22=Cov(Yi²,Yj²)` | 先加回 mean products 得到 raw11/21/22，按次数除 a2/a3/a4，再中心化 | 正确；不能直接将 pk21/pk22 除径向系数 |
| marginal 1–4 | Gaussian raw moments | `pk1/a1, pk2, pk3/a3, pk4/a4`，随后 standard K4 中心化 | 正确 |
| K22 | Gaussian power/ordinary conversion | `ps22−2ms_j*ps21−2ms_i*ps21.T−2ps11²+4ms_i*ms_j*ps11`，对称、清零对角 | 正确；对角 K4 单独携带 |
| represented Gaussian K3 birth | A/P legs 与 w2/s/e/Rr | 完成母版 Gaussian birth 后，仅 w2、s、e、Rr 残差列分别 `/a3` | 整体 represented tensor `/a3`；保留 A 是必要的，提前覆盖 Cs 会改变 birth |
| covariance carrier | Gaussian first-post C | newborn P slot 复制 sphere C | 与下一层 sphere covariance 运输一致；和 Gaussian A leg 分开 |
| special K3 | 无 | B=cg、v=mg、pair=`1/a3−1/a1`、cube=`1/a3−3/a1+2/a1³` | 完整 sum3 matrix/vector 与 rank1 cube，包含 repeated 与 all-distinct |
| special D3 | 无 | `3*pair*diag(B)*v+cube*v³` | 与 tensor 对角一致 |
| special D21 | 无 | `pair*(diag(B)_i*v_j+2B_ij*v_i)+cube*v_i²*v_j`，清零对角 | 与 tensor (i,i,j) 一致 |
| first transport | first-post newborn raw W | li1 B→W B W.T，v→W v；li>=2 用 W diag(w1_prev) | 正确；初次就用 WD 会重复施加 first-ReLU gain |
| subsequent residual | 母版 `K3−D3_w−D3_new` 与 `K21−rep21` | D3/D21 先加 special，再参与原 residual subtraction | 避免 special repeated content 再出生；完整 nonlinear 真值仍取决于母版闭合 |
| reordered coordinates | 母版 W 的 rows/cols permutation | special 用同一 current W，B/v 保存 previous reordered coordinates | 顺序正确，special 不受 parent sources 裁剪 |
| first-output restoration | 原 Gaussian pk1 | sphere first-post mean `/a1` 后输出 `*a1`；depth1 final branch 不缩放未经转换的 pk1 | 默认 rider off 时，首层 mean 路径一致 |

`cg` 来自新 add allocation，不覆盖母版 pool；`mg=pk1v` 持有当层新 einsum PK1 的 view，下层重赋 PK1 不就地覆盖，v 生命周期安全。

## 已有独立合同与失败证据

报告 `results/phase2/tenfold_radial_sphere_contract_r01.json`：7 个合成角度原子 × 独立 chi4 径向积分，解析 mean/cov/K3/K4 相对误差最大约 `4.14e-14`。这是任意独立角分布的径向代数合同，不称合成角分布为均匀 sphere。

实际 production 合同 `results/phase2/tenfold_radial_production_contract_r12.json`，脚本 SHA256 `d9bc7882ad943220ebf7c95fa9e33587bf13f1017e39233701d79bfa3d77895c`：捕获真实母版 Gaussian represented birth，独立展开全部 dense K3，再和 R12 birth + special、实际下一层 D3/D21 比较。

| width/depth | full birth max absolute delta | second-pre D3 delta | D21 delta | parent / candidate FLOPs |
|---|---:|---:|---:|---:|
| 8 / 3 | 1.01110e-7 | 3.59045e-7 | 2.42991e-7 | 209105 / 218960 |
| 32 / 3 | 8.38956e-8 | 3.50449e-7 | 1.71349e-7 | 3286313 / 3627816 |

float32 合同通过。早期 first-special transport 使用 WD 的失败保存在 `results/phase2/tenfold_radial_production_contract_r12_retry.log`，width8 D3 偏差 `0.33625`、D21 `0.37889`；已修复为 first raw W，不能删除失败记录。trace instrumentation 不构成预算、资源或墙钟验收。

## 范围与必要后续检查

默认 `NO_CORR=1`，当前 strict supervisor 清除 V17 环境覆盖，第一层 delta=None；因此此轮不存在 rider 被删导致的实际分数混淆。非默认 `V17_NO_CORR=0` 时 CORR_BETA[0] 非零，first sphere conversion 使用 pk1/a1 会丢掉 first mean rider，不能将这种配置归因于纯输入 law 变化。`V19_NO_SRC_LAST=1` 探针分支不加 special；默认关闭，开启后是另一条科学路径。

R12 的 parent lowrank residual、年龄压缩、裁剪、K4 harmonic proxy 与 Gaussian 学得的 LAM/REF_R 仍是近似。完整 special 通道未裁剪，母版其他通道仍裁剪；小合同没有覆盖1024实际裁剪/共基路径，也没有证明所有第三层以后的新增 nonlinear residual 正确到真值。

每层 B sandwich 增加真实 dense GEMM，须严格实测 FLOPs、120s predict、0.4s residual、5s setup、memory 与零失败。若单网 raw 无足够信号或 adjusted 退步，不通过 node/参数盲扫包装成十倍。只有冻结 Mini-8、独立网络成对十倍、RPC 与资源检查全部通过后才递进目标。此前 CV 路线的负结果仍成立，不因新代数合同改记为分数进展。

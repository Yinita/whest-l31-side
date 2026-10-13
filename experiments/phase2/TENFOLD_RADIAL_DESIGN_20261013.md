# 单条球面母链及解析径向恢复：待实现消融

2026-10-13。版本 `tenfold-radial-sphere-r01`，本文件为目标无关设计及合同证据，**没有新整网raw/adjusted，不是已达标候选**。不改冻结parent、研究队列或score threshold。

## 单一科学机制

无bias线性+ReLU网络正齐次。标准高斯输入 `X=R U`，`R~chi_n` 与均匀sphere方向U独立。定义 `S=sqrt(n)*U`，每层都有

```
Y_G = (R/sqrt(n))*Y_S
a_p = E[(R/sqrt(n))^p]
m_G(raw degree p) = a_p*m_S(raw degree p)
E[Y_G] = a1*E[Y_S]
```

因此以**球面残余分布**走一条cumulant母链，最后每层mean乘a1，与原高斯目标一致；这里没有采样、reference、额外两条完整条件链。n1024下a1=`0.9997558892134076`，a2=1、a3=`(1+1/n)*a1`、a4=`1+2/n`。球面不是高斯，直接把parent输出乘a1会引入约2.44e-4相对偏差，不能作为该方法。

科学上可能有价值的原因：`Cov_G=Cov_S+(1-a1²)*mu_S mu_S.T` 中rank1项是持续的精确半径贡献，而当前memoryless K4代理可能逐层处理此相关结构时累积误差。这只是待验证假设；不能由半径方差小或n大直接推出十倍。

| 版本 | 名称 | before | after | 验收 |
|---|---|---|---|---|
| tenfold-radial-sphere-contract-r01 | 解析径向/球面合同 | 高斯或合成角分布×chi半径 | rawmoments degree scaling与centered K1–K4转换；球面首ReLU边缘积分 | 已完成；不是score |
| tenfold-radial-sphere-r01（未实现） | 单球面链 | 原高斯K3/H18链 | exact sphere初层1–4rawmoments + 新radial K3 block + 后续冻结parent + 每层a1恢复 | 先n8/32合同，后n1024 strict1，同cycle成对，raw需有大幅收益才扩大 |
| tenfold-radial-sphere-r02（未实现） | 单链持久解析radiuschannel | memoryless再生的radial代理 | 从sphere残余精确恢复Gaussian K4 USE/feed，单独持久knownradius贡献 | 科学机制另立；不在r01同时改LAM/metric |

## 已完成合同及实际数值

脚本 `experiments/phase2/tenfold_radial_sphere_contract.py`，原报告 `results/phase2/tenfold_radial_sphere_contract_r01.json`，0.56秒。使用7个合成角度原子、非线性正齐次两层真实权重，独立256点chi4径向积分，解析mean/cov/K3/K4中心化公式相对误差最大`4.14e-14`。这是任意独立角分布的代数合同，不将7原子称为均匀sphere或Gaussian网络真值。

另外对sphere坐标密度作独立192/384点Gauss–Legendre积分，固定n8/32/64/256/1024、alpha−2/−1/0/1/2、sigma1.3。n1024 alpha0：高斯首ReLU近似的mean误差`−1.26633e-4`；加入确切边缘K4后`−2.62255e-7`；加入边缘K6与K4²后`−1.48126e-9`。固定非零alpha时K4/K6符号与密度积分相符，但标量affine mean不满足完整网络的径向正齐次恢复，不能对非零alpha单scalar直接multiplya1声称Gaussian exact。

球面输入K2=I、K3=0，

```
K4_abcd = -2/(n+2)*(delta_ab delta_cd+delta_ac delta_bd+delta_ad delta_bc)
K4 marginal after linear W = -6*v_i²/(n+2)
K6 marginal = 240*v_i³/((n+2)(n+4))
```

首层预测若只使用Gaussianmode0而不调整initial K4，有明确可检出的偏差。边缘K6/K4square合同不是完整mixed高阶闭合，不能单独当完整K6方法。

## 精确半径转换的中心化公式

令sphere输出mean=m、cov=C、K3=T、K4=Q。`Sym`在以下表达中指把所有不同位置项**求和**，不是除以置换数；实现时必须统一factor库average-Sym约定。

```
muG = a1*m
CG = C+(1-a1²)*m*m.T
TG = a3*T+(a3-a1)*sum3(C_ij*m_k)
     +(a3-3a1+2a1³)*m³
QG = a4*Q+(a4-a1*a3)*sum4(T_ijk*m_l)
     +(a4-1)*sum3(C_ij*C_kl)
     +(a4-2a1*a3-1+2a1²)*sum6(C_ij*m_k*m_l)
     +(a4-4a1*a3-3+12a1²-6a1⁴)*m⁴
```

这些公式已同独立积分直接centered tensors比较。新路线核心是保留已知radiuschannel，避免把它重复解释为genericK4局部出生。完整sphere残余仍有其它高阶内容，不能声称删除rank1项已经精确。

## 可实施的首层状态重组

优先用Gaussian firstlayer→sphere转换，避免一开始改所有nonGaussian Wick/feed图项。Gaussian首层每神经元原始moments1–4都有解析公式，pair raw21/raw22已有父mode0；sphere对应degree p分别除a_p，随后用原PK→K中心化公式得到sphere mean/C/K3v/K4v/K21/K22。这些首层marginal/pair对象转换精确到原Gaussianpair程序近似与浮点误差。

具体首层 `muS=muG/a1`、`CS=CG+(1-1/a1²)*muGmuG.T`。CG父firstlayerGaussian covariance尚是截断pair核，不能称数值exact；程序保持完整represented对象并原样记录。

**不能只变mean/C然后沿用Gaussian K3 sources。** 全distinct T_S必须按上式逆变换：

```
T_S = (T_G-(a3-a1)*sum3(CS_ij*muS_k)
           -(a3-3a1+2a1³)*muS³)/a3
```

当前parent Gaussian firstbirth representedtensor可整段乘1/a3；sphere新增 `Sym(CS⊗muS)` 用rank-n matrix×vector block、`muS³`用rank1 block。birth repeatedpart必须从完整block去除，再补转换得到的exactK3v/K21；现parent M-source residual处理约定需小shape独立tensor核验，不沿用Gaussian S3c和S21直接混搭。新增首层两个source块存活15层，成本需实测；并非零费或只O(n²)。复用matrix腿时可能把covariance×mean特殊结构改为薄channel，但须完整合同。

后续使用冻结H18 Wick/source闭合及固定LAM/REF_R作为第一单因素球面路线；这些表原为Gaussian训练，潜在分布错配必须注明，先不重新拟合或同时beta=0。sphere K4post来自新的K4v/K22和原harmonicprojection，但普通代理不等于完整sphereK4。每层最终stackmean乘a1恢复Gaussian目标，reference保持官方原样。

备选从initialsphereK4直接改li0：firstpre V=w0.Tw0、core=γV/metric=V，γ=−6/(n+2)。完整USE slices为g4_i=γv_i²、K22_ij=γ(v_i v_j+2Vij²)/3、K31_ij=γv_iVij。必须改mode0去执行相应terms、D3/D21零而K4非零。官方 `factor_k3.py`395–453要求三个K4→K3feed，当metric dense时#2不能省；core∝metric使其与leadingB1结构相关，但fullV对角拆分、重复坐标剔除及传播不能靠缩放既有offdiagB1解决。此备选更复杂，应在Gaussianfirststate重组路径之后评估。

## 严格判定

先冻结上述一个机制，smallshape tensor parity；然后真实n1024同cycleparent/candidate单网raw、adjusted、FLOPs、120s/.4s/5s、memory与失败。当前stage必要raw<5.779e-9，任何仅0.2%/2%收益不足以进入十倍确认。候选若有足够raw信号，再Min8及独立network相同reference成对十倍、RPC/数值与资源；无合格证据不上传、不改变阈值。保留负结果。

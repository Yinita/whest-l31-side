# 一次条件母链与廉价积分差校正

新机制定义 `P0=P[X|u·X=0]` 的昂贵父链近似，`S(t)` 为只依赖权重的廉价条件Gaussian skeleton，以固定GH8计算：

```
estimate = P0 + sum_a weight_a*S(node_a) - S(0)
```

u沿用r08/r09权重产生的expected-mean Jacobian方向，所有node在同一BudgetContext中串行；无需额外昂贵父pilot、测试真值、条件幅度调参。合适称呼是“确定性积分差近似校正”。它不是零均值已知的随机控制变量或无偏恒等式：`P0` 估计一个条件切片，`S` 的t响应若不准确，结果可恶化。

## 防止重复和不一致

昂贵母链必须使用 rank1023 输入cov `I-u u.T`、inputmean0；不能拿原无条件V37b输出再加 `E[S(t)]-S(0)`，那会重复加入一维贡献。廉价S(0)与各S(t)必须相同skeleton/共用conditionalcovariance，不能用原unconditionalS作扣减。无需平均sourcelegs或mergeK3，母链完全独立，校正只加到最终每层mean输出。

原父meanrider、LAM/REF_R、source压缩、pruning仍用于conditional母链；经验条件bias可能影响校正。无条件规则不可能仅凭代数保证保留r09的raw改善。

## 廉价表示的两个版本

**diag**：第一层使用完整条件Cpre，后层只传递variance和mean。高斯ReLU边际mean/variance均精确公式；忽略中间相关性。固定方向后第一层 Cpre 与sigma独立于t，可合法缓存一次并共用9个node（GH8及0），数值合同后实施。后层每node每层数个矩阵向量乘；新增费用大致O(9Ln²)，比第二完整父链便宜。

**fullcov-firstchaos**：每层 `Cpre=W.T@C@W`；高斯边际ReLU精确，postcov用

```
Cpost = diag(Phi)*Cpre*diag(Phi)
      + diag(max(Var[ReLU]-Phi²*Varpre,0))
```

此式保留firstHermite chaos跨神经元cov和完整边际variance。由于diag更高阶Hermite variance非负，PSD传播具有清楚构造，但不是精确bivariateGaussian ReLU covariance，少了第二及更高chaos的offdiag。每node每层2GEMM；GH8加0共9次，n1024depth16float64粗估新增约1.17e12FLOPs，父链另约.45e12，总约.74总预算。该估算没有真实wall/residual检查，单线程120秒可能不满足。

若实现secondHermite chaos offdiag `0.5*outer(w21,w21)*Cpre²`，可以增加一个合法独立候选；coef及diagonal必须重新去重。不能直接加入后仍将第一chaos的diagvariance correction保留全部，否则重复第二阶diag。它属于新精度机制，不在当前toy gate中。

## 已完成的target-free toy gate

[tenfold_conditioning_cv_toy.py](./tenfold_conditioning_cv_toy.py) 使用3个固定随机n3depth4网络。unconditional toy truth用offline GH40/GH56三维积分，conditionalzero用二维GH56积分，因此把母链P0误差从这项诊断中排除，只检验S校正的质量。GH40→56变化逐层保留，有限积分非精确真值。解析数据不使用比赛reference或整网score。

证据 [tenfold_conditioning_cv_toy_r01.json](../../results/phase2/tenfold_conditioning_cv_toy_r01.json)：

| toy | 末层不校正MSE | diag校正MSE | fullcov-firstchaos校正MSE |
|---|---:|---:|---:|
| 0 | 2.9891e-3 | 8.3219e-6 | 9.8844e-7 |
| 1 | 2.1737e-5 | 4.6960e-6 | 4.3309e-6 |
| 2 | 5.3536e-4 | **6.2581e-4** | 1.4821e-4 |

diag第三个toy发生恶化；fullcov三个末层均改善，但后层accuracy尺度仍离“精确”很远。这个有限样本不用于调方向/幅度，不证明1024×16泛化。first-layerGH8本身在大方向投影时也有积分截断，toy0/1首层校正残差约1e-4，不能称首层完全精确。

## 建议实验顺序

先跑**diag校正单网**是最低工程/资源代价的可证伪pilot，并明确可能失败；一条t0完整父链加9条便宜skeleton，不用做两完整链。只有同原reference的raw收益和费用联合通过才扩大。若diag不佳，下一候选fullcov-firstchaos与先验证节点缓存/父callwall成本，不能调小分布方向幅度来救分。

当前只提供独立设计与toy组件，不改root候选，不新增线上或完整1024分数。

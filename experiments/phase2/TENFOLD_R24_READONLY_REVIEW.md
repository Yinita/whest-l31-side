# r24 / canonical capture 独立只读审计

2026-10-13。没有修改已运行脚本。报告和两个当时源码snapshot/SHA保存于`results/phase2/tenfold_completed_state_attribution_r24_readonly_audit_r01.json`。

当前r24实数结果未发现矩系数、排列、prediction配对或V29/H18混用错误。`covariance_causal_parent_trace_r02`源是H18 SHA `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`；aggregate历史summary的`rows.V29`来自另一旧cache，但r24仅取aggregate中的两流MC矩，parent对象用H18自己的canonical trace与prediction。parent完整链positive/cross与既有fullchain final吻合至6.2e-22/1.4e-22。

独立核验original reference name `logan-fitzgerald`、rawseed6319981554997072999、weights逐值等于RPC原始float32。两strict网络ID均为该network0，graderseed3909487726。两trace firstmode D3/g4为zero，PK1与自prediction完全相同，feature形状分别16×1024×8/11且均finite。跨层canonical `mu_l ≈ W_l.T @ own_prediction_(l−1)`最大差2.470e-6/2.618e-6，符合当前float32/Strassen轨迹；inverseperm方向与输出恢复一致。

固定ReLU cumulant响应为K2 phi/(2sigma)、K3 −alpha phi/(6sigma²)、K4 (alpha²−1)phi/(24sigma³)，与冻结归因一致。MCvariance无偏M2/(N−1)，k4中用populationvar=M2/N；r24沿用该约定。样本三/四阶cumulant有限N偏差未完全消除，cross只扣独立随机误差部分；没有严格置信界或候选十倍意义。propagation每版本分别使用该call自己的cdf和prediction。

可修健壮性缺口：

- capture只清V17–V29，未清影响当前source的V30_、V36_、V37_。本次当前环境这些变量均未设，证据没有污染；未来新版本应清理所有源码读取的Vxx配置或保存完整明确env，避免同sourceSHA不同算法。
- r24应断言feature_order与canonical flag，original name/rawseed/weights、aggregate权重数据commit、strict network IDs均一致。目前事实已独立核验，缺口尚未导致此次错误。
- 当前trace numerical drift已报告：r20 final capture raw2.146677618256e-8 vs strict2.144773350256e-8，trace max2.861e-6；不使用strict误差配trace状态计算oracle residual。

机制结论：r20 final K2 missing cross6.077882368108e-8→5.850246982847e-8（约3.75%），K3 6.526329187255e-5→6.515891717511e-5（约0.16%），K4 5.397729258022e-7→5.397468660915e-7（约0.005%）。localinnovation4.419388523453e-9→4.438109598406e-9，jointoracle local residual几乎不变。r20证明一部分合法遗漏项有真实小改进；继续只追residual rank不具备已验证十倍机制，需要联合生产K2/K3/K4突破。

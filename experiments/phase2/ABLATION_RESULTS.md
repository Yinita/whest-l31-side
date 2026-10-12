# WhestBench 消融执行结果

总任务129。当前状态：{'completed_positive': 12, 'blocked_derivation': 58, 'completed_negative': 3, 'pending': 44, 'completed_diagnostic': 8, 'partial_diagnostic': 3, 'partial_derivation': 1}。完成快筛不等于独立确认；阶段1尚未达标。

| 版本 | key | 状态 | before raw | after raw | before adjusted | after adjusted | adjusted变化 | FLOPs变化 | 失败 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| v30.001 | C01 | completed_positive | 2.29046648315e-08 | 2.29064394119e-08 | 6.11683233498e-09 | 6.08690650979e-09 | -0.489237% | -0.496946% | 0 |
| v30.003 | C03 | completed_negative | 2.29046648315e-08 | 2.29254784045e-08 | 6.11683233498e-09 | 6.12256687422e-09 | +0.093750% | +0.002877% | 0 |
| v30.005 | C05 | completed_positive | 2.29046648315e-08 | 2.28935501667e-08 | 6.11683233498e-09 | 6.11390439228e-09 | -0.047867% | +0.000659% | 0 |
| v30.006 | C06 | completed_positive | 2.29046648315e-08 | 2.29046417388e-08 | 6.11683233498e-09 | 6.11682904772e-09 | -0.000054% | +0.000047% | 0 |
| v30.007 | C07 | completed_positive | 2.29046648315e-08 | 2.29046648315e-08 | 6.11683233498e-09 | 6.11682139186e-09 | -0.000179% | -0.000179% | 0 |
| v30.008 | C08 | completed_positive | 2.29046648315e-08 | 2.28957883763e-08 | 6.11683233498e-09 | 6.11445658898e-09 | -0.038839% | -0.000086% | 0 |
| v30.010 | C10 | completed_positive | 2.29046648315e-08 | 2.29046648315e-08 | 6.11683233498e-09 | 6.11621386699e-09 | -0.010111% | -0.010111% | 0 |
| v30.011 | C11 | completed_positive | 2.29046648315e-08 | 2.28947385494e-08 | 6.11683233498e-09 | 6.09062553181e-09 | -0.428437% | -0.385267% | 0 |
| v30.012 | C12 | completed_negative | 2.28947385494e-08 | 2.29095018511e-08 | 6.09062553181e-09 | 6.09455297313e-09 | +0.064483% | +0.000000% | 0 |
| v30.014 | C14 | completed_positive | 2.29046648315e-08 | 2.2903094532e-08 | 6.11683233498e-09 | 6.11624916111e-09 | -0.009534% | -0.002678% | 0 |
| v30.015 | C15 | completed_positive | 2.29046648315e-08 | 2.29023822129e-08 | 6.11683233498e-09 | 6.11622274759e-09 | -0.009966% | +0.000000% | 0 |
| v30.022 | C22 | completed_negative | 2.29046648315e-08 | 2.29066454693e-08 | 6.11683233498e-09 | 6.11736893529e-09 | +0.008773% | +0.000125% | 0 |
| v30.031 | H01 | completed_positive | 2.29046648315e-08 | 2.2881678774e-08 | 6.11683233498e-09 | 6.11070767115e-09 | -0.100128% | +0.000228% | 0 |
| v30.032 | H02 | completed_positive | 2.29046648315e-08 | 2.29027996568e-08 | 6.11683233498e-09 | 6.11635118582e-09 | -0.007866% | +0.000277% | 0 |
| v30.048 | H18 | completed_positive | 2.2702818514e-08 | 2.218065287e-08 | 5.77897975577e-09 | 5.64681649387e-09 | -2.286965% | +0.002807% | 0 |

## 归因与证据

C07/C10单网raw逐值评分相同，收益来自收费减少。C08/C14/C15的微小raw差尚不能排除浮点重排漂移，completed_positive只表示初筛分数更低。诊断条目不产生算法after分数，记录为不适用。

### v30.001 C01 递归限制C_pre上三角输出

before：V29仅在第一层输出分块计算11、12、22三个block；两个对角block内部仍完整计算

after：各尺度只算上非对角block，对角block递归到32×32，最后对角叶子批量dense；下非对角copy transpose

状态：completed_positive

predict=28.503507s；residual=0.311622s；FLOPs=584344372719；peak child RSS=3897800KiB；失败=0。

- [ablation_v30_001_r01_mini1_local.json](../../results/phase2/ablation_v30_001_r01_mini1_local.json)
- [ablation_v30_001_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_001_r01_mini1_local.manifest.json)
- [ablation_v30_001_r01_mini1_local.log](../../results/phase2/ablation_v30_001_r01_mini1_local.log)
- [ablation_v30_001_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_001_r01_mini1_local.launcher.log)
- [ablation_v30_001_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_001_r01_mini1_local.validate.log)

### v30.003 C03 将Z与Zf薄腿合并运输

before：同一WD分别乘Z_st与Zf_st，两个GEMM调用

after：将两个薄列组装成一个持久stack，用一次WD矩阵乘法，再view split

状态：completed_negative

验收：Rejected: FLOPs increased despite item no-increase gate。

predict=28.801238s；residual=0.240171s；FLOPs=587279650287；peak child RSS=3999824KiB；失败=0。

- [ablation_v30_003_r01_mini1_local.json](../../results/phase2/ablation_v30_003_r01_mini1_local.json)
- [ablation_v30_003_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_003_r01_mini1_local.manifest.json)
- [ablation_v30_003_r01_mini1_local.log](../../results/phase2/ablation_v30_003_r01_mini1_local.log)
- [ablation_v30_003_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_003_r01_mini1_local.launcher.log)
- [ablation_v30_003_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_003_r01_mini1_local.validate.log)

### v30.005 C05 PPL与Z的末次薄腿收缩写成扁平GEMM

before：einsum kiq,kcq->ic产生sum_k,q PPL_kiq Z_kcq

after：k和q合并为列，计算PPL_flat @ Z_flat.T

状态：completed_positive

验收：Rejected: FLOPs increased despite item no-increase gate。

predict=29.216297s；residual=0.238922s；FLOPs=587266625007；peak child RSS=3968564KiB；失败=0。

- [ablation_v30_005_r01_mini1_local.json](../../results/phase2/ablation_v30_005_r01_mini1_local.json)
- [ablation_v30_005_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_005_r01_mini1_local.manifest.json)
- [ablation_v30_005_r01_mini1_local.log](../../results/phase2/ablation_v30_005_r01_mini1_local.log)
- [ablation_v30_005_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_005_r01_mini1_local.launcher.log)
- [ablation_v30_005_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_005_r01_mini1_local.validate.log)

### v30.006 C06 最后一层M残差的D3只做对角收缩

before：trim仍构造完整Z L.T，再乘P并对P收缩

after：用PP @ L后逐元素乘Z、沿k,q求和，不构造完整Z L.T

状态：completed_positive

predict=29.496303s；residual=0.242965s；FLOPs=587263030767；peak child RSS=3921276KiB；失败=0。

- [ablation_v30_006_r01_mini1_local.json](../../results/phase2/ablation_v30_006_r01_mini1_local.json)
- [ablation_v30_006_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_006_r01_mini1_local.manifest.json)
- [ablation_v30_006_r01_mini1_local.log](../../results/phase2/ablation_v30_006_r01_mini1_local.log)
- [ablation_v30_006_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_006_r01_mini1_local.launcher.log)
- [ablation_v30_006_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_006_r01_mini1_local.validate.log)

### v30.007 C07 省掉最终层四阶矩阵切片

before：trim时regen仍形成g22c及完整wk4m，后续没有读取

after：trim时仅计算dG和g4row，不创建wk4m

状态：completed_positive

predict=28.287719s；residual=0.234743s；FLOPs=587261703663；peak child RSS=3921660KiB；失败=0。

- [ablation_v30_007_r01_mini1_local.json](../../results/phase2/ablation_v30_007_r01_mini1_local.json)
- [ablation_v30_007_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_007_r01_mini1_local.manifest.json)
- [ablation_v30_007_r01_mini1_local.log](../../results/phase2/ablation_v30_007_r01_mini1_local.log)
- [ablation_v30_007_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_007_r01_mini1_local.launcher.log)
- [ablation_v30_007_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_007_r01_mini1_local.validate.log)

### v30.008 C08 最终层仅运行PK1均值输出

before：trim仍计算(1,),(2,),(3,),(4,)四个一维PK1输出

after：trim专用progs仅包含(1,)且只写pk1v

状态：completed_positive

predict=27.798984s；residual=0.230228s；FLOPs=587262251608；peak child RSS=3969364KiB；失败=0。

- [ablation_v30_008_r01_mini1_local.json](../../results/phase2/ablation_v30_008_r01_mini1_local.json)
- [ablation_v30_008_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_008_r01_mini1_local.manifest.json)
- [ablation_v30_008_r01_mini1_local.log](../../results/phase2/ablation_v30_008_r01_mini1_local.log)
- [ablation_v30_008_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_008_r01_mini1_local.launcher.log)
- [ablation_v30_008_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_008_r01_mini1_local.validate.log)

### v30.010 C10 稀疏Wick selector改indexed选取

before：SL2/SR2为onehot/单系数表却做dense GEMM @ WT

after：依已有wl,wr indices选择WT行并乘coef

状态：completed_positive

predict=27.952077s；residual=0.243611s；FLOPs=587203376623；peak child RSS=3974988KiB；失败=0。

- [ablation_v30_010_r01_mini1_local.json](../../results/phase2/ablation_v30_010_r01_mini1_local.json)
- [ablation_v30_010_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_010_r01_mini1_local.manifest.json)
- [ablation_v30_010_r01_mini1_local.log](../../results/phase2/ablation_v30_010_r01_mini1_local.log)
- [ablation_v30_010_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_010_r01_mini1_local.launcher.log)
- [ablation_v30_010_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_010_r01_mini1_local.validate.log)

### v30.011 C11 非线性term按输出组分开收缩

before：IND2 onehot但统一einsum输出三个g通道

after：按pk11/pk21/pk22将term批次分组，只求各自sum

状态：completed_positive

predict=27.463335s；residual=0.250184s；FLOPs=585000224239；peak child RSS=4002416KiB；失败=0。

- [ablation_v30_011_r01_mini1_local.json](../../results/phase2/ablation_v30_011_r01_mini1_local.json)
- [ablation_v30_011_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_011_r01_mini1_local.manifest.json)
- [ablation_v30_011_r01_mini1_local.log](../../results/phase2/ablation_v30_011_r01_mini1_local.log)
- [ablation_v30_011_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_011_r01_mini1_local.launcher.log)
- [ablation_v30_011_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_011_r01_mini1_local.validate.log)

### v30.012 C12 去重AB矩阵乘积缓存

before：AB2按term重复fnp.multiply相同pair进abbuf

after：每unique unordered AB pair只算一次，term通过index view复用

状态：completed_negative

验收：No FLOP reduction relative to explicit C11 parent; no claimed gain。

predict=27.677843s；residual=0.256349s；FLOPs=585000224239；peak child RSS=4002268KiB；失败=0。

- [ablation_v30_012_r01_mini1_local.json](../../results/phase2/ablation_v30_012_r01_mini1_local.json)
- [ablation_v30_012_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_012_r01_mini1_local.manifest.json)
- [ablation_v30_012_r01_mini1_local.log](../../results/phase2/ablation_v30_012_r01_mini1_local.log)
- [ablation_v30_012_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_012_r01_mini1_local.launcher.log)
- [ablation_v30_012_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_012_r01_mini1_local.validate.log)

### v30.014 C14 避免二次covariance对称平均

before：pk11已平均其transpose；assembly又C=(K11+K11.T)*0.5

after：若pk11逐值确实对称且finite，copy K11后fill diagonal

状态：completed_positive

predict=27.772471s；residual=0.232835s；FLOPs=587247025647；peak child RSS=3921008KiB；失败=0。

- [ablation_v30_014_r01_mini1_local.json](../../results/phase2/ablation_v30_014_r01_mini1_local.json)
- [ablation_v30_014_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_014_r01_mini1_local.manifest.json)
- [ablation_v30_014_r01_mini1_local.log](../../results/phase2/ablation_v30_014_r01_mini1_local.log)
- [ablation_v30_014_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_014_r01_mini1_local.launcher.log)
- [ablation_v30_014_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_014_r01_mini1_local.validate.log)

### v30.015 C15 D21绝对值行和替代ones GEMV

before：D21n=abs(D21) @ ones_n

after：D21n=sum(abs(D21),axis=1)

状态：completed_positive

predict=27.781575s；residual=0.230117s；FLOPs=587262754287；peak child RSS=3916268KiB；失败=0。

- [ablation_v30_015_r01_mini1_local.json](../../results/phase2/ablation_v30_015_r01_mini1_local.json)
- [ablation_v30_015_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_015_r01_mini1_local.manifest.json)
- [ablation_v30_015_r01_mini1_local.log](../../results/phase2/ablation_v30_015_r01_mini1_local.log)
- [ablation_v30_015_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_015_r01_mini1_local.launcher.log)
- [ablation_v30_015_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_015_r01_mini1_local.validate.log)

### v30.022 C22 PK1 duplicate一维因子缓存去重

before：B1stack每termAB但有重复ones,d3,g4pair

after：一维unique AB factors及term indexed选择

状态：completed_negative

predict=33.644752s；residual=0.297424s；FLOPs=587263489519；peak child RSS=3968804KiB；失败=0。

- [ablation_v30_022_r01_mini1_local.json](../../results/phase2/ablation_v30_022_r01_mini1_local.json)
- [ablation_v30_022_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_022_r01_mini1_local.manifest.json)
- [ablation_v30_022_r01_mini1_local.log](../../results/phase2/ablation_v30_022_r01_mini1_local.log)
- [ablation_v30_022_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_022_r01_mini1_local.launcher.log)
- [ablation_v30_022_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_022_r01_mini1_local.validate.log)

### v30.024 C24 逐组件实际计费归因

before：只有全网FLOPs，不知source/PK/core/transport占比

after：单网诊断分别计入明确operator区段，另跑同hash总数复核

状态：completed_diagnostic

诊断结论：Official actual operator cost sum matches total; semantic component attribution remains optional followup；算法分数前后不适用。

- [ablation_v30_024_operator_costs.json](../../results/phase2/ablation_v30_024_operator_costs.json)

### v30.025 C25 浮点加法顺序与误差归因

before：单次raw小差异容易归因新机制

after：same-source repeated run，固定线程库和inputs，输出逐层预测difference

状态：completed_diagnostic

predict=30.551551s；residual=0.250971s；FLOPs=587262754287；peak child RSS=3921232KiB；失败=0。

前次快筛：Mini1 development only; no independent generalization claim；raw=2.2898658969e-08；adjusted=6.11522843228e-09；报告 `results/phase2/ablation_v30_025_r01_mini1_local.json`。

前次快筛：Mini1 development only; no independent generalization claim；raw=2.290385126e-08；adjusted=6.1166150657e-09；报告 `results/phase2/ablation_v30_025_r02_mini1_local.json`。

- [ablation_v30_025_r03_mini1_local.json](../../results/phase2/ablation_v30_025_r03_mini1_local.json)
- [ablation_v30_025_r03_mini1_local.manifest.json](../../results/phase2/ablation_v30_025_r03_mini1_local.manifest.json)
- [ablation_v30_025_r03_mini1_local.log](../../results/phase2/ablation_v30_025_r03_mini1_local.log)
- [ablation_v30_025_r03_mini1_local.launcher.log](../../results/phase2/ablation_v30_025_r03_mini1_local.launcher.log)
- [ablation_v30_025_r03_mini1_local.validate.log](../../results/phase2/ablation_v30_025_r03_mini1_local.validate.log)
- [ablation_v30_025_repeat_summary.json](../../results/phase2/ablation_v30_025_repeat_summary.json)
- [ablation_v30_025_prediction_drift.json](../../results/phase2/ablation_v30_025_prediction_drift.json)

### v30.026 C26 持久pool最大shape资源审计

before：pool在width/depth切换后保留历史大buffer，subprocess内存失败归因不清

after：诊断同进程shape序列与峰值RSS/RLIMIT_AS、pool keys/bytes

状态：partial_diagnostic

诊断结论：Actual pool retains old shape allocations; full estimator maximum scratch/resource audit remains pending；算法分数前后不适用。

- [ablation_v30_026_pool_shape_probe.json](../../results/phase2/ablation_v30_026_pool_shape_probe.json)
- [ablation_v30_026_full_pool_r0.json](../../results/phase2/ablation_v30_026_full_pool_r0.json)
- [ablation_v30_026_full_pool_r1.json](../../results/phase2/ablation_v30_026_full_pool_r1.json)
- [ablation_v30_026_full_pool_r2.json](../../results/phase2/ablation_v30_026_full_pool_r2.json)

### v30.031 H01 边际D3平方产品补全

before：实际V29 p1含D3²w(6,1)/72；p2,p3,p4均只含Gaussian、D3线性、g4线性

after：仅对p2,p3,p4新增D3²w(6,p)/72，p1保持原表

状态：completed_positive

predict=28.261638s；residual=0.239960s；FLOPs=587264090607；peak child RSS=3921848KiB；失败=0。

- [ablation_v30_031_r01_mini1_local.json](../../results/phase2/ablation_v30_031_r01_mini1_local.json)
- [ablation_v30_031_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_031_r01_mini1_local.manifest.json)
- [ablation_v30_031_r01_mini1_local.log](../../results/phase2/ablation_v30_031_r01_mini1_local.log)
- [ablation_v30_031_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_031_r01_mini1_local.launcher.log)
- [ablation_v30_031_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_031_r01_mini1_local.validate.log)

### v30.032 H02 边际D3乘g4产品补全

before：实际V29 p1含D3*g4*w(7,1)/144；p2,p3,p4无该产品

after：仅对p2,p3,p4新增D3*g4*w(7,p)/144

状态：completed_positive

predict=27.668589s；residual=0.240839s；FLOPs=587264382447；peak child RSS=3922316KiB；失败=0。

- [ablation_v30_032_r01_mini1_local.json](../../results/phase2/ablation_v30_032_r01_mini1_local.json)
- [ablation_v30_032_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_032_r01_mini1_local.manifest.json)
- [ablation_v30_032_r01_mini1_local.log](../../results/phase2/ablation_v30_032_r01_mini1_local.log)
- [ablation_v30_032_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_032_r01_mini1_local.launcher.log)
- [ablation_v30_032_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_032_r01_mini1_local.validate.log)

### v30.033 H03 单轴截断集合一致性诊断

before：四个单轴幂矩使用不同cumulant产品集合

after：建立实际逐产品差异表并用独立多项式oracle判断差异是否为有意剪枝

状态：completed_diagnostic

诊断结论：Exact current truncation inventory confirms p2/3/4 omit both products；算法分数前后不适用。

- [ablation_v30_033_inventory.json](../../results/phase2/ablation_v30_033_inventory.json)

### v30.048 H18 K4边际真实metric系数

before：g4row=METRIC_C*dG with fixed mean metric

after：只将边际metric改为current diag(W W^T)，保持Gproxy和其他通道

状态：completed_positive

逐网络时间/residual见原始报告，不与单网比较aggregate；FLOPs=4478163825528；peak child RSS=6707620KiB；失败=0。

前次快筛：network0 development only; no generalization claim；raw=2.26948380089e-08；adjusted=6.06095895252e-09；报告 `results/phase2/ablation_v30_048_r01_mini1_local.json`。

- [ablation_v30_048_r02_mini8_local.json](../../results/phase2/ablation_v30_048_r02_mini8_local.json)
- [ablation_v30_048_r02_mini8_local.manifest.json](../../results/phase2/ablation_v30_048_r02_mini8_local.manifest.json)
- [ablation_v30_048_r02_mini8_local.log](../../results/phase2/ablation_v30_048_r02_mini8_local.log)
- [ablation_v30_048_r02_mini8_local.launcher.log](../../results/phase2/ablation_v30_048_r02_mini8_local.launcher.log)
- [ablation_v30_048_r02_mini8_local.validate.log](../../results/phase2/ablation_v30_048_r02_mini8_local.validate.log)
- [ablation_v30_048_r02_paired_mini8.json](../../results/phase2/ablation_v30_048_r02_paired_mini8.json)
- [ablation_v30_048_r03_mini1_rpc.json](../../results/phase2/ablation_v30_048_r03_mini1_rpc.json)
- [ablation_v30_048_r03_mini1_rpc_scored.json](../../results/phase2/ablation_v30_048_r03_mini1_rpc_scored.json)
- [ablation_v30_048_r04_mini8_rpc_scored.json](../../results/phase2/ablation_v30_048_r04_mini8_rpc_scored.json)

### v30.073 O11 K111乘covariance edge完整公式核验

before：只有静态遗漏判定

after：独立显式tensor与diagram枚举一致

状态：completed_diagnostic

诊断结论：distinct-index explicit formula error0; official 3 Sym coefficient3; repeated indices masked; filtered mechanism, not full production parity；算法分数前后不适用。

- [ablation_v30_073_edge_formula.json](../../results/phase2/ablation_v30_073_edge_formula.json)
- [ablation_v30_073_edge_coefficient.json](../../results/phase2/ablation_v30_073_edge_coefficient.json)

### v30.074 O12 K111乘edge的exact final D3成本证明

before：仅已知rank扩大障碍

after：证明exact复杂度和内存或给出反例

状态：partial_derivation

诊断结论：Exact no-cubic-intermediate diagonal contraction verified; O(R*n^3), not unqualified O(n^3). Full cumulative repeated-content production and charged operator formula remain pending；算法分数前后不适用。

- [ablation_v30_074_edge_cost.json](../../results/phase2/ablation_v30_074_edge_cost.json)

### v30.086 O24 高阶边缘同阶组合inventory

before：选择性补项缺完整预算说明

after：budget和exact coefficient可追溯inventory

状态：completed_diagnostic

诊断结论：Exact scalar derivative-degree<=8 inventory; 32 output rows; not multiaxis diagram inventory；算法分数前后不适用。

- [ablation_v30_086_marginal_inventory.json](../../results/phase2/ablation_v30_086_marginal_inventory.json)

### v30.087 O25 degree5/6 ordinary与power累积量独立转换

before：仅degree≤4转换probe

after：degree5/6转换独立parity

状态：completed_diagnostic

诊断结论：6 degree5/6 repeated-index slices on17 finite-support atoms; official pK->K coefficients versus independent moments/set partitions maxerror6.66e-16; not full production tensor execution；算法分数前后不适用。

- [ablation_v30_087_degree6_conversion.json](../../results/phase2/ablation_v30_087_degree6_conversion.json)

### v30.097 L02 冻结网络级训练选择流程

before：Mini8同时用于既有快筛；缺少修正器专用网络级选择协议。

after：IDs0–7用leave-one-network-out；IDs8–15保留为一次公共确认集；另列新种子最终独立集。

状态：partial_diagnostic

诊断结论：Network-level leave-one-network-out selection frozen without reading confirmation targets; feature cache prerequisite L01 pending；算法分数前后不适用。

- [ablation_v30_097_selection_protocol.json](../../results/phase2/ablation_v30_097_selection_protocol.json)

### v30.125 L30 末层尾概率数值稳定性

before：极端alpha路径只由原正态cdf/pdf与float32演算处理。

after：保持原公式，新增离线极端alpha及sigma小值数值对照。

状态：completed_diagnostic

诊断结论：32 alpha/sigma combinations finite and monotonic; max relative error1.1498e-5 against70digit independent integral; scalar Gaussian mean path only；算法分数前后不适用。

- [ablation_v30_125_relu_tail.json](../../results/phase2/ablation_v30_125_relu_tail.json)

### v30.127 L33 参考dtype贡献诊断

before：官方targets以原float32使用，误差可能受量化贡献。

after：离线估计原targets float32 ULP尺度与MSE阈值关系，保留原targets不重评分。

状态：completed_diagnostic

诊断结论：Mini8 float32 uniform rounding proxy 6.120378688436097e-16; sampling uncertainty not estimated；算法分数前后不适用。

- [ablation_v30_127_reference_dtype.json](../../results/phase2/ablation_v30_127_reference_dtype.json)

### v30.129 L35 多次选择偏差账本

before：长消融列表容易反复查看同一确认集而失去独立性。

after：每次拟合/调参/查看reference都记录selection ledger，确认集首次查看后降为开发集。

状态：partial_diagnostic

诊断结论：Backlog candidate/report/hash selection ledger generated; pre-backlog screens and full training manifest integration pending；算法分数前后不适用。

- [ablation_v30_129_selection_ledger.json](../../results/phase2/ablation_v30_129_selection_ledger.json)


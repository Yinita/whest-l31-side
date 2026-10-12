# WhestBench 消融执行结果

总任务129。当前状态：{'completed_positive': 18, 'completed_negative': 10, 'partial_derivation': 40, 'completed_diagnostic': 20, 'partial_diagnostic': 24, 'partial_negative': 15, 'pending': 2}。完成快筛不等于独立确认；阶段1尚未达标。

| 版本 | key | 状态 | before raw | after raw | before adjusted | after adjusted | adjusted变化 | FLOPs变化 | 失败 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| v30.001 | C01 | completed_positive | 2.2702818514e-08 | 2.27084098192e-08 | 5.77897975577e-09 | 5.75045498951e-09 | -0.493595% | -0.518029% | 0 |
| v30.002 | C02 | completed_negative | 2.29046648315e-08 | 2.29302052901e-08 | 6.11683233498e-09 | 6.15664203556e-09 | +0.650822% | +0.538714% | 0 |
| v30.003 | C03 | completed_negative | 2.29046648315e-08 | 2.29254784045e-08 | 6.11683233498e-09 | 6.12256687422e-09 | +0.093750% | +0.002877% | 0 |
| v30.004 | C04 | completed_negative | 2.29046648315e-08 | 2.29219256909e-08 | 6.11683233498e-09 | 6.12153417628e-09 | +0.076867% | +0.001507% | 0 |
| v30.005 | C05 | completed_positive | 2.29046648315e-08 | 2.28935501667e-08 | 6.11683233498e-09 | 6.11390439228e-09 | -0.047867% | +0.000659% | 0 |
| v30.006 | C06 | completed_positive | 2.29046648315e-08 | 2.29046417388e-08 | 6.11683233498e-09 | 6.11682904772e-09 | -0.000054% | +0.000047% | 0 |
| v30.007 | C07 | completed_positive | 2.29046648315e-08 | 2.29046648315e-08 | 6.11683233498e-09 | 6.11682139186e-09 | -0.000179% | -0.000179% | 0 |
| v30.008 | C08 | completed_positive | 2.29046648315e-08 | 2.28957883763e-08 | 6.11683233498e-09 | 6.11445658898e-09 | -0.038839% | -0.000086% | 0 |
| v30.009 | C09 | completed_positive | 2.29046648315e-08 | 2.29010748143e-08 | 6.11683233498e-09 | 6.11572771322e-09 | -0.018059% | -0.002385% | 0 |
| v30.010 | C10 | completed_positive | 2.29046648315e-08 | 2.29046648315e-08 | 6.11683233498e-09 | 6.11621386699e-09 | -0.010111% | -0.010111% | 0 |
| v30.011 | C11 | completed_positive | 2.29046648315e-08 | 2.28947385494e-08 | 6.11683233498e-09 | 6.09062553181e-09 | -0.428437% | -0.385267% | 0 |
| v30.012 | C12 | completed_negative | 2.28947385494e-08 | 2.29095018511e-08 | 6.09062553181e-09 | 6.09455297313e-09 | +0.064483% | +0.000000% | 0 |
| v30.013 | C13 | completed_negative | 2.29046648315e-08 | 2.29035599375e-08 | 6.11683233498e-09 | 6.14547572661e-09 | +0.468272% | +0.473118% | 0 |
| v30.014 | C14 | completed_positive | 2.29046648315e-08 | 2.2903094532e-08 | 6.11683233498e-09 | 6.11624916111e-09 | -0.009534% | -0.002678% | 0 |
| v30.015 | C15 | completed_positive | 2.29046648315e-08 | 2.29023822129e-08 | 6.11683233498e-09 | 6.11622274759e-09 | -0.009966% | +0.000000% | 0 |
| v30.016 | C16 | completed_positive | 2.2702818514e-08 | 2.27096401684e-08 | 5.77897975577e-09 | 5.750992776e-09 | -0.484289% | -0.513692% | 0 |
| v30.017 | C17 | completed_positive | 2.29046648315e-08 | 2.29078516156e-08 | 6.11683233498e-09 | 6.11241916501e-09 | -0.072148% | -0.086049% | 0 |
| v30.018 | C18 | completed_positive | 2.29046648315e-08 | 2.29071996927e-08 | 6.11683233498e-09 | 6.11611893274e-09 | -0.011663% | -0.022727% | 0 |
| v30.019 | C19 | completed_positive | 2.29046648315e-08 | 2.28964882609e-08 | 6.11683233498e-09 | 6.11494351448e-09 | -0.030879% | +0.004821% | 0 |
| v30.021 | C21 | completed_negative | 2.28947385494e-08 | 2.29046648315e-08 | 6.09062553181e-09 | 6.09291669639e-09 | +0.037618% | -0.005736% | 0 |
| v30.022 | C22 | completed_negative | 2.29046648315e-08 | 2.29066454693e-08 | 6.11683233498e-09 | 6.11736893529e-09 | +0.008773% | +0.000125% | 0 |
| v30.027 | C27 | completed_negative | 2.29046648315e-08 | 2.29075904912e-08 | 6.11683233498e-09 | 6.1176136505e-09 | +0.012773% | +0.000000% | 0 |
| v30.031 | H01 | completed_positive | 2.29046648315e-08 | 2.2881678774e-08 | 6.11683233498e-09 | 6.11070767115e-09 | -0.100128% | +0.000228% | 0 |
| v30.032 | H02 | completed_positive | 2.29046648315e-08 | 2.29027996568e-08 | 6.11683233498e-09 | 6.11635118582e-09 | -0.007866% | +0.000277% | 0 |
| v30.048 | H18 | completed_positive | 2.2702818514e-08 | 2.218065287e-08 | 5.77897975577e-09 | 5.64681649387e-09 | -2.286965% | +0.002807% | 0 |
| v30.049 | H19 | completed_negative | 2.29046648315e-08 | 2.27738645719e-08 | 6.11683233498e-09 | 7.08275516689e-09 | +15.791226% | +16.456266% | 0 |
| v30.050 | H20 | completed_negative | 2.29046648315e-08 | 2.31574261988e-08 | 6.11683233498e-09 | 7.20257411616e-09 | +17.750066% | +16.464834% | 0 |
| v30.052 | H22 | completed_positive | 2.2702818514e-08 | 2.26464844655e-08 | 5.77897975577e-09 | 5.76448249921e-09 | -0.250862% | +0.013369% | 0 |

## 离线学习快筛

仅为开发网络LOO raw误差，尚无合法候选的adjusted分数。

| 版本 | key | before raw | after raw | raw变化 | adjusted |
|---|---|---:|---:|---:|---|
| v30.098 | L03 | 2.27016599414e-08 | 2.28241554222e-08 | +0.539588% | 未计费验证 |
| v30.099 | L04 | 2.27016599414e-08 | 2.27967393532e-08 | +0.418821% | 未计费验证 |
| v30.100 | L05 | 2.27016599414e-08 | 2.28401608462e-08 | +0.610092% | 未计费验证 |
| v30.101 | L06 | 2.27016599414e-08 | 2.28442126705e-08 | +0.627940% | 未计费验证 |
| v30.102 | L07 | 2.27015416385e-08 | 2.28343933704e-08 | +0.585210% | 未计费验证 |
| v30.103 | L08 | 2.27016599414e-08 | 2.28241554222e-08 | +0.539588% | 未计费验证 |
| v30.104 | L09 | 2.27016599414e-08 | 2.28416995875e-08 | +0.616870% | 未计费验证 |
| v30.105 | L10 | 2.27016599414e-08 | 2.33958133005e-08 | +3.057721% | 未计费验证 |
| v30.106 | L11 | 2.27016599414e-08 | 2.4684767688e-08 | +8.735519% | 未计费验证 |
| v30.107 | L12 | 2.27016599414e-08 | 2.2825262429e-08 | +0.544465% | 未计费验证 |
| v30.108 | L13 | 2.27016599414e-08 | 2.27400364103e-08 | +0.169047% | 未计费验证 |
| v30.109 | L14 | 2.27016599414e-08 | 2.28363089459e-08 | +0.593124% | 未计费验证 |
| v30.110 | L15 | 2.27016599414e-08 | 2.31375281148e-08 | +1.919984% | 未计费验证 |
| v30.124 | L29 | 2.27016596632e-08 | 3.44085568893e-08 | +51.568464% | 未计费验证 |

## 归因与证据

C07/C10单网raw逐值评分相同，收益来自收费减少。C08/C14/C15的微小raw差尚不能排除浮点重排漂移，completed_positive只表示初筛分数更低。诊断条目不产生算法after分数，记录为不适用。

### v30.001 C01 递归限制C_pre上三角输出

before：V29仅在第一层输出分块计算11、12、22三个block；两个对角block内部仍完整计算

after：各尺度只算上非对角block，对角block递归到32×32，最后对角叶子批量dense；下非对角copy transpose

状态：completed_positive

逐网络时间/residual见原始报告，不与单网比较aggregate；FLOPs=4454840605560；peak child RSS=6682348KiB；失败=0。

前次快筛：Mini1 development only; no independent generalization claim；raw=2.29064394119e-08；adjusted=6.08690650979e-09；报告 `results/phase2/ablation_v30_001_r01_mini1_local.json`。

- [ablation_v30_001_r02_mini8_local.json](../../results/phase2/ablation_v30_001_r02_mini8_local.json)
- [ablation_v30_001_r02_mini8_local.manifest.json](../../results/phase2/ablation_v30_001_r02_mini8_local.manifest.json)
- [ablation_v30_001_r02_mini8_local.log](../../results/phase2/ablation_v30_001_r02_mini8_local.log)
- [ablation_v30_001_r02_mini8_local.launcher.log](../../results/phase2/ablation_v30_001_r02_mini8_local.launcher.log)
- [ablation_v30_001_r02_mini8_local.validate.log](../../results/phase2/ablation_v30_001_r02_mini8_local.validate.log)

### v30.002 C02 Strassen–Winograd共享加法链

before：现_Strassen采用标准7乘法18加减的组合/组装

after：保持每层7个乘法，使用Winograd15加减共享中间链

状态：completed_negative

predict=27.773421s；residual=0.240963s；FLOPs=590426419695；peak child RSS=5776884KiB；失败=0。

前次快筛：Mini1 development only; no independent generalization claim；raw=0.943476319313；adjusted=0.943476319313；报告 `results/phase2/ablation_v30_002_r01_mini1_local.json`。

- [ablation_v30_002_r02_mini1_local.json](../../results/phase2/ablation_v30_002_r02_mini1_local.json)
- [ablation_v30_002_r02_mini1_local.manifest.json](../../results/phase2/ablation_v30_002_r02_mini1_local.manifest.json)
- [ablation_v30_002_r02_mini1_local.log](../../results/phase2/ablation_v30_002_r02_mini1_local.log)
- [ablation_v30_002_r02_mini1_local.launcher.log](../../results/phase2/ablation_v30_002_r02_mini1_local.launcher.log)
- [ablation_v30_002_r02_mini1_local.validate.log](../../results/phase2/ablation_v30_002_r02_mini1_local.validate.log)

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

### v30.004 C04 将两条D21反馈收缩合并薄列

before：GYR R2/F2与GXR R1/F1分别GEMM后sum

after：合并[GYR,GXR]和[F2,F1]薄列一次乘法并一次k-sum

状态：completed_negative

predict=28.215100s；residual=0.235373s；FLOPs=587271601646；peak child RSS=3975412KiB；失败=0。

- [ablation_v30_004_r01_mini1_local.json](../../results/phase2/ablation_v30_004_r01_mini1_local.json)
- [ablation_v30_004_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_004_r01_mini1_local.manifest.json)
- [ablation_v30_004_r01_mini1_local.log](../../results/phase2/ablation_v30_004_r01_mini1_local.log)
- [ablation_v30_004_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_004_r01_mini1_local.launcher.log)
- [ablation_v30_004_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_004_r01_mini1_local.validate.log)

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

### v30.009 C09 稀疏wick多项式代替dense表乘

before：APOW @ C1、APOW @ C2、SB @ SELC计入零系数乘法

after：逐WICK_PAIR用明确稀疏Horner/单项式选择构造

状态：completed_positive

predict=28.130338s；residual=0.257135s；FLOPs=587248745967；peak child RSS=3924308KiB；失败=0。

- [ablation_v30_009_r01_mini1_local.json](../../results/phase2/ablation_v30_009_r01_mini1_local.json)
- [ablation_v30_009_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_009_r01_mini1_local.manifest.json)
- [ablation_v30_009_r01_mini1_local.log](../../results/phase2/ablation_v30_009_r01_mini1_local.log)
- [ablation_v30_009_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_009_r01_mini1_local.launcher.log)
- [ablation_v30_009_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_009_r01_mini1_local.validate.log)

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

### v30.013 C13 PK11和PK22对称输出block化

before：每组计算完整PK矩阵再平均transpose

after：每个上三角输出block同时合并正向/反向term，复制下三角

状态：completed_negative

predict=27.790413s；residual=0.266294s；FLOPs=590041202159；peak child RSS=3924376KiB；失败=0。

- [ablation_v30_013_r01_mini1_local.json](../../results/phase2/ablation_v30_013_r01_mini1_local.json)
- [ablation_v30_013_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_013_r01_mini1_local.manifest.json)
- [ablation_v30_013_r01_mini1_local.log](../../results/phase2/ablation_v30_013_r01_mini1_local.log)
- [ablation_v30_013_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_013_r01_mini1_local.launcher.log)
- [ablation_v30_013_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_013_r01_mini1_local.validate.log)

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

### v30.016 C16 小核心Sj只计算上三角

before：Sj=FAj diag(dA) FAj.T+FPj diag(dP) FPj.T完整GEMM

after：对真正对称的weighted Gram计算上三角block并copy下三角

状态：completed_positive

逐网络时间/residual见原始报告，不与单网比较aggregate；FLOPs=4455034786680；peak child RSS=6716400KiB；失败=0。

前次快筛：Mini1 development only; no independent generalization claim；raw=2.28905161492e-08；adjusted=6.08189781882e-09；报告 `results/phase2/ablation_v30_016_r01_mini1_local.json`。

前次快筛：Mini8 development only; no independent generalization claim；raw=0.213074359073；adjusted=0.213074345897；报告 `results/phase2/ablation_v30_016_r02_mini8_local.json`。

- [ablation_v30_016_r03_mini8_local.json](../../results/phase2/ablation_v30_016_r03_mini8_local.json)
- [ablation_v30_016_r03_mini8_local.manifest.json](../../results/phase2/ablation_v30_016_r03_mini8_local.manifest.json)
- [ablation_v30_016_r03_mini8_local.log](../../results/phase2/ablation_v30_016_r03_mini8_local.log)
- [ablation_v30_016_r03_mini8_local.launcher.log](../../results/phase2/ablation_v30_016_r03_mini8_local.launcher.log)
- [ablation_v30_016_r03_mini8_local.validate.log](../../results/phase2/ablation_v30_016_r03_mini8_local.validate.log)

### v30.017 C17 Sg双边旋转的末次乘法对称化

before：Sg=Tq @ (Sg @ Tq.T)完整输出

after：保留第一乘法，末次上三角block计算并copy

状态：completed_positive

predict=28.293577s；residual=0.338997s；FLOPs=586757419119；peak child RSS=3967088KiB；失败=0。

- [ablation_v30_017_r01_mini1_local.json](../../results/phase2/ablation_v30_017_r01_mini1_local.json)
- [ablation_v30_017_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_017_r01_mini1_local.manifest.json)
- [ablation_v30_017_r01_mini1_local.log](../../results/phase2/ablation_v30_017_r01_mini1_local.log)
- [ablation_v30_017_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_017_r01_mini1_local.launcher.log)
- [ablation_v30_017_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_017_r01_mini1_local.validate.log)

### v30.018 C18 S2子空间投影的末次乘法三角化

before：S2=Un.T @ (G2 @ Un)完整r2×r2

after：仅末次计算上三角输出并copy

状态：completed_positive

predict=28.053274s；residual=0.284572s；FLOPs=587129284559；peak child RSS=3933056KiB；失败=0。

- [ablation_v30_018_r01_mini1_local.json](../../results/phase2/ablation_v30_018_r01_mini1_local.json)
- [ablation_v30_018_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_018_r01_mini1_local.manifest.json)
- [ablation_v30_018_r01_mini1_local.log](../../results/phase2/ablation_v30_018_r01_mini1_local.log)
- [ablation_v30_018_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_018_r01_mini1_local.launcher.log)
- [ablation_v30_018_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_018_r01_mini1_local.validate.log)

### v30.019 C19 joiner两路Gram operator共享布局

before：Aj.T Om→Aj weighted结果与Pj两路四GEMM

after：组[Aj,Pj]在一个持久layout两batched GEMM，最后相加

状态：completed_positive

predict=29.327210s；residual=0.248162s；FLOPs=587291065839；peak child RSS=3968104KiB；失败=0。

- [ablation_v30_019_r01_mini1_local.json](../../results/phase2/ablation_v30_019_r01_mini1_local.json)
- [ablation_v30_019_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_019_r01_mini1_local.manifest.json)

### v30.020 C20 final D3 feedback的线性两项薄收缩

before：D3 feedback先显式Xt、Yt完整n²矩阵，再计算A·Yt和Xt·A·w2

after：在trim将两个线性项重排为(AP) @ R2T与(AP·w2) @ R1T再与F2/F1逐行收缩，XtYt交叉项保持旧式

状态：partial_derivation

诊断结论：Exact two linearterm factor formulas checked; crossXtYt still needs denseXt/Yt; naive newGEMMs increase cost, candidate/full score pending；算法分数前后不适用。

- [ablation_v30_020_feedback_cost.json](../../results/phase2/ablation_v30_020_feedback_cost.json)

### v30.021 C21 nonlinearity中广播d3/g4向量不materialize

before：d3col/g4col先乘ones_row变n²矩阵，再用于AB products

after：只保存n×1/1×n view；fnp广播时构造真正需要的AB

状态：completed_negative

predict=26.507250s；residual=0.247601s；FLOPs=584966669807；peak child RSS=3994384KiB；失败=0。

- [ablation_v30_021_r01_mini1_local.json](../../results/phase2/ablation_v30_021_r01_mini1_local.json)
- [ablation_v30_021_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_021_r01_mini1_local.manifest.json)
- [ablation_v30_021_r01_mini1_local.log](../../results/phase2/ablation_v30_021_r01_mini1_local.log)
- [ablation_v30_021_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_021_r01_mini1_local.launcher.log)
- [ablation_v30_021_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_021_r01_mini1_local.validate.log)

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

### v30.023 C23 数值对称误差审计

before：目前X=W C舍入而输出视为对称，没有量化逐层偏差

after：离线包装捕获C,X,Y,按原同一浮点数据比较XY和XY.T

状态：completed_diagnostic

诊断结论：Network0 state/cores plus raw X@Y offline asymmetry captured; original chargedFLOPs saved; no time score；算法分数前后不适用。

- [ablation_v30_023_symmetry_trace.json](../../results/phase2/ablation_v30_023_symmetry_trace.json)

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

### v30.027 C27 shape变更时释放无用旧pool

before：pool保留之前n/L的buffer；当前已有smm重建但_Pool按key存历史shape

after：当模型shape发生变化明确新建_Pool并释放历史引用

状态：completed_negative

predict=27.788835s；residual=0.233054s；FLOPs=587262754287；peak child RSS=3921036KiB；失败=0。

- [ablation_v30_027_r01_mini1_local.json](../../results/phase2/ablation_v30_027_r01_mini1_local.json)
- [ablation_v30_027_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_027_r01_mini1_local.manifest.json)
- [ablation_v30_027_r01_mini1_local.log](../../results/phase2/ablation_v30_027_r01_mini1_local.log)
- [ablation_v30_027_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_027_r01_mini1_local.launcher.log)
- [ablation_v30_027_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_027_r01_mini1_local.validate.log)

### v30.028 C28 分区清理不再需要的scratch常驻

before：最终层仍保留早层宽大bufs，8GiB虚拟内存接近界限

after：明确final-only路径可复用废弃PK2/legs buffers而非申请新scratch

状态：partial_derivation

诊断结论：Static scratch use bounds inventoried; dynamic escapedview alias/poison contracts required before reuse；算法分数前后不适用。

- [ablation_v30_028_static_liveness.json](../../results/phase2/ablation_v30_028_static_liveness.json)
- [ablation_v30_028_poison_alias_contract.json](../../results/phase2/ablation_v30_028_poison_alias_contract.json)

### v30.029 C29 三角C_pre阶段批量copy调度

before：C01逐blockPython循环触发多次copyto

after：保持相同上三角算法，将同尺度的输入/输出布局用持久块slab按批次copy

状态：partial_negative

诊断结论：Exact layout packing charges75% more; full scatter and candidate pending；算法分数前后不适用。

- [ablation_v30_029_layout_contract.json](../../results/phase2/ablation_v30_029_layout_contract.json)
- [ablation_v30_029_scatter_contract.json](../../results/phase2/ablation_v30_029_scatter_contract.json)

### v30.030 C30 不同backend上合法matmul计费与resource对照

before：math/local与RPC、普通subprocess环境行为差异已有但新重组未确认

after：固定candidate/source/data同网络逐operator/API和报告对照

状态：partial_diagnostic

诊断结论：C01 full Mini8 local/RPC measured; ordinary subprocess original resource-cap diagnostic still pending；算法分数前后不适用。

- [ablation_v30_030_r01_c01_mini8_rpc_scored.json](../../results/phase2/ablation_v30_030_r01_c01_mini8_rpc_scored.json)
- [ablation_v30_001_r02_mini8_local.json](../../results/phase2/ablation_v30_001_r02_mini8_local.json)

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

### v30.034 H04 精确Gaussian pair核替换

before：纯Gaussian pair使用有限Hermite系列

after：用原始Gaussian ReLU幂矩核的合法在线插值替换pK11/pK21/pK22纯Gaussian部分

状态：partial_derivation

诊断结论：Zero/nonzeroGaussian pair quadrature checked; actual correlation distribution, highorder subtraction and lawfulcandidate pending；算法分数前后不适用。

- [ablation_h04_pair_gap.json](../../results/phase2/ablation_h04_pair_gap.json)
- [ablation_v30_034_nonzero_pair_contract.json](../../results/phase2/ablation_v30_034_nonzero_pair_contract.json)

### v30.035 H05 Gaussian pair尾项有理重求和

before：pure covariance pair finite polynomial

after：加入由Gaussian核解析系数约束的rational tail，同时保持已知低阶项

状态：partial_derivation

诊断结论：Fixed zero-mean analytic Pade tail and pole-free denominator derived; nonzero/candidate/cost pending；算法分数前后不适用。

- [ablation_v30_035_pair_tail_pade.json](../../results/phase2/ablation_v30_035_pair_tail_pade.json)

### v30.036 H06 未剪枝SIMPLE生产表恢复

before：V29主体62项剪枝表

after：恢复固定生成器的未剪枝SIMPLE生产表，其余闭合不变

状态：partial_derivation

诊断结论：Lean static table25 current official vs older generator count recorded; source version affects diagram budget. Candidate mapping/reproduction pending；算法分数前后不适用。

- [ablation_v30_036_simple_table.json](../../results/phase2/ablation_v30_036_simple_table.json)
- [ablation_v30_036_r02_generator_history.json](../../results/phase2/ablation_v30_036_r02_generator_history.json)

### v30.037 H07 WK22到B1出生来源恢复

before：B1删除上游d(w3)*WK22*d(w2)类项

after：仅恢复该出生项并扣除其已重复dslices

状态：partial_derivation

诊断结论：Arbitrary source birth products checked; official coefficients, charged lowrank representation and candidate pending；算法分数前后不适用。

- [ablation_h07_h12_birth_derivation.json](../../results/phase2/ablation_h07_h12_birth_derivation.json)
- [ablation_h07_h10_wk22_separable_rank.json](../../results/phase2/ablation_h07_h10_wk22_separable_rank.json)
- [ablation_h07_h10_distinct_contract.json](../../results/phase2/ablation_h07_h10_distinct_contract.json)
- [ablation_h07_h10_thin_transport_contract.json](../../results/phase2/ablation_h07_h10_thin_transport_contract.json)

### v30.038 H08 B1的WK12对称反馈恢复

before：V29只保留部分D21 thin feedback

after：补充上游B1中d(w2)*WK12*d(w2)缺失部分

状态：partial_derivation

诊断结论：Arbitrary source birth products checked; official coefficients, charged lowrank representation and candidate pending；算法分数前后不适用。

- [ablation_h07_h12_birth_derivation.json](../../results/phase2/ablation_h07_h12_birth_derivation.json)

### v30.039 H09 B2 leading D21乘C来源恢复

before：B2出生整体DS-shadow

after：仅恢复WK12乘WK11 leading hub，移除其重复出生dslices

状态：partial_derivation

诊断结论：Arbitrary source birth products checked; official coefficients, charged lowrank representation and candidate pending；算法分数前后不适用。

- [ablation_h07_h12_birth_derivation.json](../../results/phase2/ablation_h07_h12_birth_derivation.json)

### v30.040 H10 B2 WK22乘C来源恢复

before：B2的WK22乘WK11 fully mixed channel缺失

after：仅恢复该channel而保留B2其他shadow

状态：partial_derivation

诊断结论：Arbitrary source birth products checked; official coefficients, charged lowrank representation and candidate pending；算法分数前后不适用。

- [ablation_h07_h12_birth_derivation.json](../../results/phase2/ablation_h07_h12_birth_derivation.json)
- [ablation_h07_h10_wk22_separable_rank.json](../../results/phase2/ablation_h07_h10_wk22_separable_rank.json)
- [ablation_h07_h10_distinct_contract.json](../../results/phase2/ablation_h07_h10_distinct_contract.json)
- [ablation_h07_h10_thin_transport_contract.json](../../results/phase2/ablation_h07_h10_thin_transport_contract.json)

### v30.041 H11 B2 D21平方来源恢复

before：B2的WK12乘WK21等D21产品shadow

after：只恢复D21-square symmetric source

状态：partial_derivation

诊断结论：Arbitrary source birth products checked; official coefficients, charged lowrank representation and candidate pending；算法分数前后不适用。

- [ablation_h07_h12_birth_derivation.json](../../results/phase2/ablation_h07_h12_birth_derivation.json)

### v30.042 H12 B2 WK22乘D21来源恢复

before：B2高阶mixed birth shadow

after：只恢复WK22乘D21来源轨道

状态：partial_derivation

诊断结论：Arbitrary source birth products checked; official coefficients, charged lowrank representation and candidate pending；算法分数前后不适用。

- [ablation_h07_h12_birth_derivation.json](../../results/phase2/ablation_h07_h12_birth_derivation.json)

### v30.043 H13 B2保留一次线性运输再shadow

before：B2从出生就删除

after：保留完整指定B2 newborn one transition，随后仅保留其已产生边际/slice修正

状态：partial_derivation

诊断结论：FiniteCP one/twosteptransport contract; actualdiagram/repeat/eviction pending；算法分数前后不适用。

- [ablation_h13_h14_h30_lifetime_contract.json](../../results/phase2/ablation_h13_h14_h30_lifetime_contract.json)

### v30.044 H14 B2两步输出导向收缩

before：B2完全缺失

after：运输指定下一两层mean/variance敏感收缩而不保存完整B2

状态：partial_derivation

诊断结论：FiniteCP one/twosteptransport contract; actualdiagram/repeat/eviction pending；算法分数前后不适用。

- [ablation_h13_h14_h30_lifetime_contract.json](../../results/phase2/ablation_h13_h14_h30_lifetime_contract.json)

### v30.045 H15 历史K3来源生成K211 capped低rank核心

before：当前g_prev只由K4和K22对角汇总，offdiag由lambda*C代理

after：从actual historical K3 sources求K211 capped，再保留明确lowrank core residual

状态：partial_derivation

诊断结论：CP fullcontractionrank and distinctcap corrections checked; diagonalexclusion may makematrixfullrank; fullproduction/candidate pending；算法分数前后不适用。

- [ablation_v30_045_source_cap_rank_bound.json](../../results/phase2/ablation_v30_045_source_cap_rank_bound.json)
- [ablation_v30_045_radial_capped_contract.json](../../results/phase2/ablation_v30_045_radial_capped_contract.json)

### v30.046 H16 actual K31低rank核心残差

before：lambda*C代理offdiag core

after：从actual K31 production加12/(n+4)*Sym(K31)的去重低rank残差

状态：partial_derivation

诊断结论：ExactK31conversion/projectioncoeff; actualproduction/proxyremoval pending；算法分数前后不适用。

- [ablation_v30_046_k31_conversion_contract.json](../../results/phase2/ablation_v30_046_k31_conversion_contract.json)
- [ablation_v30_046_gaussian_core_projection.json](../../results/phase2/ablation_v30_046_gaussian_core_projection.json)

### v30.047 H17 Gaussian K211完整二阶矩阵核心

before：已测Gaussian core只压成scalar lambda且失败

after：保持核验Gaussian capped K211 matrix，作为去重core residual直接运输

状态：partial_derivation

诊断结论：Exact three Gaussian covariance paths and distinct cappedsum; full cumulant/coreproxy subtraction and legal candidate pending；算法分数前后不适用。

- [ablation_v30_047_gaussian_capped_k211.json](../../results/phase2/ablation_v30_047_gaussian_capped_k211.json)
- [k211_wing_quadrature.json](../../results/phase2/k211_wing_quadrature.json)

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

### v30.049 H19 WK431真实metric混合项

before：WK431=(METRIC_C/2)*lambda*Coff

after：对固定Gproxy使用0.5*(Gij*Mii+Gii*Mij)，仅改WK431 use

状态：completed_negative

predict=28.808261s；residual=0.239175s；FLOPs=683904274401；peak child RSS=3946664KiB；失败=0。

- [ablation_v30_049_r01_mini1_local.json](../../results/phase2/ablation_v30_049_r01_mini1_local.json)
- [ablation_v30_049_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_049_r01_mini1_local.manifest.json)
- [ablation_v30_049_r01_mini1_local.log](../../results/phase2/ablation_v30_049_r01_mini1_local.log)
- [ablation_v30_049_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_049_r01_mini1_local.launcher.log)
- [ablation_v30_049_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_049_r01_mini1_local.validate.log)

### v30.050 H20 WK22真实metric交叉项

before：wk4m=(METRIC_C/6)*(dGi+dGj)

after：使用(Gii*Mjj+Gjj*Mii+4Gij*Mij)/6，仅改WK22 use

状态：completed_negative

predict=28.545874s；residual=0.241713s；FLOPs=683954591727；peak child RSS=3946384KiB；失败=0。

- [ablation_v30_050_r01_mini1_local.json](../../results/phase2/ablation_v30_050_r01_mini1_local.json)
- [ablation_v30_050_r01_mini1_local.manifest.json](../../results/phase2/ablation_v30_050_r01_mini1_local.manifest.json)
- [ablation_v30_050_r01_mini1_local.log](../../results/phase2/ablation_v30_050_r01_mini1_local.log)
- [ablation_v30_050_r01_mini1_local.launcher.log](../../results/phase2/ablation_v30_050_r01_mini1_local.launcher.log)
- [ablation_v30_050_r01_mini1_local.validate.log](../../results/phase2/ablation_v30_050_r01_mini1_local.validate.log)

### v30.051 H21 恢复非对角metric的K4到K3 feed

before：mean diagonal metric下上游某feed项为零

after：使用Moff恢复该被零化feed来源

状态：partial_derivation

诊断结论：Official three-factor1/4,1,1/4 matches direct distinct K211 Wick feed; repeat/boundedrank/cost integration pending；算法分数前后不适用。

- [ablation_v30_051_nondiagonal_feed_contract.json](../../results/phase2/ablation_v30_051_nondiagonal_feed_contract.json)

### v30.052 H22 D21反馈basis输出敏感sketch

before：D21 feedback sketch用当前weight列

after：相同rank16下改用未来一两层weight和live average ReLU gate构造sketch

状态：completed_positive

逐网络时间/residual见原始报告，不与单网比较aggregate；FLOPs=4478636790648；peak child RSS=6712364KiB；失败=0。

前次快筛：Mini1 development only; no independent generalization claim；raw=2.23081926265e-08；adjusted=5.95829990493e-09；报告 `results/phase2/ablation_v30_052_r01_mini1_local.json`。

- [ablation_v30_052_r03_mini8_local.json](../../results/phase2/ablation_v30_052_r03_mini8_local.json)
- [ablation_v30_052_r03_mini8_local.manifest.json](../../results/phase2/ablation_v30_052_r03_mini8_local.manifest.json)
- [ablation_v30_052_r03_mini8_local.log](../../results/phase2/ablation_v30_052_r03_mini8_local.log)
- [ablation_v30_052_r03_mini8_local.launcher.log](../../results/phase2/ablation_v30_052_r03_mini8_local.launcher.log)
- [ablation_v30_052_r03_mini8_local.validate.log](../../results/phase2/ablation_v30_052_r03_mini8_local.validate.log)
- [ablation_v30_052_r04_mini8_rpc_scored.json](../../results/phase2/ablation_v30_052_r04_mini8_rpc_scored.json)

### v30.053 H23 old-source basis按K21收缩能源选择

before：old basis按sum leg*leg^T Gram

after：同rank和age按legal current K21-contraction proxy选择basis

状态：partial_derivation

诊断结论：K21PSD operator checked; livestate/cost/score pending；算法分数前后不适用。

- [ablation_v30_053_k21_psd_sketch.json](../../results/phase2/ablation_v30_053_k21_psd_sketch.json)

### v30.054 H24 old-source basis按下一层D3敏感方向选择

before：腿Gram不区分mean敏感方向

after：用next W和legal w(3,1)proxy reweight相同rank basis

状态：partial_derivation

诊断结论：Fixed downstream gate-weighted PSD residual objective contract; lawful gate estimate/full-state/candidate/cost pending；算法分数前后不适用。

- [ablation_h22_h24_sketch_objective.json](../../results/phase2/ablation_h22_h24_sketch_objective.json)
- [ablation_v30_054_r02_output_metric_mapping.json](../../results/phase2/ablation_v30_054_r02_output_metric_mapping.json)
- [ablation_v30_054_d3_mean_coefficient.json](../../results/phase2/ablation_v30_054_d3_mean_coefficient.json)

### v30.055 H25 Rres新增解析可分leading项

before：Rres=S21-S_sep统一rank16近似

after：从已知生产表提取一个准确第二leading separable项；余项仍rank16

状态：partial_derivation

诊断结论：TwoD3column separableterms exact; fullproductionrepeatcoverage pending；算法分数前后不适用。

- [ablation_v30_055_second_separable_term.json](../../results/phase2/ablation_v30_055_second_separable_term.json)

### v30.056 H26 Rres物理机制分组basis

before：Gaussian/K3/K4影响混入单一Rres后QR

after：固定总rank16，对三类已明确分解来源形成统一合成basis

状态：partial_derivation

诊断结论：GroupedPSD fixedtotalrank contract; actualRresmechanismdecomposition pending；算法分数前后不适用。

- [ablation_v30_056_grouped_basis_contract.json](../../results/phase2/ablation_v30_056_grouped_basis_contract.json)

### v30.057 H27 K3 newborn重复内容一致性oracle

before：birth rep3/rep21从保留hub dslices定义，影子来源移入M

after：独立逐类检查full mixed和repeated内容账是否一致，只有发现具体错配才修复

状态：completed_diagnostic

诊断结论：Arbitrary identity-born hub and oldCP Wick full/repeated content exact on three smallshapes; no bookkeeping mismatch; not full truncated production accuracy；算法分数前后不适用。

- [ablation_v30_057_bookkeeping.json](../../results/phase2/ablation_v30_057_bookkeeping.json)
- [source_birth_combination_algebra.json](../../results/phase2/source_birth_combination_algebra.json)

### v30.058 H28 再生core对角和offdiag联合对象闭合

before：diag由K4/K22，offdiag由lambda*C分别给出

after：从同一个可计算selected production object产生两部分

状态：partial_derivation

诊断结论：Jointcore transport contract; selectednonlinearobject/candidate pending；算法分数前后不适用。

- [ablation_v30_058_joint_core_closure.json](../../results/phase2/ablation_v30_058_joint_core_closure.json)

### v30.059 H29 K111乘covariance边新来源

before：当前factored source过滤或剪枝缺部分mixed K3+edge diagram

after：恢复一个明确vector partition轨道的mixed新born source

状态：partial_derivation

诊断结论：Rank1edge*CP algebra and distinctmask verified; live dominantmode, full diagram Wick/coef/coverage and candidatecost pending；算法分数前后不适用。

- [ablation_v30_075_rank1_edge_contract.json](../../results/phase2/ablation_v30_075_rank1_edge_contract.json)
- [ablation_v30_059_060_edge_transport_repeats.json](../../results/phase2/ablation_v30_059_060_edge_transport_repeats.json)

### v30.060 H30 K111乘covariance边一层收缩

before：同H29 mixed来源缺失

after：仅追踪H29 newborn下一层K21/variance收缩，一层后删除

状态：partial_derivation

诊断结论：FiniteCP one/twosteptransport contract; actualdiagram/repeat/eviction pending；算法分数前后不适用。

- [ablation_h13_h14_h30_lifetime_contract.json](../../results/phase2/ablation_h13_h14_h30_lifetime_contract.json)
- [ablation_v30_059_060_edge_transport_repeats.json](../../results/phase2/ablation_v30_059_060_edge_transport_repeats.json)

### v30.061 H31 exact covariance triangle无偏收缩

before：trianglelast用rank16 approximate C

after：用原始Coff的lawful random estimator估计exact alloutput triangle contraction

状态：partial_derivation

诊断结论：Unbiased expectation and exact Rademacher variance law checked; actual network signal/cost/variance candidate gate pending；算法分数前后不适用。

- [ablation_v30_061_triangle_unbiased.json](../../results/phase2/ablation_v30_061_triangle_unbiased.json)
- [ablation_v30_061_triangle_variance_formula.json](../../results/phase2/ablation_v30_061_triangle_variance_formula.json)
- [ablation_v30_061_probe_cost_gate.json](../../results/phase2/ablation_v30_061_probe_cost_gate.json)

### v30.062 H32 高阶newborn直接CP因素生成

before：fully mixed K4/highorder未保留

after：直接从current state和Gaussian diagram生成固定数量CP factors供historical transport

状态：partial_derivation

诊断结论：DirectCP tree factors exact but nR² rankgrowth; fixedboundedrank/state/cost candidate gate pending；算法分数前后不适用。

- [ablation_v30_062_direct_cp_birth.json](../../results/phase2/ablation_v30_062_direct_cp_birth.json)
- [ablation_v30_062_cp_cost_gate.json](../../results/phase2/ablation_v30_062_cp_cost_gate.json)

### v30.063 O01 官方K31截断生产逐项一致性

before：仅有K31低阶Gaussian代理及精确moment oracle

after：独立核验完整官方K31基础和augmented表

状态：completed_diagnostic

诊断结论：Independent bounded vector multiset enumeration vs official canonical diagram generator; base/augmented full/filter separated,24 smallshape comparisons maxerror1.0658e-14; excludes Torch factor kernel；算法分数前后不适用。

- [ablation_v30_063_064_production_parity.json](../../results/phase2/ablation_v30_063_064_production_parity.json)

### v30.064 O02 官方capped K211逐项生产一致性

before：仅有source contraction和power转换独立probe

after：完整capped diagram生产与显式small tensor一致

状态：completed_diagnostic

诊断结论：Independent bounded vector multiset enumeration vs official canonical diagram generator; base/augmented full/filter separated,24 smallshape comparisons maxerror1.0658e-14; excludes Torch factor kernel；算法分数前后不适用。

- [ablation_v30_063_064_production_parity.json](../../results/phase2/ablation_v30_063_064_production_parity.json)

### v30.065 O03 K211完整来源抵消一致性

before：组合部件分别通过

after：一次完整非线性核心生产逐层对照

状态：partial_diagnostic

诊断结论：Completeold/birth/pK211/meanK111/CovCov joint finite-distributionalgebra exact; official matchedbudgetproduction parity pending；算法分数前后不适用。

- [ablation_v30_065_joint_source_conversion.json](../../results/phase2/ablation_v30_065_joint_source_conversion.json)

### v30.066 O04 旧lean与当前官方generator语义归因

before：仅完成pair-output False/True diff

after：完整source/output的exact factor inventory

状态：partial_diagnostic

诊断结论：126 bounded signatures; V29 coefficient mapping pending；算法分数前后不适用。

- [ablation_v30_066_full_output_inventory.json](../../results/phase2/ablation_v30_066_full_output_inventory.json)
- [ablation_v30_066_wick_gap_map.json](../../results/phase2/ablation_v30_066_wick_gap_map.json)

### v30.067 O05 完整K31/K211生产核的离线表示上限

before：Goff=layer lambda*Coff

after：完整production Gnew与regen G成对诊断

状态：partial_diagnostic

诊断结论：Synthetic full cumulant diag+oraclelambda representation bound/spectrum measured; actual official truncated production benchmark state pending；算法分数前后不适用。

- [ablation_v30_067_core_representation_bound.json](../../results/phase2/ablation_v30_067_core_representation_bound.json)

### v30.068 O06 完整生产核单层替换诊断

before：全部层lambda核

after：预先指定单层完整production核

状态：partial_diagnostic

诊断结论：SyntheticUSE/FEED path split only; fullactualproduction and replacement pending；算法分数前后不适用。

- [ablation_v30_068_071_kernel_path_split.json](../../results/phase2/ablation_v30_068_071_kernel_path_split.json)

### v30.069 O07 完整核对角修正非零性与候选

before：gprev由K4+K22 rowsum

after：加入由完整trace投影证明非零的对角修正

状态：partial_diagnostic

诊断结论：SyntheticUSE/FEED path split only; fullactualproduction and replacement pending；算法分数前后不适用。

- [ablation_v30_068_071_kernel_path_split.json](../../results/phase2/ablation_v30_068_071_kernel_path_split.json)

### v30.070 O08 完整production核仅USE替换

before：USE wk431=lambda*Cpreoff

after：USE wk431=完整production核运输结果

状态：partial_diagnostic

诊断结论：SyntheticUSE/FEED path split only; fullactualproduction and replacement pending；算法分数前后不适用。

- [ablation_v30_068_071_kernel_path_split.json](../../results/phase2/ablation_v30_068_071_kernel_path_split.json)

### v30.071 O09 完整production核仅FEED替换

before：FEED X3使用regen diagonal+lambda

after：FEED X3使用完整production核

状态：partial_diagnostic

诊断结论：SyntheticUSE/FEED path split only; fullactualproduction and replacement pending；算法分数前后不适用。

- [ablation_v30_068_071_kernel_path_split.json](../../results/phase2/ablation_v30_068_071_kernel_path_split.json)

### v30.072 O10 完整核剩余误差的state模态诊断

before：单lambda mode

after：增加预先定义state模态的离线解释力测量

状态：partial_diagnostic

诊断结论：Synthetic core residual spectrum50/90/99pct ranks recorded; actual official benchmark state prerequisite pending；算法分数前后不适用。

- [ablation_v30_072_synthetic_core_modes.json](../../results/phase2/ablation_v30_072_synthetic_core_modes.json)

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

### v30.075 O13 K111乘edge covariance主模态代理

before：该mixed diagram遗漏

after：使用live covariance主模态近似edge并保留K111来源

状态：partial_derivation

诊断结论：Rank1edge*CP algebra and distinctmask verified; live dominantmode, full diagram Wick/coef/coverage and candidatecost pending；算法分数前后不适用。

- [ablation_v30_075_rank1_edge_contract.json](../../results/phase2/ablation_v30_075_rank1_edge_contract.json)
- [ablation_v30_075_076_actual_first_layer_modes.json](../../results/phase2/ablation_v30_075_076_actual_first_layer_modes.json)

### v30.076 O14 K111乘edge代理误差门槛

before：proxy仅通过代数

after：proxy与exact收缩成对误差评估

状态：partial_derivation

诊断结论：Fixedproxy error bound checked; actualnetwork dominantmode/source loss threshold pending；算法分数前后不适用。

- [ablation_v30_076_edge_proxy_error_bound.json](../../results/phase2/ablation_v30_076_edge_proxy_error_bound.json)
- [ablation_v30_075_076_actual_first_layer_modes.json](../../results/phase2/ablation_v30_075_076_actual_first_layer_modes.json)

### v30.077 O15 doubled-edge早期来源贡献门槛

before：已否证last-birth三matmul候选

after：有界诊断早期birth跨层贡献

状态：partial_diagnostic

诊断结论：SyntheticearlybirthD3signal only; actualnetworkcost/gain gate unmet；算法分数前后不适用。

- [ablation_v30_077_doubled_edge_lifetime.json](../../results/phase2/ablation_v30_077_doubled_edge_lifetime.json)

### v30.078 O16 covariance triangle exact多层small诊断

before：仅低秩last-only阴性证据

after：exact small triangle跨层传播

状态：partial_diagnostic

诊断结论：Exact distinct synthetic triangle three-layer transport; real benchmark signal/conversion/cost gate pending；算法分数前后不适用。

- [ablation_v30_078_exact_triangle_transport.json](../../results/phase2/ablation_v30_078_exact_triangle_transport.json)

### v30.079 O17 当前官方WK22 path与旧lean等价审计

before：旧复合X/Y生产

after：逐diagram核对current official WK22路径

状态：completed_diagnostic

诊断结论：Official budget3 and composite budget4 WK22 mechanisms differ; separate candidate needed；算法分数前后不适用。

- [ablation_v30_079_wk22_budget_equivalence.json](../../results/phase2/ablation_v30_079_wk22_budget_equivalence.json)

### v30.080 O18 现有birth薄反馈乘积预算审计

before：复合因子乘积未完整power-count

after：每个出生乘积标明预算与重复补偿

状态：partial_diagnostic

诊断结论：All16 lean B1/B2 products power-counted; feedback*feedback cost4 exceedsaug3. Full V29 position/coef/isomorphism attribution and repeated compensation pending；算法分数前后不适用。

- [ablation_v30_080_birth_power_inventory.json](../../results/phase2/ablation_v30_080_birth_power_inventory.json)

### v30.081 O19 K5边缘生产闭合

before：无独立K5状态

after：经验证的选择性marginal K5生产

状态：partial_derivation

诊断结论：ExactGaussianReLUdegree5/6 marginal moment recurrence/quadrature verified; officialaugmented propagatedK5 closure still pending；算法分数前后不适用。

- [ablation_degree5_6_marginal_derivation.json](../../results/phase2/ablation_degree5_6_marginal_derivation.json)
- [ablation_v30_081_k5_gaussian_marginal.json](../../results/phase2/ablation_v30_081_k5_gaussian_marginal.json)

### v30.082 O20 K5独立边缘readout

before：PK1无kappa5线性项

after：PK1加入合法K5 marginal readout

状态：partial_derivation

诊断结论：Isolated linearcumulant ReLU coefficient derived; validfullstateproduction/chargedcandidate/score pending；算法分数前后不适用。

- [ablation_v30_082_k5_readout_contract.json](../../results/phase2/ablation_v30_082_k5_readout_contract.json)

### v30.083 O21 K6纯径向投影生产公式

before：无K6channel

after：完整可审计radial scalar生产定义

状态：partial_derivation

诊断结论：Radial6 average15pairings/tripletrace and actualWmetric transport exact; full production/candidate/cost pending；算法分数前后不适用。

- [ablation_v30_083_084_radial_k6_contract.json](../../results/phase2/ablation_v30_083_084_radial_k6_contract.json)
- [ablation_v30_083_084_k6_production_marginals.json](../../results/phase2/ablation_v30_083_084_k6_production_marginals.json)

### v30.084 O22 K6径向状态合法线性运输

before：无K6运输

after：加入radial harmonic state W运输

状态：partial_derivation

诊断结论：Radial6 average15pairings/tripletrace and actualWmetric transport exact; full production/candidate/cost pending；算法分数前后不适用。

- [ablation_v30_083_084_radial_k6_contract.json](../../results/phase2/ablation_v30_083_084_radial_k6_contract.json)
- [ablation_v30_083_084_k6_production_marginals.json](../../results/phase2/ablation_v30_083_084_k6_production_marginals.json)

### v30.085 O23 K6独立边缘readout

before：PK1无kappa6线性项

after：合法K6对角读入PK1

状态：partial_derivation

诊断结论：Isolated linearcumulant ReLU coefficient derived; validfullstateproduction/chargedcandidate/score pending；算法分数前后不适用。

- [ablation_v30_085_k6_readout_contract.json](../../results/phase2/ablation_v30_085_k6_readout_contract.json)

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

### v30.088 O26 图表收益的上游调参分割审计

before：候选网络分割标签不够细

after：逐机制报告训练/selection/confirmation来源矩阵

状态：partial_diagnostic

诊断结论：Frozenupstream README split/calibration disclosures captured; exactauthortrainIDs unknown, L21sealed；算法分数前后不适用。

- [ablation_v30_088_upstream_split_audit.json](../../results/phase2/ablation_v30_088_upstream_split_audit.json)

### v30.089 O27 封存新分片的diagslice状态分布偏移

before：只有最终分数分割

after：新分片state-only机制分布诊断

状态：partial_diagnostic

诊断结论：Development lawful feature quantiles fixed without labels; reserved-shard state collection pending；算法分数前后不适用。

- [ablation_v30_089_frozen_shift_features.json](../../results/phase2/ablation_v30_089_frozen_shift_features.json)

### v30.090 O28 新seed网络的diagram规模law核验

before：只有fixed1024表规模预期

after：预定width/seed上的state diagram rms law

状态：partial_diagnostic

诊断结论：First-layer fixed3width/3seed tree and edge scale measured; deep capped211 and 1024 validation pending；算法分数前后不适用。

- [ablation_v30_090_diagram_width_law.json](../../results/phase2/ablation_v30_090_diagram_width_law.json)

### v30.091 O29 高阶校正信号与reference不确定性的分离

before：gain与noise proxy混用

after：paired候选信号和reference误差分离账本

状态：partial_diagnostic

诊断结论：Candidate delta prediction RMS measured without labels; paired independent strong references still absent, noise separation unfulfilled；算法分数前后不适用。

- [ablation_v30_091_h22_prediction_signal.json](../../results/phase2/ablation_v30_091_h22_prediction_signal.json)

### v30.092 O30 新增图表array操作计量路径合同

before：只运行small数值公式

after：actual fnp路径日志与成本逐算子核验

状态：partial_diagnostic

诊断结论：Actual local fnp float32/64 out/copy/alias contracts checked; complete diagram and remote coverage pending；算法分数前后不适用。

- [ablation_v30_092_new_operator_contract.json](../../results/phase2/ablation_v30_092_new_operator_contract.json)

### v30.093 O31 完整K31生产对称化与轴方向合同

before：K31方向未actual生产确认

after：actual fnp K31输出与固定axis枚举

状态：completed_diagnostic

诊断结论：Row3/col1 asymmetry and distinct-index cap tested on n2,3,4,7,8; no transpose averaging; diagonal output explicit; generator production separately validated；算法分数前后不适用。

- [ablation_v30_093_094_production_contracts.json](../../results/phase2/ablation_v30_093_094_production_contracts.json)
- [ablation_v30_063_064_production_parity.json](../../results/phase2/ablation_v30_063_064_production_parity.json)

### v30.094 O32 capped211 diagonal mask因式表示合同

before：mask分布在多个函数

after：mask逐阶段追踪与explicit枚举一致

状态：completed_diagnostic

诊断结论：Row3/col1 asymmetry and distinct-index cap tested on n2,3,4,7,8; no transpose averaging; diagonal output explicit; generator production separately validated；算法分数前后不适用。

- [ablation_v30_093_094_production_contracts.json](../../results/phase2/ablation_v30_093_094_production_contracts.json)
- [ablation_v30_063_064_production_parity.json](../../results/phase2/ablation_v30_063_064_production_parity.json)

### v30.095 O33 complete harmonic core actualW transport对照

before：transport代数与production分离

after：完整生成G的actualW数值parity

状态：completed_diagnostic

诊断结论：Six actualW fourth-order tensor transport versus G/M transport checks maxerror1.776e-15; marginals and trace checked; birth metric and transported metric differ；算法分数前后不适用。

- [ablation_v30_095_actual_transport.json](../../results/phase2/ablation_v30_095_actual_transport.json)

### v30.096 L01 冻结原轨迹特征缓存

before：只有 aggregate 与逐层 MSE；没有可拟合的逐神经元内部特征缓存。

after：新建 V29 诊断副本，保存原 13 特征、原预测和网络 ID；不改变预测公式。

状态：partial_diagnostic

诊断结论：All8 networks original13 features/predictions cached; per-network charged FLOPs equal frozen; same-run untraced numerical paired control pending; tracing timing excluded；算法分数前后不适用。

- [ablation_v30_096_features_manifest.json](../../results/phase2/ablation_v30_096_features_manifest.json)
- [ablation_v30_096_cache_pairing.json](../../results/phase2/ablation_v30_096_cache_pairing.json)
- [ablation_v30_096_r02_untraced_control.json](../../results/phase2/ablation_v30_096_r02_untraced_control.json)

### v30.097 L02 冻结网络级训练选择流程

before：Mini8同时用于既有快筛；缺少修正器专用网络级选择协议。

after：IDs0–7用leave-one-network-out；IDs8–15保留为一次公共确认集；另列新种子最终独立集。

状态：partial_diagnostic

诊断结论：Network-level leave-one-network-out selection frozen without reading confirmation targets; feature cache prerequisite L01 pending；算法分数前后不适用。

- [ablation_v30_097_selection_protocol.json](../../results/phase2/ablation_v30_097_selection_protocol.json)

### v30.098 L03 最后层13特征残差线性修正

before：原V29固定CORR_BETA后仍有final残差。

after：只在final输出追加一个开发网络拟合的13特征线性残差预测；早层与原Beta不变。

状态：partial_negative

诊断结论：OfflineCV negative; charged estimator pending；算法分数前后不适用。

- [ablation_learning_loocv_r01.json](../../results/phase2/ablation_learning_loocv_r01.json)
- [ablation_v30_098_charged_readout_component.json](../../results/phase2/ablation_v30_098_charged_readout_component.json)

### v30.099 L04 删除常数特征的修正对照

before：L03含截距特征，可能只学习整体偏置。

after：只删除ones列，保留其余12列及同样训练流程。

状态：partial_negative

诊断结论：OfflineCV negative; charged estimator pending；算法分数前后不适用。

- [ablation_learning_loocv_r01.json](../../results/phase2/ablation_learning_loocv_r01.json)
- [ablation_learning_charged_components_r02.json](../../results/phase2/ablation_learning_charged_components_r02.json)

### v30.100 L05 阈值形状特征增量

before：L03仅含alpha及abs(alpha)。

after：追加alpha²phi、alpha³phi、alpha⁴phi三个确定性特征。

状态：partial_negative

诊断结论：OfflineCV negative; charged estimator pending；算法分数前后不适用。

- [ablation_learning_loocv_r01.json](../../results/phase2/ablation_learning_loocv_r01.json)
- [ablation_learning_charged_components_r02.json](../../results/phase2/ablation_learning_charged_components_r02.json)

### v30.101 L06 标准化偏度特征增量

before：L03直接用D3，未表达相对于方差的规模。

after：追加D3/(sigma³)与该值乘phi的两列，分母仅使用原正sigma。

状态：partial_negative

诊断结论：OfflineCV negative; charged estimator pending；算法分数前后不适用。

- [ablation_learning_loocv_r01.json](../../results/phase2/ablation_learning_loocv_r01.json)
- [ablation_learning_charged_components_r02.json](../../results/phase2/ablation_learning_charged_components_r02.json)

### v30.102 L07 标准化峰度特征增量

before：L03含全层K4_sigma，但没有每神经元g4row。

after：追加g4row/sigma⁴及乘phi的两列。

状态：partial_negative

诊断结论：Realg4row two features samecycleLOO worsensrelativeL03 +0.04419%; multithreadoffline diagnostic only, fixedthreadcandidate unverified；算法分数前后不适用。

- [ablation_v30_102_g4_features.json](../../results/phase2/ablation_v30_102_g4_features.json)
- [ablation_v30_102_g4_cv.json](../../results/phase2/ablation_v30_102_g4_cv.json)

### v30.103 L08 最后线性均值传播轨迹特征

before：L03仅用最后pre的本层13特征。

after：追加W_final乘上一层预测均值误差代理向量与本层mu的差异特征。

状态：partial_negative

诊断结论：NetworkCV predefined extra features negative; chargedcandidate pending；算法分数前后不适用。

- [ablation_learning_extra_r01.json](../../results/phase2/ablation_learning_extra_r01.json)
- [ablation_v30_103_zero_trajectory_contract.json](../../results/phase2/ablation_v30_103_zero_trajectory_contract.json)

### v30.104 L09 前层方差轨迹特征

before：L03缺少方差的来源分解。

after：追加(W²)@var_prev与var−(W²)@var_prev两列。

状态：partial_negative

诊断结论：NetworkCV predefined extra features negative; chargedcandidate pending；算法分数前后不适用。

- [ablation_learning_extra_r01.json](../../results/phase2/ablation_learning_extra_r01.json)

### v30.105 L10 跨神经元均值残差特征

before：L03每神经元只看局部特征。

after：追加final层mean(mu),mean(var),mean(D3)三个广播全层状态。

状态：partial_negative

诊断结论：OfflineCV negative; charged estimator pending；算法分数前后不适用。

- [ablation_learning_loocv_r01.json](../../results/phase2/ablation_learning_loocv_r01.json)
- [ablation_learning_charged_components_r02.json](../../results/phase2/ablation_learning_charged_components_r02.json)

### v30.106 L11 输出幅度归一化回归

before：L03在绝对均值误差上拟合。

after：训练目标改成残差/sigma，推理修正乘sigma；特征集保持13列。

状态：partial_negative

诊断结论：OfflineCV negative; charged estimator pending；算法分数前后不适用。

- [ablation_learning_loocv_r01.json](../../results/phase2/ablation_learning_loocv_r01.json)
- [ablation_learning_charged_components_r02.json](../../results/phase2/ablation_learning_charged_components_r02.json)

### v30.107 L12 网络等权训练

before：L03将训练网所有神经元等权拼接。

after：每网络归一化为同等总权重，再拟合原残差。

状态：partial_negative

诊断结论：OfflineCV negative; charged estimator pending；算法分数前后不适用。

- [ablation_learning_loocv_r01.json](../../results/phase2/ablation_learning_loocv_r01.json)
- [ablation_learning_charged_components_r02.json](../../results/phase2/ablation_learning_charged_components_r02.json)

### v30.108 L13 逐层共享线性修正

before：L03只训练final输出残差。

after：对全部层训练一个共享13特征系数，但仍只用于final输出。

状态：partial_negative

诊断结论：OfflineCV negative; charged estimator pending；算法分数前后不适用。

- [ablation_learning_loocv_r01.json](../../results/phase2/ablation_learning_loocv_r01.json)
- [ablation_learning_charged_components_r02.json](../../results/phase2/ablation_learning_charged_components_r02.json)

### v30.109 L14 分段线性阈值修正

before：L03一个全域线性模型。

after：按固定alpha区间(-∞,-1),[-1,0),[0,1),[1,∞)训练四个线性头，final分段使用。

状态：partial_negative

诊断结论：Continuous four-head LOO raw +0.5931%; charged candidate and parent gain gate unmet；算法分数前后不适用。

- [ablation_v30_109_continuous_heads_cv.json](../../results/phase2/ablation_v30_109_continuous_heads_cv.json)

### v30.110 L15 小网络非线性修正

before：L03固定13列线性预测残差。

after：一个固定13→8→1 tanh小网络预测final残差。

状态：partial_negative

诊断结论：Frozen13-8-1 networkLOO raw+1.91998%, all121gradient finite-difference contract passed; chargedcandidate and independent confirmation not passed；算法分数前后不适用。

- [ablation_v30_110_gradient_contract.json](../../results/phase2/ablation_v30_110_gradient_contract.json)

### v30.111 L16 单前层修正递推注入

before：验证的修正仅改变final输出，不改变内部状态。

after：仅第8层将已验证修正写入mu，后续原V29递推。

状态：partial_derivation

诊断结论：87 central moment checks; translation preserves higher cumulants, mean-only reestimation needs explicit distribution model；算法分数前后不适用。

- [ablation_v30_111_mean_shift_contract.json](../../results/phase2/ablation_v30_111_mean_shift_contract.json)

### v30.113 L18 残差条件可预测性上界

before：只知道偏置oracle解释约0.46% MSE。

after：用开发网络级CV测13特征解释方差，给出达到10倍需约90% MSE减少的差距。

状态：partial_diagnostic

诊断结论：Held-network13feature R2 negative;10x requires0.9. Charged adjusted candidate pending；算法分数前后不适用。

- [ablation_v30_113_predictability.json](../../results/phase2/ablation_v30_113_predictability.json)

### v30.114 L19 独立参考两次采样一致性

before：官方十亿样本reference只有单份，条件噪声proxy不能当误差地板。

after：在新生成少量确认网络上制作两份互不重叠输入种子的独立reference。

状态：partial_derivation

诊断结论：CPU/reference sample uncertainty budget planned; actual two-reference sampling pending；算法分数前后不适用。

- [ablation_v30_114_reference_budget_plan.json](../../results/phase2/ablation_v30_114_reference_budget_plan.json)
- [ablation_v30_114_reference_cpu_pilot.json](../../results/phase2/ablation_v30_114_reference_cpu_pilot.json)
- [ablation_v30_114_reference_uncertainty_gate.json](../../results/phase2/ablation_v30_114_reference_uncertainty_gate.json)

### v30.115 L20 新种子网络最终独立确认

before：公共IDs8–15仅是项目内保留集，上游是否使用未知。

after：候选及配置冻结后生成预注册未见种子网络，候选与上一冻结基线成对评分。

状态：partial_diagnostic

诊断结论：Newweight/grader/reference streams and officialHe generation protocol frozen; strongreferencegeneration and independentpairedscore pending；算法分数前后不适用。

- [ablation_v30_115_independent_protocol.json](../../results/phase2/ablation_v30_115_independent_protocol.json)
- [ablation_v30_115_independent_weights.json](../../results/phase2/ablation_v30_115_independent_weights.json)
- [ablation_v30_114_reference_uncertainty_gate.json](../../results/phase2/ablation_v30_114_reference_uncertainty_gate.json)

### v30.117 L22 输出误差与阈值位置分解

before：MSE按网络/层汇总，无法定位哪些神经元贡献。

after：固定alpha分箱报告最后误差均值、方差、MSE与样本量。

状态：completed_diagnostic

诊断结论：Fixedalpha/trainonlyskew bins error statistics; localization only；算法分数前后不适用。

- [ablation_v30_117_118_error_bins.json](../../results/phase2/ablation_v30_117_118_error_bins.json)

### v30.118 L23 输出误差与偏度分解

before：未区分高偏度和低偏度神经元误差。

after：按固定标准化D3绝对值分位边界，用训练数据定义边界后测逐网MSE。

状态：completed_diagnostic

诊断结论：Fixedalpha/trainonlyskew bins error statistics; localization only；算法分数前后不适用。

- [ablation_v30_117_118_error_bins.json](../../results/phase2/ablation_v30_117_118_error_bins.json)

### v30.119 L24 误差沿层积累的增量分解

before：只有逐层MSE，不区分运输误差和局部新误差。

after：计算e_l−Phi_l⊙(W_l@e_{l−1})的离线一阶残差。

状态：completed_diagnostic

诊断结论：120 network/layer first-order residual transport decomposition; reference oracle only; nonlinear/reference noise caveat retained；算法分数前后不适用。

- [ablation_v30_119_error_transport.json](../../results/phase2/ablation_v30_119_error_transport.json)

### v30.120 L25 输出敏感性来源排序诊断

before：AGE_OLD按年龄统一压缩；不知道每来源最终影响。

after：用合法末端mean Jacobian代理估计来源贡献并排序，不看reference。

状态：partial_derivation

诊断结论：Mean adjoint and CPsource exactD3 contraction checked; full hub/repeat/covariance sensitivity and ranking still pending；算法分数前后不适用。

- [ablation_v30_120_mean_adjoint_contract.json](../../results/phase2/ablation_v30_120_mean_adjoint_contract.json)
- [ablation_v30_120_source_d3_contract.json](../../results/phase2/ablation_v30_120_source_d3_contract.json)
- [ablation_v30_120_coupled_gaussian_jacobian.json](../../results/phase2/ablation_v30_120_coupled_gaussian_jacobian.json)

### v30.121 L26 输出敏感性来源预算分配

before：每个老来源相同年龄阈值与固定共享秩。

after：总预算不变，按照已验证L25代理把高影响来源保留dense，其余更早压缩。

状态：partial_derivation

诊断结论：Fixed2slot surrogate rule exact; livesourcefeedback/eviction/score pending；算法分数前后不适用。

- [ablation_v30_121_output_budget_rule.json](../../results/phase2/ablation_v30_121_output_budget_rule.json)

### v30.122 L27 来源删除负控制

before：L25代理有排序，但没有相同成本的随机删除对照。

after：固定输入权重确定性hash选一来源删除，保留其余机制。

状态：partial_derivation

诊断结论：WholeCPsource and repeat deletion consistent; livecompressedstate/score pending；算法分数前后不适用。

- [ablation_v30_122_source_deletion_contract.json](../../results/phase2/ablation_v30_122_source_deletion_contract.json)
- [ablation_v30_122_projected_eviction_contract.json](../../results/phase2/ablation_v30_122_projected_eviction_contract.json)
- [ablation_v30_122_basis_reselection_repeats.json](../../results/phase2/ablation_v30_122_basis_reselection_repeats.json)

### v30.123 L28 末端均值控制变量相关性

before：未确认廉价样本支路能解释V29残差。

after：在少量开发网络离线测合法低阶代理与真实输出样本的残差相关性。

状态：partial_derivation

诊断结论：Fixed train/held control-variate variance/cost gate derived; actualknown-expectation networkproxy and pairedsignals pending；算法分数前后不适用。

- [ablation_v30_122_control_variate_budget.json](../../results/phase2/ablation_v30_122_control_variate_budget.json)

### v30.124 L29 条件高斯最后层积分对照

before：最后层均值依赖Hermite边缘展开。

after：对末层pre均值/方差/偏度构造规范化一维积分诊断，和截断展开比较。

状态：partial_negative

诊断结论：Positive normalized moment-matched skew-normal finalreadout worsens all8raw networks ~52%; no chargedcandidate；算法分数前后不适用。

- [ablation_v30_123_positive_skew_family.json](../../results/phase2/ablation_v30_123_positive_skew_family.json)
- [ablation_v30_123_positive_family_network_screen.json](../../results/phase2/ablation_v30_123_positive_family_network_screen.json)

### v30.125 L30 末层尾概率数值稳定性

before：极端alpha路径只由原正态cdf/pdf与float32演算处理。

after：保持原公式，新增离线极端alpha及sigma小值数值对照。

状态：completed_diagnostic

诊断结论：32 alpha/sigma combinations finite and monotonic; max relative error1.1498e-5 against70digit independent integral; scalar Gaussian mean path only；算法分数前后不适用。

- [ablation_v30_125_relu_tail.json](../../results/phase2/ablation_v30_125_relu_tail.json)

### v30.126 L32 跨网络异质收益检查

before：候选只有网络平均分数改善。

after：在同一固定套件报告胜率、每网ratio、最差退化及paired bootstrap。

状态：completed_diagnostic

诊断结论：Per-network gains/losses reported for all8 offline readout screens; no independent confirmation；算法分数前后不适用。

- [ablation_v30_126_heterogeneous_gains.json](../../results/phase2/ablation_v30_126_heterogeneous_gains.json)

### v30.127 L33 参考dtype贡献诊断

before：官方targets以原float32使用，误差可能受量化贡献。

after：离线估计原targets float32 ULP尺度与MSE阈值关系，保留原targets不重评分。

状态：completed_diagnostic

诊断结论：Mini8 float32 uniform rounding proxy 6.120378688436097e-16; sampling uncertainty not estimated；算法分数前后不适用。

- [ablation_v30_127_reference_dtype.json](../../results/phase2/ablation_v30_127_reference_dtype.json)

### v30.128 L34 标准化计算的候选合同

before：拟合端与fnp推理端标准化可能使用不同精度或特征顺序。

after：冻结模型文件的feature_order、mean、scale、coef哈希并在小样本对照fnp预测。

状态：partial_diagnostic

诊断结论：Frozen fold standardization legal fnp small inference; end-to-end model gate pending；算法分数前后不适用。

- [ablation_v30_128_standardization_contract.json](../../results/phase2/ablation_v30_128_standardization_contract.json)

### v30.129 L35 多次选择偏差账本

before：长消融列表容易反复查看同一确认集而失去独立性。

after：每次拟合/调参/查看reference都记录selection ledger，确认集首次查看后降为开发集。

状态：partial_diagnostic

诊断结论：Prequeue benchmarks and identifiable offlineCV runs indexed; hypothesis/effective-selection count and final confirmation pending；算法分数前后不适用。

- [ablation_v30_129_selection_ledger.json](../../results/phase2/ablation_v30_129_selection_ledger.json)
- [ablation_v30_129_r02_comprehensive_selection_ledger.json](../../results/phase2/ablation_v30_129_r02_comprehensive_selection_ledger.json)


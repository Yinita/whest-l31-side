# M01 r01：论文到代码的首轮映射

状态：partial_mapping；不表示完整算法复现或任何分数提升。

一手来源：https://arxiv.org/html/2605.05179v2 第4、5、6节，S.4.1–S.4.3。官方本地冻结commit：93d091a4c26c042bfffa28f2e76a81bc0aba94bb。工具当前0.12.1/0.16.1不变。

| 论文对象 | 官方代码入口 | 本地实现/证据 | 仍缺什么 |
|---|---|---|---|
| Basic cumulant propagation，4.1/S.4.1 | kprop_harmonic.py:get_all_terms_iso, nonlin_kprop, mlp_kprop；kind=SIMPLE/factor=False | H06历史90项展开；official_augmented_terms_audit.json；O04 full_output_inventory | 完整状态与各阶端到端对照；90项补齐并不等于完整basic |
| Augmented，4.2/S.4.2 | get_vec_cond预算提升；kind=AUGMENT/factor=False | H/O新增term库存和固定来源单项实验 | 全部高阶状态及完整历史；固定birth负结果不否决完整方法 |
| Factorized，4.3/S.4.3 | factor_k3.py:FactoredTensor/factored_nonlin_kprop_k3；factor_k4.py；factor=True | V29共享hub、LA/LP收缩、薄反馈、年龄分层；official_factored111_v29_audit.json | V29压缩、拟合memoryless K4和source省略与论文方法的完整误差拆分 |
| Factorized augmented | kprop_harmonic.py:factored_keeps_term及nonlin_kprop警告 | O04分类和官方augmented差分库存 | **不等同dense augmented**；nonfactorable diagrams被丢弃，不能用相同阶数宣称恒等 |
| Power cumulants和奇数阶K+1 trace | S.3.3/S.4、6.4；harmonic/metric处理 | O/H birth和power inventories；H17Gaussian cappedcore仅一子机制 | 完整trace和harmonic对应证明；K5边缘修正非完整K5 |
| 理论误差/成本 | 第5节及S.5 | O28有限宽度单层实验 | 固定深度、多项式激活条件；ReLU实测不构成定理完整验证 |

论文报告：basic成本上界c_K L n^(K+1)，factorized(K≥3)为c_K L² n^K；计量使用论文自己的对称核调整，不能直接移植为本比赛flopscope收费。论文经验实验宽度4–256、深度2–12、阶数1–4，不等于竞赛1024×16验证。论文指出宽256深8的K4可差于采样，需先成本门槛。

后续M01：冻结每个official入口参数、状态shape、输出项与本地完整对应；随后M02完整Gaussian闭合与M03小/中shape四种变体，竞赛候选只通过允许算子端到端计费。已有D02 covariance-only负结果/D04三通道Hermite尾项分别复用，不称完整Wright方法。

## M03 r11/r12：最早分叉与残差全秩消融（2026-10-13）

相同float32权重、64×4、两固定seed下，V29与官方SIMPLE/AUGMENT第一层均值逐元素完全一致；差距首次出现在第二层。SIMPLE第二层均值差MSE为9.614e-8/1.252e-7。这是预测间差异，不是真值误差或分数提升。

| 单因素版本 | seed | before raw MSE | after raw MSE | 变化 | charged FLOPs before → after |
|---|---|---:|---:|---:|---:|
| M03-r12，R_RES16→64，R_FB保持16 | 13026031 | 1.9517205e-5 | 1.8978465e-5 | −2.7603% | 28,541,708 → 44,239,628 |
| 同上 | 13026032 | 7.5005931e-6 | 7.9081352e-6 | +5.4335% | 同上 |

两次零异常，但全秩残差不能恢复论文逐层一致性，不扩大秩扫描。小形状使用旧共同MC reference，仅作诊断，不能转换为比赛adjusted或独立泛化声明。下一项是第二层preactivation协方差和K3对角收缩分解。

证据：`results/phase2/paper_m03_r11_layer_{v29,official,summary}.json`与`paper_m03_r12_resfull_{same64,summary}.json`；实现脚本`experiments/phase2/paper_m03_layer_parity.py`、`paper_m03_resfull_same64.py`。ruff和diff检查通过。

M03r13: Frozen two64x4 inputs; firstact covariance exact. Secondpre covariance MSE1.290e-15/1.323e-15 maxabs2.384e-7/2.980e-7 (roundoff). Secondpre K3diag MSE6.100e-7/9.513e-7 maxabs.002970/.004226; K321 MSE1.844e-7/2.123e-7. Ordinary/power/harmonic mapping required before defect attribution. Charged28,541,708 each equals baseline; extraction timing nondeployment. First official KeyError3 implicitGaussianzero retained; retry complete. No oracle, scoregain or submission. Next firstbirth tensor semantics. Evidence results/phase2/paper_m03_r13_state_{v29,official,summary}.json.

M03r14 independent frozen firstbirth ordinaryK3 contract: official factor_k3 step4 DS_pK_to_K converts power to ordinary; K3 returned FactoredTensor, no K3harmonic projection (K4 separate). NumPy float64 staged dense Kprime_ijk=sum_abc Wia Wjb Wkc Kabc; D3=Kprime_iii/D21=Kprime_iij offdiag. Both seeds densevs official max4.52e-8(diag)/5.00e-8(pair); densevsV29diagMSE6.100e-7/9.513e-7,pair1.844e-7/2.123e-7 persists. Thus returnedofficial contraction verified, pure pK/K naming insufficient to explain gap; V29birthrepresentation vs dslicearithmetic still unresolved. Offlineanalytic100,663,296multiplyaddFLOPs notcontestbilling. Frozen NPZ objects/hashes local, summary/sourcehash/formulas recorded in paper_m03_r14_birth_contract.json; no truth/score. Next independently reconstruct V29 firstbirth tensor and compare both dense contractions. Ruff passes.

M03r15 frozen firstbirth representation: T=Sym(3A,A diag(w2),I)+Sym(M,I,I), M=diag(s)+3A diag(e)+RLtranspose. Independentfloat64 dense transported D3/D21 agrees V29 within6.412e-8/7.710e-8(seed31),4.015e-8/9.792e-8(seed32); officialbirth disagreementMSE1.197e-8/1.160e-8,max.003572/.006607. Thus tested slice arithmetic matches represented tensor; prioritize birth approximation, no scoregain/blanket bugclaim. Chargedcapture28,541,708unchanged, offlineanalytictransport100,663,296notcontestbilling. Capture objects/inputs/sourcehashes frozen in r15birthcapture, formula/errors/scriptSHA in r15birthdense. Diskfull interrupted firstscriptwrite before execution; emptyfile repaired after removing only downloadable pipcache68.4MB, originaldata preserved. Remaining71MB requires capacity remedy before scaling. Next exact repeatedslice residual versus representedbirth.

M03r16 indexpartition: all-distinct249984entries birthmaxerror2.22e-9/2.64e-9; all-equal64entriesMSE2.062e-6/3.749e-6; exactlytwoequal12096entriesMSE2.485e-7/2.315e-7. Gaplocalized repeatedindices, not generic all-distincthub. No scoregain. R15 executedsource archived from ce91498, SHA matches historicalreport; later I→identity lint edits explain currenthashdifference, report unmodified. Capacity52MB→2.5GB via uvprune and ChromeHTTP/Code caches only, cleanupmanifestcache_cleanup_20261013.json; allbenchmarkdata/evidence/unknown tmp preserved. Next exact marginaldiagonal versus repeatedpair correction decomposition.

M03r17 frozenbirthdifference independently split/transported. Seed31 diagonly D3MSE5.841e-8,paironly6.244e-7,cross−7.283e-8,total6.100e-7; seed32diagonly1.651e-7,paironly7.810e-7,cross5.208e-9,total9.513e-7. D21paironlyMSE1.803e-7/1.878e-7 dominates diagonly1.047e-8/2.652e-8. All-distincttransport~1e-18MSE. Pair repeatedbirthpriority supported, no predictiongain. Analyticdense3splits301,989,888FLOPs notcompetitionbilling; frozenbaseline28,541,708charged. ExecutedsourceSHA in r17report; keptsourceunchanged to preserve binding, ruffoneunusedpreviousread(F841) recorded, no runtime failure. Disk1.6GBavailable,noexpandbatch. Next exact21residualrepresentation contract.

M03r18 frozen residual exactreplacement identifies firstbirthrank16approximation: ordinarybirthMSE1.197e-8/1.160e-8→1.594e-19/1.348e-19; pre1D3disagreement6.100e-7/9.513e-7→1.836e-16/1.577e-16. Replacing RLtranspose by3Rres.transpose only; no truth. r19 minimallegalcandidate subtracts3sum(Q*Bm.transpose) from S3c to cancel lowrank diagonal leakage; exactbirthmarginal restored0/~2.8e-17contracterror. Small64x4oldcommonreference raw1.9517205e-5→1.9490693e-5(−.136%),7.5005931e-6→7.4234604e-6(−1.028%);charged28,541,708→28,548,044. Not genericprecisionclaim. RealMini1r20running PID1799029,120/.4/5limits,thread1. Source/provenance frozen; ruffscripts pass. Online335672r03still submittedscore null. No newupload.

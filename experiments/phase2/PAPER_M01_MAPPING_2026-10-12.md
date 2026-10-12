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

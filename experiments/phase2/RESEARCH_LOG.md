# Phase 2 研究闭环记录

## 2026-10-10 阶段目标

阶段1 adjusted <5.779e-10；验证达标后阶段2 <5.779e-11，之后继续除以10。十分钟是接续频率，不能把时间到点当成阶段达标。每阶段均要求独立网络成对复验、RPC及资源检查、零失败。

## P2-E003-age3-mini8

问题：当前 V29 raw≈2.27e-8、util≈0.2545。因评分折扣下限为0.1，假定误差不变，仅削减计算最多带来约2.55倍收益；要达到10倍，还必须减少误差，不能只追计量下降。

调研：504aldo derivation_code_map 的历史来源运输/共享基底定义及 findings F72/F73 支持做来源年龄的控制消融。这是公开机制的本地诊断，不是自身算法贡献。

假设：在固定rank、其他参数及参考答案的条件下，让历史三阶来源早一层进入共享子空间，可降低运输成本；检验最后均值的误差损失是否超过省计算的价值。唯一变化：AGE_OLD默认4→3，保留16层返回值与原始公式。

控制：相同Mini前8个网络、十亿样本官方reference、seed=0、单线程、2**41 FLOPs、120秒、residual0.4秒、工具版本锁定。原始V29不改。候选输入不得读取reference；真值仅在评分端使用。

结果：待运行。PID与运行路径见research_state.json；不能将待运行或oracle诊断写成达标。

下一项：根据age3的误差—成本变化，决定做age5的反向消融，或进入按来源年龄/输出敏感性分组的代理设计。Mini8属于开发集，正结果必须独立复验。

来源：https://github.com/504aldo/whest-p2-cumulant-k3/blob/main/docs/derivation_code_map.md （已读取的来源快照commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d）；官方评分折扣规则见项目报告S4。

完成：age3 adjusted=6.45911031371232e-9（比V29恶化11.769%），raw=2.6029440824970607e-8（恶化14.653%），8/8零失败。每网络FLOPs下降2.393%–3.349%，raw全部增加，7/8 adjusted恶化。local峰值RSS=6012860KiB；完整时间/逐网络数据及版本见原始报告、manifest和age3_vs_v29_local.json。结论：拒绝提前压缩，尚未达标，不进入RPC确认。

## P2-E004-age5-mini8

唯一变化AGE_OLD 4→5，其余默认不变；检验保留来源一层能否逆转age3的精度损失，区分来源压缩误差与其他闭合误差。数据、预算、线程、reference沿用冻结Mini8。若收益很小，停止年龄门槛扫描，转向高阶闭合/来源扰动诊断。该开发集结果不构成独立验证。

检查点 2026-10-10T09:58:22.769947+00:00：age5仍运行，PID535199/start_ticks2290150匹配；JSON尚未完整，未启动重复实验，无新增得分。查阅上游固定commit derivation_code_map（本轮raw地址成功读取；GitHub页面超时）。作者文档将K4→K3反馈与K4再生区分，且明确部分推导是未经独立验证的LLM解释。准备下一项条件消融：若age5收益很小，从原V29仅禁用NO_FEED，测量高阶矩阵核心反馈贡献；不同时禁用再生或改变压缩秩，避免归因混淆。执行须等待age5结束。原论文公式仍待核查，不能把该文档解释当作已验证事实。

完成age5：adjusted=5.799663472450135e-9，raw=2.2292124812750558e-8，util=0.26017454294878917，零失败。raw改善约1.81%，adjusted恶化约0.36%；未达标，停止年龄扫描。逐网络成本、时间、内存详见age5_mini8_local.json及manifest，成对报告age5_vs_v29_local.json。

## P2-E005-nofeed-mini8

从冻结V29仅将NO_FEED默认0→1，禁用K4→K3来源反馈；保留再生、WK431、年龄门槛及秩。这是负对照，用于识别矩阵核心反馈的贡献，不预设提分。沿用相同Mini8/reference/seed/预算及工具版本。代码开关只控制1311行的反馈块。原论文 https://arxiv.org/abs/2605.05179 已核对身份与摘要，说明使用累积量和Hermite分布近似；尚未复核补充公式，具体开关语义来自公开实现。候选不读取reference。若反馈重要，下一项单独研究共享来源基底的秩/输出敏感性；若去反馈改善，则逐层定位反馈失配。

用户调整：取消每10分钟定时触发，改为当前会话持续研究；已通过app删除whestbench-10。阶段目标和验证门槛保持不变；nofeed现有进程继续，不重复启动。

持续研究补充诊断：age3前4层误差与V29一致，第5层开始上升；age5前5层一致，第6层开始略改善，后层raw收益最多约3.5%。这符合门槛改变首次发生时点，但只能观察相关，不能单凭aggregate判定单来源因果。完整逐层ratio见source_age_layer_diagnostic.json。年龄修改不能解释所需十倍收益。原论文v2 HTML的4.2、4.3和S.4.3已读取：augmented保留更高阶迹结构、factorization讨论相同输出的表示重组；这并不证明V29再生/压缩近似准确。来源 https://arxiv.org/html/2605.05179v2 。nofeed仍运行，保持现有候选哈希不变。

公共benchmark扩展：prepare_public.py新增--shard选项，默认首分片路径保持原样。分片1下载session39510，输出benchmark/data/mini_shard01；使用相同固定数据commit、逐分片LFS SHA校验、原targets、网络ID记录。本地未参与调参，上游是否用过未知，不能据此宣称完全独立泛化。nofeed进程仍运行，无重复实验。

完成nofeed：adjusted=6.127991349972541e-9、raw=2.4084116923006604e-8、util=0.25449351901579575，8/8零失败；约6.1%恶化，成本几乎不变。反馈有正贡献，拒绝删除，不进入RPC候选确认。逐网络见nofeed_vs_v29_local.json。

## P2-E006-noconfine-mini8

唯一机制：关闭历史来源共享子空间压缩，其他反馈、再生、系数均沿用V29。目的：测量保留完整来源运输后的误差变化，判断输出导向压缩能回收的空间；不是oracle，不使用reference选方向，也不是独立泛化。full transport可能增加时间/内存，保持原120秒和预算，不放宽失败约束。若raw仍接近V29则不能靠压缩恢复十倍精度，应转向高阶闭合诊断；若大幅改善，再考虑合法输出敏感性代理。

参考噪声诊断：固定Mini8 metadata n_samples=1e9，与采样成本array/1024推导一致。平均reported avg_variance/N约7.914e-11 raw，详见reference_noise_proxy_mini8.json。假设为iid、avg_variance是最终输出方差；非实测CI，未测GPU归约偏差/样本依赖，不是算法下界，不能据此降低目标或宣称噪声地板。后续阈值接近该量级时需独立参考复验。noconfine PID/start_ticks核验仍在运行。

资源检查点：noconfine父PID566583身份匹配、计算worker566789正常运行，RSS约10538416KiB，完整来源状态代价明显增加。local数学诊断不具备8GiB普通worker通过证据，不因此放宽或宣称RPC资源通过。分片1下载进程563071仍活着，已约277/538MB，禁止重复启动。论文官方代码链接web返回404，git只做只读clone探测session63999；不把不可获取源码当作已复现。下一项高阶核心诊断应区分NO_REGEN组合开关与单独WK431/feedback，避免将多机制禁用误写单因素。

分片下载失败：session39510 exit1，TLS BAD_RECORD_MAC，非SHA校验结果。原失败日志保留。下载程序每块加入3次有限transport/HTTP重试，SHA验收仍强制；确认旧进程终止后启动session28174、新日志prepare_shard01_retry1.log。未把未完成缓存作为可评测数据。

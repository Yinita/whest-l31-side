# 今天追赶前三：讨论驱动冲刺

公开目标约1.5e-9，冻结Mini8目标应作为开发代理而非线上保证。保留原十倍研究阶段目标，新增今日top3挑战不自动改stage。

已获取97主题301回复，完整索引见DISCUSSION_FULL_INDEX_2026-10-12.md。方法原作者结果均不作为自己复现。

1. D01 leaf8：V29仅STRASSEN_MIN32→8；同期单网adjusted6.116832334984139e-9→5.936366116712754e-9（−2.950%），raw2.290466483145792e-8→2.2903643426275266e-8，零失败；全Mini8运行。来源topic18218post6。
2. D02 Price8covariance：剥离纯Gaussian11二阶项并替换Price积分，mixedchannels及21/22保留；独立6组积分maxerror3.15e-13。单网adjusted6.183921595963314e-9、raw2.29890222414042e-8，零失败，相对V29恶化1.097%，否决covariance-only替换；H04完整21/22不因此完成。
3. 最高价值剩余：K4完整对象（211/31及旧source抵消）而非K22；旧来源输出敏感压缩（避r²重组费）；纯Gaussian完整11/21/22核保持mixedchannels。先明确公式和成本，再计费。
4. 组合只有各单因素通过后才进行，以最强冻结候选为parent，避免把公共数值或未计费oracle注入候选。候选达标再RPC、资源、公共配对确认，真独立reference缺口单独披露。

经济判断：现有约3%降费远小于所需72%降低。讨论没有公开已验证能把V29直接降至前三的完整方案，不承诺当天达到。优先真正结构变化，不能用扩大盲扫替代机制证据。

D01全Mini8：adjusted5.573952118067259e-9、raw2.2720653136687474e-8、零失败、maxresidual0.3018511s，adjusted−3.5478%；raw略恶化，属于降费。D03组合已开始。

D03单网组合：adjusted5.818785780892923e-9，raw2.2695706647368752e-8，零失败；较V29−4.872%，较leaf8−1.9807%。叶块8RPC正在运行；组合Mini8已串行排队，结果未出不宣称全集收益。

D04 consistentGaussian12：11/21/22全通道尾项，冻结V29parent、保留mixedterms，独立conditional积分grid最大截断差2.081e-7；不是exactnear1。排在RPC→组合Mini8之后串行快筛。此前q3已测，本次只用于完整consistenttail的额外gate，不冒充H04完成。

D01 RPC completed: adjusted5.571416345804054e-9/raw2.270960353101259e-8,8zero failures,maxrelative rawdelta0.129780%,fixed16777216 extraFLOPs pernetwork,clientpeakRSS1484756KiB,maxpredict49.9501s/maxresidual0.175762s. Localdevelopment1GiBarraycap remains deployment limitation, no onlinegrade. D03Mini8running serial; D04consistentpair12 queued. D05 target-free current previous-layer marginal D3 output-sensitive sharedbasis DGD,unscale+QR,unchangedranks/ages prepared on leaf8parent; independent existing synthetic mapping contract, no newscore yet, not fullH24nextlayeradjoint. AST passes/no newruff findings;37inherited lint findings retained. Candidatehash/parent in discussion_outputmetric_r01_manifest.json.

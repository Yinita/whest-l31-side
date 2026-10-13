# 高阶边缘状态误差诊断（离线 oracle）

2026-10-13。这份记录不报告候选提分或十倍目标完成。使用 public network0 `logan-fitzgerald` 原始权重 SHA256 `7fb2e3f519b4a90f2c3de9e64d45ad89b7f20ffd254d887e4ba4fa5c1fdf0754`，官方数据commit `aa99830fdc09fad15407b10e8e3459d3e18bba0a`。原始 `mlp_seed=6319981554997072999`；RPC转换后的grader/setup seed为 `3909487726`。运行r01 manifest误把grader seed标成mlp_seed，源文件与原证据保留，后处理明确更正并逐值核验权重；r02已更正字段。

## 计算与证据

用户已经授权Modal CPU研究。r01限定64×1CPU/1GiB、256片×262144输入、两独立流，每片timeout300秒、retry0，实际只有13成功/243失败；保留 timeout、SIGKILL137、preemption自动重启、heartbeat错误，不能视为完整参考。r01总成功3,407,872样本，`completion.complete=false`，仍然保存所有片与失败。实际账单未知；ledger不把失败CPU费隐藏。

r02仅作离线诊断资源修复：65536输入短片、2GiB worker、64×1CPU、交错两独立流、新固定seed，1024片目标总67,108,864样本，timeout300、retry0。最终已自然结束：1015成功、9个timeout基础设施失败，共66,519,040有效输入，stream0/1为33,226,752/33,292,288；`completion.complete=false`保留失败事实。下文partial是历史检查点，最终结果见文末“最终冻结结果”，不能把partial或离线oracle当候选达标。

浮点forward float32、OpenBLAS/OMP/MKL线程1。逐层pre/post均值与M2/M3/M4用float64 batch中心矩＋Pébay并合，降低rawpower相减抵消；每片独立npz和SHA、journal每记录fsync。原r01错误与所有原始资料不覆盖。没有给生产候选提供MC矩或reference。

H18目标无关capture r02记录canonical `mu,var,sigma,phi,Phi,pk1v,D3,g4row` 与对应预测；inverse permutation统一恢复。r01/r02同代码预测maxdelta2.384e-6是数值轨迹漂移，后处理绑定同capture的pred，不能当改进；instrumented资源不视为比赛验收。

## 固定理论归因

使用原官方mean参考只做离线local innovation：

`local_l = (truth_l-pred_l) - Phi_l * W_l.T*(truth_(l-1)-pred_(l-1))`。

对MC真实中心矩构造 `k3=mu3`、`k4=mu4-3*variance²`，与父D3/g4比较；固定readout系数是 `phi/(2sigma)`、`-alpha*phi/(6sigma²)`、`(alpha²-1)*phi/(24sigma³)`。独立256阶Legendre Hermite积分在15组合核验最大误差1.50e-15。该公式是线性边缘归因，省略非线性/高阶混合/重新基底，不能把oracle校正当新算法。

每独立流按shardID分8个不重叠组，保留整个1024-neuron向量；只对输入随机流分块，**不假设神经元独立**。对应两流组的effect乘积在期望上抵消独立MC噪声，再对8pairedchunk给标准误。有限组数、当前部分完成、参考噪声、float32/trace漂移与MC高阶偏差仍存在，负residual估计只是噪声波动。

| 层 | local MSE | K2+K3+K4后 residual 估计 | 8chunk标准误 |
|---:|---:|---:|---:|
| 8 | 2.6813e-9 | 6.50e-11 | 2.21e-10 |
| 9 | 2.8053e-9 | -3.02e-11 | 1.93e-10 |
| 10 | 3.4238e-9 | 3.93e-11 | 1.85e-10 |
| 11 | 4.0155e-9 | -1.22e-10 | 1.09e-10 |
| 12 | 3.1214e-9 | -6.93e-11 | 1.98e-10 |
| 13 | 3.6431e-9 | 2.87e-10 | 8.98e-11 |
| 14 | 4.7425e-9 | 1.10e-10 | 1.32e-10 |
| 15 | 4.4194e-9 | 1.44e-10 | 1.23e-10 |

末层只用K2残余2.306e-9、只K3为3.305e-9、只K4为3.234e-9；组合显示这些状态误差都相关。layer0理论Gaussian应没有高阶状态误差，组合residual估3.392e-9±2.881e-9，尚受较大MC噪声影响，不作精度下界声明。

末层标准化D3父RMS.07029、MC两流.11189/.11197，ΔRMS.08628/.08641；其中父256个zero主要处于饱和/死区，zero区域读出effect约总效应1.1%。活跃 `|alpha|<4` 的ΔD3 RMS.02998/.03004，父.07935，误差不能由撤销零行直接解释。末层标准化g4父RMS.05983、MC.06802/.06822，ΔRMS.01877/.01879，meanΔ.00644/.00666，Δ与父signedcos.310/.321。所有这些仍属于单公共开发网、离线diagnostic；不能声称泛化或把拟合校正加入候选。

## 可复现入口

`capture_covariance_causal_parent_r02.py`、`summarize_covariance_causal_r01.py`、`covariance_causal_chunk_cross_r01.py`、`covariance_causal_highorder_r01.py`、`covariance_causal_signed_bias_r01.py`保留源文件。后3项报告记录script/helper/capture/manifest/journal SHA与实际命令。新的输出version必须唯一，不覆盖冻结证据。

```bash
uv run --frozen python experiments/phase2/covariance_causal_highorder_r01.py --root results/phase2/modal_covariance_causal_r02 --version next_unique_r03 --chunks 8
uv run --frozen python experiments/phase2/covariance_causal_signed_bias_r01.py --root results/phase2/modal_covariance_causal_r02 --version next_unique_r03
```

当前证据为 `results/phase2/modal_covariance_causal_r02/highorder_partial_repro8chunk_r02.json` 与 `signed_bias_partial_repro_r02.json`。下一步等待完整双流，确认这些归因，再围绕实际完整K2/K3/K4闭合提出合法生产机制；不再将未经闭合的lambda细调当十倍路线。

17:56新增检查点：r02已344记录342成功/2个timeout基础设施失败，完整仍在运行。更大独立双流314成功的 `highorder_partial_wholepositive_r04.json` 增加真正非负的单流/平均effect残差，避免只看噪声扣除估计：末层两单流残差3.583e-10/3.377e-10，平均effect后2.214e-10，独立whole-stream cross残差9.487e-11；8chunk估1.178e-10±7.354e-11。层9早先partial曾出现显著负估计，扩大后whole-stream cross1.040e-10；有限chunk标准误不能当严格置信区间。

官方 `sampling_budget_breakdown` 表明network0参考使用1908×524288=1,000,341,504输入，final avgvariance.0786760，名义iid参考均值MSE约7.865e-11。Bake为torch GPU；dtype与局部innovation参考噪声协方差未确认，不能用名义值当误差证书。证据 `covariance_causal_reference_noise_provenance_r01.json`。当归因残余接近这个量级，应称接近参考不确定性，不能继续从oracle诊断宣称指数级可验证进展。

18:02，超过16m/stream检查点：`highorder_partial520_8chunk_r06.json`实际541成功/3基础设施失败；与其先后采集的signedbias报告流样本约17.89m/18.28m。末层8chunk估residual9.255e-11±4.585e-11，whole-cross8.788e-11；单流真正非负残差2.325e-10/2.314e-10、平均effect后1.599e-10。理论K2/K3/K4 oracle信号在独立seed更大样本复现，仍不等于能在合法候选中生产这些矩，也不等于完整链十倍score。

`capture_covariance_causal_parent_post_r03.py`一次增加canonical p2/p3/p4 raw moments并同预测；与r02 samecode predmaxdelta5.245e-6、featuremax2.580e-5的数值漂移明确保留，post诊断只绑定r03自身对象。`covariance_causal_postproduction_r01.py`加入固定pre mean/K2/K3/K4线性响应：末层原postsecond MSE约2.23e-7/1.85e-7，oraclepre响应后1.60e-11/1.53e-11；postvariance误差2.99e-8/2.75e-8变为7.16e-11/7.59e-11。该比较使用相同输入的pre/postMC矩，噪声相关会降低残差，并未使用新候选；可将其解释为大部分p2偏差继承prestate，不能直接排除任何具体高阶生产机制。公式为 `2 EGaussianReLU delta_mu + Phi delta_var + phi/(3sigma) delta_k3 - alpha*phi/(12sigma²) delta_k4`，无truth拟合系数。第一层Gaussian校验afterp2仍约1.6e-9噪声量级。全部原证据见 `postproduction_partial515_r01.json`。

18:07新增完整链归因：`covariance_causal_fullchain_r01.py`按固定理论K2/K3/K4局部效应，加 `Phi_l * W_l.T` 递推穿过全部16层；whole neuronvector和同输入跨层相关保留。冻结 `fullchain_partial_frozenjournal_r02.json`含703成功/4失败、约22.94m/23.13m双流：parent final诊断MSE2.22819e-8，oracle联合K2/K3/K4平均effect后真实positive残差1.92733e-9，whole-stream cross8.08764e-10；8chunk cross估3.03585e-10±7.47032e-10。局部oracle信号虽然强，整链早层MC噪声经输运放大，不能从此声称可部署候选十倍或可靠的残差下界。另冻结局部 `highorder_partial_frozenjournal_r07.json`713成功/5失败，wholecross8.609e-11。

取证修复：旧postprocess开始读journal后、报告结束再次hash正在追加的livejournal，导致多份旧报告的SHA不等于实际输入prefix。旧报告未改；`journal_snapshot_recovery_audit_r01.json`与`journal_usedprefix_recovery_audit_r02.json`明确区分原reportedSHA prefix和按报告count恢复的实际prefix，不能伪称所有原hash正确复现。旧已执行脚本另存 `.beforejournalfix.executed.py`。新脚本改为仅一次读取journalbytes、立即保存 `.journal_input.jsonl`并对同bytes计SHA，上述两个frozen报告的snapshotSHA已核验。精选固定检查点文件/SHA见 `frozen_checkpoint_commit_manifest_r01.json`；livejournal与大npz不纳为完成结果。

18:13参考噪声校验：`covariance_causal_reference_noise_r02.py`从相同immutablefirstweight计算float64解析 `mean=σ/√(2π)`，对官方firstlabel MSE6.97802e-10；解析postvariance均值.68123443，除bake样本数给nominal6.81002e-10，观测为1.0247倍。Parent firstmean对解析值MSE2.15e-15，没有firstlayer算法误差量级证据。`reference_noise_partial836_r01.json`877成功/7失败的逐层MCvariance给nominalmean-noise从6.81e-10下降至7.865e-11；实际firstlabel解析误差经PhiW递推到final仅1.563e-11，不能将全链oracle剩余约1e-9归咎firstlabel误差。参考跨层相关、MC早层variance噪声和有限高阶模型仍需区分。

`covariance_causal_r18_direction_r01.py`固定outputs0–3检查V29 omitted K111×edge单轨道信号：同V29sourceSHA、canonical下一weight方向，与H18state无法直接配对。MC剩余Δκ3 outputs1–3均负（约−.49e-3/−.39e-3/−.56e-3），r18信号均正2.5–3.6e-5；output0虽同号但小信号/MCnoise且超补。结果仅四固定outputs、两个capturecall轨迹不同，不拟合系数，不否决完整历史方法，不称泛化或提分。证据 `r18_direction_partial775_r01.json`。


## 最终冻结结果（2026-10-13 18:17 后）

[Modal r02](https://modal.com/apps/gydctet/main/ap-5dwgqaCWAycdsGLatB0BYw) launcher PID1575604已退出，wall时间2025.246690秒。1024条计划记录全部收齐；1015成功、9个FunctionTimeoutError失败，没有重跑或丢弃失败。成功输入66,519,040，两独立流33,226,752/33,292,288；成功worker报告CPU合计33.605371小时，失败timeout按上限估计另0.75小时，自动preemption重启的实际CPU及账单未知，实际费用不填推断值。该2GiB离线诊断资源配置不能称比赛资源约束验证通过。原始逐片npz保留本地，不加入Git。

最终5份高阶/整链/符号/生产/参考噪声报告都绑定同一个停止追加的1024条journal；各自`.journal_input.jsonl` SHA256与报告字段逐一核验一致（`3c2d80ec35fa501bccc7de54750cca7cc7f24e5e6041a34e8967d46a6fa3cfb7`）。报告script SHA也与当前保存源码一致。旧partial及其SHA问题仍按前述恢复审计保留，没有覆盖。

末层local innovation MSE为4.419388523453e-9。固定K2+K3+K4 oracle线性校正后，8块两流cross估计为1.016281307112e-10、块间标准误1.146256047149e-11；whole-stream cross9.107177172826e-11，单流真正非负MSE为1.732778859603e-10/1.622429604901e-10，两流平均effect后的真正非负MSE1.294160974767e-10。接近参考采样噪声，不能把cross估计、局部误差或oracle归因为提交adjusted score。

完整16层按`Phi_l W_l.T`输运同一固定局部校正，保留跨层同输入及全neuron向量。Parent capture r02 final raw MSE为2.228189543707e-8；该capture的数值轨迹与父strict parent不同，应各自成对比较。

| 完整链固定oracle机制 | 两流平均effect后真实MSE | whole-stream cross估计 | 8块cross均值 | 块间标准误 |
|---|---:|---:|---:|---:|
| K2 | 1.239411477e-08 | 1.179082731e-08 | 1.176701279e-08 | 7.205345938e-10 |
| K3 | 1.873513535e-08 | 1.862918950e-08 | 1.864132787e-08 | 1.890675476e-10 |
| K4 | 1.611589834e-08 | 1.606395155e-08 | 1.607571784e-08 | 8.890308132e-11 |
| K2K3 | 8.194903168e-09 | 7.583966806e-09 | 7.861114775e-09 | 5.444787807e-10 |
| K2K3K4 | 1.484393919e-09 | 8.280727330e-10 | 1.052309197e-09 | 6.313004667e-10 |

K2/K3/K4联合整链平均effect后的真实MSE1.484393919390e-9与cross估计8.280727329980e-10不同，且8块标准误6.313004667076e-10；早层高阶MC噪声经输运放大明显。因此不把这些数值当端到端候选分数、不声称已经达到十倍，也不作精度下界或严格置信区间声明。

最终符号诊断：末层标准化variance ΔRMS .001455829/.001446232，mean Δ −.000304367/−.000309803；D3 parent RMS .070294381，MC真实 .111847317/.111897255，但活跃`|alpha|<4`的800行ΔRMS .029964830/.029993744，parent active RMS .079350917。256个parent-zero行占D3读出effect cross仅1.081229242280e-11/1.018353132572e-9（约1.06%），取消死区zero解释不了主要误差。g4 parent RMS .059825323、真实 .067890712/.067978079，ΔRMS .018398685/.018499989、mean Δ .006389316/.006467943，signedcos .314147/.316101。这里只定位误差，不拟合truth校准系数。

r03同capture的post生产诊断：末层p2原MSE1.853710015852e-7/1.899481414996e-7，固定oracle prestate响应后1.538852839435e-11/1.442502680692e-11；postvariance原MSE2.802039042518e-8/2.751738049335e-8，后6.981299521316e-11/6.943422551624e-11。pre/post矩同随机输入，相关噪声可减小残差，不是误差证书；当前信号优先支持改进pre K2/K3/K4闭合，继续增加单边缘Taylor阶数的优先级降低。

最终参考噪声：官方bake 1908×524288=1,000,341,504输入；first解析Gaussian对官方label MSE6.978023613641e-10，与解析postvariance/N的6.810018637929e-10相符（1.024670×），parent对解析MSE2.151487200196e-15。末层nominal label mean-noise约7.8647e-11；first官方label误差经相同PhiW输运到final只有1.562996095613e-11，不能解释整链oracle余量~1e-9，不能盲减跨层相关label噪声。GPU bake dtype与完整跨层reference噪声协方差未确认。

最终证据入口为`highorder_final_r08.json`、`fullchain_final_r03.json`、`signed_bias_final_r07.json`、`postproduction_final_r02.json`、`reference_noise_final_r02.json`，以及`completion.json`、`ledger_final_r02.json`。精选文件/SHA和验证结果见`frozen_final_commit_manifest_r03.json`；不包含livejournal、原始npz、aggregate或凭据。

## 下一项生产机制优先级

1. target-free地补全prestate的K2/K3/K4联动闭合：严格追踪完整source history、重复support和协方差metric。不能把离线MC状态注入生产候选。
2. 先用小宽度可精确计算的Gaussian/polynomial合同判别混合cumulant生产与运输，再用原始network0固定输出方向门验证；通过后串行strict RPC成对完整网评分。
3. 比较整体与年龄分块source readout，区分最新source局部生产误差与旧source累积输运误差。r18四固定输出单轨道方向不足以否决完整历史；不据MC残差调符号或系数。
4. 终止无机制依据的lambda/边缘阶数扫描；r07实际preactivation LS重构与r15边缘二阶Edgeworth均已在真实strict1恶化，分别保留负结果。十倍目标仍需合法candidate Mini8、独立未见网络和RPC/合同/资源全套证据。

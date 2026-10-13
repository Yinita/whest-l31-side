# 官方 Resources 页面覆盖审计

2026-10-13 实际读取：[dynamic_contents](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/dynamic_contents)。结论：**没有全部用上**。工程入口已有实际执行，数据覆盖、论文完整方法、环境复刻和资料更新仍有缺口。阅读、收集、代数诊断、端到端实验分别记录，不给出混合这些层级的覆盖率。

页面使用动态 React 正文。搜索服务只返回 Loading，随后通过浏览器读取完整正文、所有资料链接及 Phase 2 / Phase 1 / Warm-up 三个配置标签。当前无额外资源 iframe；五个章节为 Start here、Tools & links、Dataset、Quick reference、Cite & organizers。Tools & links 共11个去重资源入口，Explorer出现两次。原HTML缓存为 `benchmark/data/dynamic_contents_20261013.html`，哈希与逐项状态见 `results/phase2/dynamic_contents_coverage_20261013.json`。缓存是观察快照，不代表未来页面不变。

## 页面资料入口逐项对照

| 官方条目 | 实际覆盖 | 项目证据 | 缺口及优先级 |
|---|---|---|---|
| [starter kit](https://github.com/AIcrowd/whest-starterkit) | 已实施 | `benchmark/phase2.py`；`benchmark/README.md`；提交包manifest | 验证、运行、打包、上传均执行；完整Mini100尚缺，见数据项 |
| [flopscope](https://github.com/AIcrowd/flopscope) | 已实施及计量诊断 | `uv.lock`；V29 local/RPC报告；`consistency_335146.json` | 0.12.1工具冻结；RPC astype多16,777,216 FLOPs已记录。P0：setup与硬超时、官方server参数仍需对齐 |
| [Explorer](https://aicrowd.github.io/whestbench-explorer/) | 有说明，未找到实际实验使用证据 | `docs/advanced/use-whestbench-explorer.md`；配套SVG | P2：用于小网络逐层观察后绑定假设/seed；浏览器TF.js工具不替代flopscope计费benchmark |
| [public dataset](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026) | 部分实施 | `benchmark/data/mini_first8/provenance.json`；`mini_shard01/provenance.json`；公开8网确认报告 | 实际16个官方公共网络；Mini100与Full1000未完成。P0：冻结更大成对网络集，再扩展全量 |
| [AIcrowd participate](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026) | 已使用 | #335146 graded；#335672/#335677/#335678上传记录 | 上榜与上传有证据，排队不是评分；两新包本次仍score=null |
| [challenge rules](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules) | 已阅读并部分执行 | 项目报告规则章节；`benchmark/README.md`；package manifests | 资源限制实测有缺口；最终两项指定与获奖后开源事项尚未到完成状态 |
| [Wu等论文](https://arxiv.org/abs/2605.05179) | 映射、部分复现及诊断 | `PAPER_M01_MAPPING_2026-10-12.md`；`paper_m03_r04_matrix_summary.json`；r06/r14/r18 | P1：完整K3合法比赛实现及成本门槛。小shape官方全调用不等于1024×16比赛复现；K4近似不等于完整K4 |
| [ARC announcement](https://www.alignment.org/blog/announcing-the-arc-white-box-estimation-challenge/) | 已阅读用于背景，本次复核 | 项目报告第14节S19 | 256×8为历史暖场配置；实时Phase2取16×1024，不能沿用旧限制 |
| [discussion](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/discussion) | 快照采集、索引、部分真实消融 | `DISCUSSION_FULL_INDEX_2026-10-12.md`；`DISCUSSION_SPRINT_2026-10-12.md` | 97topics/301posts/29PDF只是当时快照；非全部方法复现。本次补读18238第4帖，P1：按增量刷新 |
| [starter kit issues](https://github.com/AIcrowd/whest-starterkit/issues) | 旧文档有入口；本次补读当前4个open issue | `DISCUSSION_EVIDENCE_AUDIT_2026-10-12.md`；本审计下节 | #33/#36/#39/#43记录补齐；未逐项审计所有历史closed issue，不能称全覆盖 |
| 主办方邮箱 `arc-whestbench@aicrowd.com` | 联系方式已知 | 规则/公告 | 不属于必须使用的研究机制；本次未发送邮件 |

## 其余四个章节的执行覆盖

| 页面要求或内容 | 已有执行证据 | 尚未完成或适用边界 |
|---|---|---|
| clone、uv sync、baseline、validate/run/package/login/submit | 工程及提交记录，真实包源码hash核对 | 根目录旧Phase1 estimator是采样对照；当前候选须显式指定路径 |
| 可执行estimator、16×1024、有限数值 | V29真实形状检查、local8与RPC8；新候选RPC8 | 每个候选验收独立记录，不能继承父版全部通过结论 |
| FLOP预算、超限零fallback | `phase2.py`固定2**41；`rpc_score.py`保留违规并按零输出评分 | 页面2.20e12为取整展示；不要改变冻结预算。失败不删去 |
| 附件数据必须打包、禁网络/编译代码/依赖 | 提交包仅estimator、许可、来源；参考真值不上传 | 本地研发server不能证明官方禁网/实例隔离已经完全模拟 |
| Mini100 / Full1000、不重叠、10⁹参考 | 官方前两个分片LFS校验及原targets | 未本地验收全部1100网络/完整split分离；自生成低样本参考只诊断 |
| mlp_id/name/seed、weights、final_means、all_layer_means、avg_variance | provenance列与benchmark逐网记录；权重和targets对齐 | 不意味着每字段都已有独立算法消融。已反复使用的16网不能称最终未见holdout |
| whestbench≥0.14、flopscope≥0.10、v2-phase2 | 0.16.1 / 0.12.1；数据commit `aa99830fdc09fad15407b10e8e3459d3e18bba0a` | 变更版本须重跑冻结基线 |
| 120秒predict、400ms residual | local runner实际限制；RPC评分后验检查 | **RPC未证明硬截止；5秒setup也无硬超时证据**。P0修补，不把旧RPC报告冒充完整grader沙箱 |
| client8GB、instance64GB、代码1core/2vCPU、flopscope7core/14vCPU | RPC client RLIMIT_AS=8GB；线程数固定 | 普通subprocess4/8内存失败；官方核分配/64GBserver未完全模拟。研发server单数组1GiB非确认官方值 |
| score=MSE×max(0.1,F/B)，Phase2不计residual价格 | local与RPC逐网计费，lambda=0 | 算术降费与raw精度分别报告；线上aggregate不证明同网络数值一致 |
| 每日10提交、最后最多指定2项 | 规则已知；保存提交ID且不重复上传 | 最终指定未完成；不是新的算法机制 |
| Wu等引用、ARC/AIcrowd职责、开源30日/Phase1 7日/公开3年 | 项目来源与MIT/NOTICE保留，论文映射 | 获奖后的发布义务尚未触发；不能列作已执行完毕 |
| 历史Phase1/Warm-up标签 | 本次已展开读取，旧Phase1材料归档 | Phase1：32×256/v1/60s/residual计价；Warm-up：8×256/60s。只供历史参考，当前用Phase2 |

## 本次新补的动态资料

- [discussion 18238第4帖](https://discourse.aicrowd.com/t/how-the-hidden-set-evaluation-invokes-whest-run/18238/4)：旧缓存3帖，实时4帖；新帖解释RPC client不保存普通本地操作累积器，相关累积器在trusted server。server生命周期和上限仍需主办方确认，不构成提分证据。
- [issue36](https://github.com/AIcrowd/whest-starterkit/issues/36)、[39](https://github.com/AIcrowd/whest-starterkit/issues/39)、[43](https://github.com/AIcrowd/whest-starterkit/issues/43)：同一默认seed/采样前缀及MSE列解释问题，不能算三种独立算法机会。项目使用原始10⁹参考、Modal各shard独立seed，已有实质处理，补的是来源记录。
- [issue33](https://github.com/AIcrowd/whest-starterkit/issues/33)：旧版文档漏列SetupContext.seed/MLP.seed及API-key旧链接；项目已有显式seed使用，本次未发现新的算法收益。

## 论文覆盖边界与推进优先级

M01仍partial_mapping。M02完整逐层Gaussian闭合使用order12有限协方差级数，单网adjusted4.05726905e-7，不能称Wright无限精确核。M03官方K2/K3/K4×SIMPLE/AUGMENT×dense/factor在小网络调用与birth合同已做，完整比赛移植未完成。M04控制变量、M05条件积分/混合仍pending_derivation；L28门槛与L29负结果不足以完成这两条路线。相关论文是资源页论文扩展研究，不是页面另外列出的条目。

1. P0：为已提交V37b+H18补同候选local8/RPC成对检查；强化RPC setup/predict/residual硬约束，并记录官方部署未知参数。
2. P0：扩大固定公共网络对照，先更大未参与当前选择的分割，再Mini100；当前16网是开发/确认记录，不当最终holdout。
3. P1：M03完整K3误差与成本分解、M04有已知期望的控制变量；方法规格明确后再交执行模型。
4. P1：discussion/issue按时间增量更新，优先官方部署和计量澄清；不把所有附件阅读等同复现。
5. P2：Explorer小网络观察与Python逐层诊断互证；获奖最终指定、报告和开源期限列入里程碑。

## 同时收集的新实验

V37b+H18完整RPC Mini8已完成，源码SHA `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`，PID72431，北京时间11:02:16启动。输出与完整逐网计量：`results/phase2/sprint_public_v37b_h18_r04_rpc8.json`、`sprint_public_v37b_h18_r04_rpc8_vs_v29.json`。

| 指标 | 冻结V29 local Mini8 | V37b+H18 RPC Mini8 |
|---|---:|---:|
| adjusted | 5.7789797558e-9 | **4.6070113621e-9** |
| raw MSE | 2.2702818514e-8 | 2.2693022350e-8 |
| 平均预算用量 | 25.4547% | 20.3136% |
| 失败 | 0/8 | 0/8 |

Adjusted下降20.2798%，raw下降0.04315%；主要收益是成本。这是**不同候选、相同网络的基线对照**，不是同候选local/RPC一致性；报告中已明确重命名相关字段。新RPC max predict57.59493s、max residual0.195844s、client峰值1,488,920KiB。未完成完整同候选local8或独立网络验收，未达到5.779e-10阶段门槛。#335677/#335678本次状态仍submitted，线上分数null。

同步纠正M01文档和M03队列的旧“r22运行中”表述：r22已完成、零失败但Mini8 adjusted恶化0.1482%，已拒绝晋升；原实验JSON及历史日志保留。

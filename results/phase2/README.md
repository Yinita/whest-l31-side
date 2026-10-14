# Phase 2 结果索引

当前最佳 **V37bH18**。证据中的范围标签是结论的一部分：公共benchmark、corpus raw诊断、CV、oracle与组件不能互换。

|证据|内容|
|---|---|
|[线上#335678](online_335678_strict_rpc_r02.json)|已graded：adjusted4.293779716792071e-9|
|[线上#335966](online_335966_r04.json)|同包复测，当前submitted/score null|
|[公共65网严格RPC](sprint_public_v37b_h18_strict_public65_r06.json)|adjusted4.459611165100404e-9，零失败|
|[R57完成摘要](onpolicy_fresh8_r57_summary_r01.json)|新8网raw+2.259982%，否决，逐网费用/时间/哈希|
|[R57完整计量](onpolicy_closed_fresh8_r57/completion.json)|8网原始完成报告；16GiB云诊断|
|[R59整网一致性](onpolicy_whole_parity_r59/completion.json)|16层maxabs0，通过，非分数|
|[R61边界审计](onpolicy_boundary_audit_r61_complete_r01.json)|1560唯一capture、70heads|
|[R62来源绑定](onpolicy_continuation_binding_r62_actual_r01.json)|815原R54＋1R60＋744R61，原失败保留|
|[R56实际组件](onpolicy_actual_model_parity_r56_actual_r01.json)|实际冻结模型组件数值核验|
|[R56源代码集成](onpolicy_integration_r56_actual_r01.json)|fnp-only和normalized算式边界|
|[V29本地/线上一致性](consistency_335146.json)|冻结基线、计量与环境失败区分|

文件命名中的rXX代表实验身份/重试或检查点，不能按文件名最大版本自动选最佳。每个新运行使用新输出路径。completion、manifest、journal保留原始哈希；summary为可公开标量归纳，不覆盖原报告。

本地保留但不发布：原始npz数组、public下载缓存、MC labels、学习系数/模型、日志和profile。已有tracked日志保留历史证据；忽略规则不会删除文件。论文/上游方法许可证见对应estimator目录。

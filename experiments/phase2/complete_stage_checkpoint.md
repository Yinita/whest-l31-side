# 连续执行检查点

未完成全部129项，持续研究仍需推进。不要把 completed_positive 初筛当完整验收。

当前C16 Mini8扩量进程由research_state跟踪。固定13→8→1tanh离线CV独立输出ablation_v30_110_tanh_cv.json/log；训练是离线诊断，不是提交计算。公式和负结果不应重跑覆盖。

待优先：C16扩量→RPC；L01cached/untraced逐值控制门槛；L03等offline负筛仍需明确门槛处置和合法路径；O03完整来源/转换联合；H07–12的official系数与thin表示；C13/C29等具体计算改造。

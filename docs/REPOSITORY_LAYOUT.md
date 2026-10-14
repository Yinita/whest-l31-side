# 仓库目录与证据发布

更新于2026-10-14。整理保留实验身份、路径和哈希；原始证据不删除、不移动，正在运行的程序不改源。首页与DEVELOPMENT给出当前状态，历史检查点保留当时结论。

|目录/文件|用途|当前入口|
|---|---|---|
|README.md|项目首页、最佳版、复跑命令|V37bH18与最新完成结果|
|DEVELOPMENT.md|当前工作、门槛、下一机制|R57否决后的联合信息研究|
|estimators/|独立版本源码、LICENSE/NOTICE|sprint_public_v37b_h18_r01；R56已否决|
|benchmark/|local/RPC/评分/比较工具|README.md，固定工具与资源|
|experiments/phase2/|实验驱动、消融队列、状态、研究日志|TENFOLD_PROGRESS、research_state、RESEARCH_LOG|
|results/phase2/|可公开结果、失败、manifest、审计|README.md索引与新输出身份|
|docs/|官方合同参考和项目管理说明|REPOSITORY_LAYOUT.md|
|archive/phase1/|历史Phase1代码与报告|独立于Phase2成绩|
|archive/research/|当前文档整理前的开发快照|DEVELOPMENT_CHECKPOINT_20261014.md|
|examples/、tests/、local_engine.py|教学、参考和合同工具|不自动作为比赛候选|

## 当前结果的层级

线上成绩来自AIcrowd已graded响应。公共65网严格RPC用于本地提交路径复核。R57是corpus N1e8 reference上的新8网raw诊断，16GiB研究环境；R59仅验证整网实现一致性。它们不具有相同资源环境或相同reference，不能混成一个排行榜分数。

R61训练完成与R62来源绑定证明完整训练及原始失败保留。模型CV、组件核验、代码审计只能证明其各自范围。R57恶化2.26%，因此R56未成为最佳候选，没有严格RPC扩量或线上提交。阶段1十倍目标仍未达。

## 保留与公开

- Git记录可公开源码、许可证、计量结果、实验失败、来源哈希和文档。JSON完成报告与总结使用新路径；不覆盖原冻结baseline。
- 原始数组、下载缓存、MC labels、拟合模型和系数只本地保存。发布的R56源码需要本地私密support，不能在GitHub上直接复现推理；builder、来源绑定和公开审核说明这一限制。
- 新执行日志、profile、progress和生成snapshot按.gitignore仅本地保留。已有tracked历史日志保持原样。忽略文件仍在磁盘，不是删实验。
- R58准备源码及合同留作研究工具；原import失败记录保留。准备完成不等于已运行、已拟合或已提分。

每次更新显式选择文件，先检查private模型/系数/凭据是否被stage，再检查diff、JSON与文档链接，提交后核对GitHub远端HEAD。credentials通过进程环境临时使用，不能写入remote URL、源码或Git配置。

本次整理修正ablation_queue.counts为134项task的实际状态（64项completed类别、70项partial类别）；不把快筛、诊断或partial说成全部验收。R58/R53准备代码保留原哈希，部分style lint及R58旧failed-module一个unused import尚存；不因整理改动冻结实验代码或宣称全仓库lint通过。

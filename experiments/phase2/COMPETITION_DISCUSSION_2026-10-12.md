# 2026-10-12 奖金距离与论坛调研

最新公开榜抓取Niko rank115，提交335146原始adjusted5.418168137938331e-9。前三展示约1.4e-9、1.4e-9、1.5e-9（页面四舍五入）；第三名门槛约需3.61倍改善、降低72.3%。当前H18 Mini8相对冻结基线仅改善2.287%，不能直接预测线上名次。

官方Phase2奖金：第一50,000美元、第二20,000、第三10,000、算法贡献20,000。社区贡献通常500–5,000美元，由主办方裁量。最终私榜与资格审核决定奖项，公开榜不是奖金保证。当前原样公开V29无自身实质贡献，官方明确小修改可能排除奖金资格。

来源与研究影响：
- 榜单：https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards
- 奖金分档：https://discourse.aicrowd.com/t/phase-1-launch-deeper-models-and-increased-prizes/18026
- Phase2公告与截止：https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197 。提交10月17日23:59UTC/北京时间18日07:59；报告24日23:59UTC/25日07:59。
- 算法奖：https://discourse.aicrowd.com/t/algorithmic-contribution-prize-guidelines-how-arc-judges-these-prizes-discretion-technical-writeups-llm-usage/18041 。机制创新、可解释的真实改善，PDF绑定一个graded成功提交；公开报告可同时考虑社区奖。
- 复用资格：https://discourse.aicrowd.com/t/eligibility-of-solutions-built-on-another-participants-mit-licensed-forum-code-4-6/18240 。许可、署名、自身实质贡献。
- 最新10月12日计量PR：https://discourse.aicrowd.com/t/flopscope-fix-pull-requests-with-tests-for-the-open-billing-issues/18241 。参赛者报告14项修复PR，涉及广播收费、einsum对称性、interp等；提到计划release和重评，但本帖不是官方已部署确认。工具变更必须冻结候选重测基线。
- K3方法与25负结果：https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218 。历史rank10不是最新名次。
- runner未决：https://discourse.aicrowd.com/t/how-the-hidden-set-evaluation-invokes-whest-run/18238 。RSS/VSZ、进程持久性、oplog增长未有官方回复；不得自行budget_reset。
- exactGaussianpair：https://discourse.aicrowd.com/t/phase-2-write-up-exact-bivariate-relu-covariance-plus-per-layer-calibration-3x-over-the-kit-baseline-with-negative-results-submission-329951/18214 。作者报告pairexact单独raw改善7.4%，整体1.32e-7，远弱于V29；原始方法参考，非本地复现。

判断：分数奖当前距离大，2%局部改进不足；更有现实价值的替代路线是原创机制与复现诊断的技术报告，但尚没有已获奖或有保证资格的自身成果。完成129实验不直接等于机制贡献或奖金。

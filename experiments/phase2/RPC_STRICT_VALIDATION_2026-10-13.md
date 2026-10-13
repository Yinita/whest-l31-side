# RPC时限与同候选一致性验证

本轮候选 `estimators/sprint_public_v37b_h18_r01/estimator.py`，SHA `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`。保留最初local1退出139及原始报告，不把复跑通过解释为崩溃已定位。

## 同候选local8的新失败

`sprint_public_v37b_h18_r05_local8.json` 使用同Mini8、原始reference、工具0.12.1/0.16.1、seed0、单线程、2**41、lambda0、120/.4/5限制。进程退出0表示评测完整结束，不表示所有网络通过。

| 网络index | local residual秒 | 结果 |
|---|---:|---|
| 0 | 0.385168 | 通过 |
| 1 | 0.395005 | 通过 |
| 2 | 0.396930 | 通过 |
| 3 | 0.398622 | 通过 |
| 4 | 0.374916 | 通过 |
| 5 | 0.411794 | residual失败，零输出 |
| 6 | 0.401516 | residual失败，零输出 |
| 7 | 0.417156 | residual失败，零输出 |

完整local adjusted **0.34217436909220283**，raw **0.34217438065589345**，3/8失败，不能删除失败后报告5网均值。原RPC8仍为4.607011362140062e-9、8/8零失败。五个双方通过网络raw最大相对差0.42244%，每网RPC多16,777,216 FLOPs。`sprint_public_v37b_h18_r05_rpc_local8_parity.json`将local失败行的数值差设为不可比较，保留所有失败和代码/报告hash核验；不能把fallback与真实预测差叫数值漂移。

## 真实API与新监督器

RPC client 0.12.1的BudgetContext只接收flop_budget、quiet、namespace，不接受wall_time_limit_s；真实签名和TypeError已核验。live wall/residual是测量，非硬截止。server --timeout180为空闲session回收，同步REP没有取消协议。

官方whestbench普通subprocess：setup父进程5秒；predict工作窗口120秒、父进程响应5秒grace；residual400ms运行后归零。普通worker的wall_time参数与RPC client不兼容，不能直接替换导入来伪称严格RPC。

新增独立研发工具：

- `benchmark/run_rpc_strict.py`：可信父进程5秒setup、125秒predict响应监督；真实summary wall>120或residual>.4记失败。失败双杀client/server并冷重建下网，保留失败与生命周期；不是只重连socket。
- `benchmark/rpc_strict_worker.py`：使用未修改的SDK与client计量；输入准备在窗外，候选predict/返回物化在真实预算窗内；正常网络保持同一setup和connection。
- `benchmark/rpc_strict_score.py`：全部网络必须有记录；名字/index/官方派生seed对齐；任何执行、形状、有限值、计量、时间异常均zero fallback且multiplier1；未知成本保持unknown。可选同候选local报告仅通过hash/规则/数据核验后比较。

不在RPC等待期间读取瞬时residual作中断判断：dispatch等待只在SDK的finally结算，途中读取会将合法server工作暂算为Python residual。评分使用关闭后真实summary，不修改flopscope计数或包。

## 故障注入证据

测试fixture只用于执行器验收，**不提交比赛**。

| fixture / 输出前缀 | 实测 | 验收 |
|---|---|---|
| setup_sleep / rpc_strict_setup_fault_r01 | 睡眠5.2秒，父等待5秒超时 | setup_timeout保留；原network0失败score0.9434763193，无discount；成本unknown |
| residual_sleep / rpc_strict_residual_fault_r01 | wall0.460332s、residual0.450403s | residual失败保留，score0.9434763193 |
| server_compute / rpc_strict_server_compute_r01 | 12次真实2048矩阵乘法，103.109GFLOPs；wall2.356052s/backend2.339101s/residual0.000241s | 正常通过；证明server计算>.4秒不误判residual。零预测是fixture，非算法分数 |
| blocked_alarm / rpc_strict_blocked_alarm_r01 | 首轮进程外部中断，原因未知；completedfalse、0行 | 未验证通过；原文件/日志保留，新r02重跑 |
| blocked_alarm / rpc_strict_blocked_alarm_r02 | 忽略SIGALRM无限循环，外部125秒响应deadline生效 | time_exhausted保留；unknown成本、zero fallback、multiplier1；client/server均终止 |
| fail_first_then_ok / rpc_strict_recovery_r01 | network0睡眠.45秒失败，network1正常 | 首失败保留；第二个client/server冷启动成功，continuous_lifecycle=false，不伪称不中断生命周期 |

严格scorer的19项回归测试通过，涵盖原V29 RPC8分数逐值一致、各种失败fallback、缺行/重复/seed/name错误、candidate/reporthash错误、local失败、非法计量NaN与float32误差溢出。命令：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest -q tests/test_rpc_strict_score.py`。关闭非项目pytest插件避免ROS插件缺lark，不改变依赖/算法。新脚本与fixtures lint/语法通过。

## 尚未完成

blockedalarm r02及失败后下网络冷恢复验收已完成。完整候选strict RPC8 `sprint_public_v37b_h18_r06_strict_rpc8.json`及同候选local8已完成，候选源码不变。24项测试通过（新增19项与既有Phase2benchmark5项）。64GBserver、官方物理cpuset、单数组上限和隐藏集deployment仍未完整模拟；严格时限工具不等于完整官方沙箱。

完整strict RPC8现已完成：adjusted **4.6043530010738635e-9**、raw **2.2679734534847285e-8**、8/8零失败。maxwall58.214717519s、maxresidual0.197831751s、setup0.063167266s，单一client/server连续完成。全网FLOPs与旧RPC8完全一致；重复分数差−0.0577%只是数值变动，不算算法收益。同候选local8 hash/数据/协议对齐，五个双方成功网络raw相对差最多0.0022403%；三次local失败仍保留，不能称全部跨runner资源验收通过。新严格score与consistency summary保留完整逐网证据。

之后`sprint_public_v37b_h18_r07_strict_public8.json`完整通过官方公共网络8–15，adjusted **4.635058601285328e-9**、raw **2.2802703725233187e-8**、8/8零失败，maxwall56.020694244s/maxresidual0.189817896s。同网冻结V29 local adjusted5.896492018734529e-9/raw2.316187108775125e-8，adjusted下降21.39295%、raw下降1.55068%；主要收益是成本。使用相同冻结原始targets/工具，before报告hash已核验；不同候选和runner的对照不冒充同候选路径一致性。逐网before/after见`sprint_public_v37b_h18_r07_public_before_after.json`。这组以前参与过H18研究，明确不是完全未见最终holdout；Mini100/Full1000尚未完成。

线上更新：335677原样V37b已graded adjusted4.3922571600308725e-9/raw2.1651017334534116e-8；335678 V37b+H18已graded adjusted**4.293779716792071e-9**/raw2.1220624581985702e-8。对历史V29线上adjusted下降20.7522%，对原样V37b线上下降2.2421%；这些是不同线上aggregate观察，不当成同网络单因素因果对照。榜单快照Niko **rank67**，最佳335678；第一档展示1.4e-9，尚有约3倍差距。证据`online_335677_strict_rpc_r01.json`、`online_335678_strict_rpc_r02.json`、`leaderboard_strict_rpc_20261013.json`。阶段1目标5.779e-10未通过。先完成本轮工具与候选验收，再扩大公共网络；Mini8不是全赛验收。

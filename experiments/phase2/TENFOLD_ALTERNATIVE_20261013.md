# 十倍目标：原始误差结构路径与公开资料增量

2026-10-13。本文件只读调查现有证据，准备一个新 target-free 分布实验；不修改冻结候选、队列或研究状态。组件数学通过不等于整网分数提高。

当前 V37bH18 线上 #335678 adjusted `4.293779716792071e-9`、raw `2.1220624581985702e-8`。阶段1目标 adjusted `<5.779e-10`。评分含最低0.1 multiplier，因此无论如何降费，**raw必须 `<5.779e-9`**；若候选仍约0.20B，raw需要 `<2.8895e-9`。到0.1B而保持现有raw只有约两倍 adjusted 改善，不能达到十倍。

## 实时公开信息核验

本轮直接读取官方HTML/公开GitHub API及论坛JSON，未用搜索摘要替代源码核验。页面分数四舍五入，不能据此给出精确实时差距。

| 10月13日公开榜快照 | 提交 | rank | adjusted展示 | raw展示 |
|---|---:|---:|---:|---:|
| J2W | 335031 | 1 | 1.4e-9 | 1.36e-8 |
| marius_binner | 335346 | 1 | 1.4e-9 | 1.15e-8 |
| oliver_chen | 335393 | 3 | 1.5e-9 | 1.20e-8 |

来源：[官方leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards)。目标5.779e-10低于榜首展示值约2.4倍；要求比这些条目的raw再改善约2倍，即使降至费用底线也不能只复制目前领先方法的展示精度。

公开源码没有新增更强版本：[acco-cyber仓库](https://github.com/acco-cyber/arc-whitebox-estimation-2026)最新 `42459cd1a4ce4ce8d111f7bd261c6d87ce75e3c9`（10月3日）；[504aldo仓库](https://github.com/504aldo/whest-p2-cumulant-k3)最新 `1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d`（9月22日）；[ARC官方代码](https://github.com/alignment-research-center/mlp_cumulant_propagation)最新 `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`（8月4日）。论坛[latest.json](https://discourse.aicrowd.com/latest.json)最新相关仍18238四帖，最新回复10月12日19:59 UTC。没有找到当前前三可直接复用的新公开源码。

[504aldo findings F75](https://github.com/504aldo/whest-p2-cumulant-k3/blob/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d/docs/findings_log.md)报告完整dense/高精度链raw约2.07e-8，当前port约2.28e-8；其归因为confinement、薄残差腿与float32，不是lambda表。他人的诊断说明完全消除这些近似也没有证明raw会降至5.779e-9，不能继续把低秩/age/lambda细调作为十倍主路线。

[Wu等论文](https://arxiv.org/html/2605.05179v2)本轮复读§6.3、Appendix D、Table1：固定深度宽度增大有n^-K经验曲线，但论文没有1024×16竞赛实证；Figure8深度误差约L^K。完整K4成本领先项 `24 n^4 L² + (79/3)n^4 L`（factor/basic）、aug更贵，不能把n/L=64代入精度斜率便宣称K4够用。Phase1的n/L=8负结果也不能直接否决Phase2，但需要新的只算所需收缩表示。

## 已有证据定位真正瓶颈

`results/phase2/ablation_v30_119_error_transport.json`把冻结V29误差做离线一阶分解：`Phi_l*(W_l e_(l-1))`为继承均值误差，余量包括局部非线性截断、状态近似、reference噪声。它使用reference，是定位oracle，不能进入候选。

| Mini8层（0起算） | 总误差MSE | 继承均值transport MSE | 局部余量MSE |
|---:|---:|---:|---:|
| 1 | 7.21058e-10 | 4.16963e-10 | 2.98976e-10 |
| 7 | 6.38643e-9 | 4.09049e-9 | 2.32449e-9 |
| 10 | 1.12295e-8 | 7.92902e-9 | 3.26778e-9 |
| 14 | 2.01876e-8 | 1.56433e-8 | 4.44423e-9 |
| 15 | 2.27017e-8 | 1.81239e-8 | 4.33669e-9 |

末层约80%平方误差已继承，cross项接近零。这不是因果上限；但说明**仅li14或最终readout修补不能作为十倍充分机制**。新完整核要通过中层状态递推改变后续误差，不能只测末层。reference iid噪声proxy均值 `7.91431e-11`，不是实测CI，远小于当前raw；目前没有证据把停滞归于reference底噪。

可控继承误差机制是现有 `G_off=lambda*C_off` 和分开设置的K4对角。它在每层参与wk4m/K31、C/K21生产、K4→K3 feed，早层偏差影响后续mean/covariance/state。完整实际图项联合core应在**冻结层1/4/8/12**分别单层替换、余层父算法固定，随后以一个预定多层配置验收；不要读reference选层或调alpha。核心下一步是同对象给diag/offdiag、去重现有proxy，并沿实际metric运输，而不是额外叠加新项。四层选择本身要预注册，不与临时收益排名混用。

现有末层13特征LOO线性/小网络均负；first-ReLU已知期望控制的固定tangent方差仅降约2.75%，pilot回归方差ratio0.955、同预算变差9.35%。这些简单修正不是十倍候选。

## 具体备选：随机二维平面的全网角向积分

准备新脚本 `tenfold_angular_plane_pilot.py`，版本 **tenfold-angular-plane-r01**，不在另一个大实验运行时重评测。它使用weights-only NPZ，不读取target、层均值或reference。

对于无bias的ReLU网络，`f(r u)=r f(u)`。标准高斯 `X=R U` 的半径与方向独立，故 `E[f(X)]=E[chi_n]*E_U[f(U)]`。对随机旋转二维正交plane和随机phase，圆周每个节点的边缘分布均匀于球面。以16个等距角度平均真实完整网络，再乘解析 `E[chi_n]`，在精确算术下对plane/phase随机性无偏。有限网格不等于每plane精确积分，它只改变组内相关性。此机制保留整个真实网络高阶分布，绕开逐层Gaussian闭合。

| 版本 | 名称 | before | after | 预注册验收 |
|---|---|---|---|---|
| tenfold-angular-plane-r01 | 全网随机平面角向平均 | iid高斯；同forward数antithetic对照 | 随机plane16点及独立sphere-radial-RB，64组×2预定seed | 只比较target-free组均值方差与乐观完整预算投影；16节点、64组、seed261013501/502固定，未看到结果不改 |
| 备选r02（未实现） | plane曲线断点精确积分 | 固定角度近似 | 沿layer重LU阈值分裂角区间，区间内正弦/余弦输出解析积分 | 先小shape曲线核验；完整1024×16的区间数、残差时间及计费若不可控便否决 |
| 备选r03（未实现） | 中层残差控制可行性gate | 当前firstReLU弱proxy | 预定层8/12/14的真实激活加冻结线性transport，独立pilot beta | 先oracle中心化只测可达到方差底线；合法center必须估计/解析且偏差计入MSE，不用oracle分数晋升 |

脚本自检已通过：sphere单位范数、正齐次、2维单ReLU角向积分与解析Gaussian均值；ruff通过。**尚未运行真实1024宽pilot，没有raw/adjusted after。**

执行命令（待主实验资源空闲）：

```bash
uv run --frozen python experiments/phase2/tenfold_angular_plane_pilot.py \
  --weights benchmark/data/rpc_inputs/sprint_public_v37b_h18_r04_rpc8/mlp_0.npz \
  --groups 64 --nodes 16 \
  --output results/phase2/tenfold_angular_plane_pilot_r01.json
```

纯采样路径的门槛很高：1024×16单forward约33,570,816 FLOPs，完整预算约65,504个forward。avgvariance≈0.078给iid完整预算raw≈1.2e-6；full-budget multiplier≈1时要到5.779e-10，需约2000倍方差/成本改善。只有“比iid低几倍”毫无晋升意义。脚本直接给完整预算乐观投影，省略所有构造/搬运费用，所以**投影尚不达标即可快速否决单独路线**；投影达标也需真正允许算子计费、独立网络、RPC与偏差核验。不得把radial或plane少量收益包装成十倍进展。

精确角积分的区间爆炸风险是真实机制障碍；随机平面pilot的目的就是用最小实验判断该方向是否值得推导，而非先花资源实现复杂region枚举。中层CV同样必须把中心偏差 `J(E[h]-center)` 算入误差：corr很高并不能补偿错误center，这是公开discussion的反复负结果。

## 可优先替代circle的多层解析控制与局部bias学习

检查现有M04脚本后，只有first-ReLU tangent和first-ReLU pilot回归；**没有测试以下多层已知期望控制**。这比重复首层线性proxy有新的机制，但仍应先做target-free方差门槛。

固定每层Gaussian-linear预测 `z_l(x)=mu_l+x A_l`，其中A可由预注册expected-gain线性链得到，或用独立训练sample的Cov(x,z_real_l)一次估计；mu/A均在held样本之前冻结。不使用网络reference。`s_l²=diag(A_l^T A_l)`、`p_l=Phi(mu_l/s_l)`，定义

```text
eta_l(x) = ReLU(mu_l+x A_l) - E_Gaussian[ReLU(mu_l+x A_l)] - p_l*(x A_l)
g(x) = x A_final + sum_l eta_l(x) J_down_l
```

每个eta及线性项的期望精确为零；J_down用父算法expected gains和真实weights固定。独立pilot拟合beta后，`mean(f-beta*g)`仍无偏（精确算术下），无需真值center。它可以覆盖不同深度的kink创新，first-ReLU特征不足不直接否决。若加独立Gaussian residual做variance匹配，必须用匹配后完整variance计算Eg；额外噪声不与真实深层波动关联，优先先不加。**不能把上一层sample ReLU值递归代入z_l并沿用Gaussian Eg**，那会失去已知期望。

最小gate：冻结network0，pilot2048和held8192两个独立seed；fixed expected-gain A/J、所有16层innovation一个组合、逐output pilot beta一次，held只能评估。sameforward等价费用需要计入所有A/J乘积、Eg、构造和pilot，不是只计算f。约3倍forward费用若方差ratio不小于1/3即连MC也不胜；即便降几十倍仍未逼近十倍赛分。合法计费与RPC尚待，不先造submission。

局部bias学习也是尚未做的独立路线。现L13是把所有层features一起训练**最终层残差**，不是当地创新。现L16递推注入仍partial。已有13特征描述当前局部状态，却无法描述来自前层的有符号误差；因此最终残差LOO R²负并不否决局部创新预测。

具体版本建议 **tenfold-local-innovation-r01**（未实现）：从现8网trace和原weights计算离线训练标签

```text
eps_l = (reference_l-pred_l) - Phi_l * W_l * (reference_(l-1)-pred_(l-1))
```

从li1开始，首层解析均值不训练。固定13特征及预定Hermite交互`alpha^k*phi`、`D3/sigma³`、`K4`后逐层或含layer的共享小ridge预测eps；每折严格排除整网并冻结ridge，不将同网神经元随机split。以前层已经预测的修正计算`delta_l=Phi_l W_l delta_(l-1)+epshat_l`，再用held网络实际总残差评估。先做一阶replay，仅诊断局部误差可预测性；若大幅改善，才接入predict逐层更新mean，并核验Cov/高阶状态一致性及真实重跑。trace母体是V29；任何向V37bH18迁移必须重采对应候选trace，不能把V29离线结果称为最佳candidate分数。

该标签使用reference只用于**离线训练**，推理仅固定系数和合法状态。Mini8反复调参的选择偏差仍在；只有预注册冻结模型后新网络split单次成对确认、实际允许算子计费与RPC通过，才可能晋升。主要判据先提高raw，而非用新增模型名掩盖停滞。

本轮实际执行了两个cached pilot，所有8网trace及weights按原manifest逐fileSHA核验，网络name/seed一致。`weights[layer].T @ e_prev`与原120条分解的局部MSE最大偏差 `<2e-16`，使用冻结V29，未复用V37 permutation。每折整网排除，layers1–14训练，layer0/15局部修正固定0，layer15只运输上一层预测修正。未对held目标拟合、调gate或归一化。

| 版本 | before raw | after raw | after/before | 改善网数 | 结果 |
|---|---:|---:|---:|---:|---|
| tenfold-local-innovation-r01，18features ridge线性 | 2.2701659941e-8 | 2.2357768384e-8 | 0.9848517 | 5/8 | −1.515%，远不足十倍；相同features训练总误差readout为2.2787174951e-8，负 |
| tenfold-local-innovation-r02-nonlinear，唯一变化learner | 2.2701659941e-8 | 2.2189885829e-8 | 0.9774565 | 5/8 | −2.254%，预注册production rawratio<0.25门槛失败；不解封public |

r02使用固定HistGradientBoostingRegressor：depth3、100iter、lr0.05、minleaf128、squared loss、early_stopping=False、random_state261013601；训练label scale仅train折统计。scikit-learn1.7.2只安装ignored `benchmark/data/paper-venv`，未改比赛工具或uv.lock，不进入submission。r01耗时1.78s、r02 6.32s，单线程只做cached计算；ruff和diffcheck通过。原始结果分别 `results/phase2/tenfold_local_innovation_pilot_r01.json` / `tenfold_local_innovation_pilot_r02_nonlinear.json`，含每网络每层、前后raw、全部源码和数据SHA及版本。

结论：当地创新确实比用当前13特征预测累计残差稍有信号，但扩展的18特征仍不足。下一项应新增**实际joint K211/K31生产残差与covariance局部修正特征**，而非对已见Mini8扫树深度/learning-rate；未有因果新输入前不开发复杂production或声称提分。以上只有一阶meanJacobian replay，未重算covariance/high-order feedback，独立reference噪声proxy非实测CI；不报告adjusted、资源通过或十倍达标。

## 多层Gaussian-linear控制实际pilot（17:08）

版本 **tenfold-multilevel-Gaussian-CV-r01** 已执行，network0/logan-fitzgerald、seed3909487726，原weights和trace身份/sha核验，只读取V29 feature列1(mu)/7(Phi)，没有读取prediction/reference。512train、512held、两次独立train/held随机流，额外同forward数antithetic对照。单层精确中心化、零方差、API权重方向合同通过。

实际实现递推为 `control_l=(control_(l-1) @ w_l)*gain_l + eta_l(x)`，初始control=x，`A_0=w_0`、`A_l=(A_(l-1)*gain_(l-1))@w_l`。它在每层保留Gaussian-linear的ReLU创新，Eg按A列范数得到确切Gaussian均值；没有假借真实高层activation的近似均值来称无偏。API column公式W=w.T对应rowbatch h@w，备用angular未运行前也补此方向合同，避免只靠identity权重自检。

| 预定随机流 | iid最终方差 | CV残差方差 | variance ratio | antithetic同forward ratio | 乐观float32 fullB adjusted投影 |
|---|---:|---:|---:|---:|---:|
| train261013711/held261013712 | 0.0800063 | 0.0531911 | 0.664837 | 0.907550 | 2.53388e-6 |
| train261013721/held261013722 | 0.0780101 | 0.0524949 | 0.672924 | 0.911648 | 2.50071e-6 |

33%左右方差降低明显强于first-ReLU单控制，但费用模型显示**3条dense forward**（实际hW、Gaussian xA、递推controlW），不是此前粗想的2条。构造A另计15denseGEMM，全部float64；实际float64 fullB方差投影5.27742e-6/5.20834e-6。float32列仅假设允许实现的乐观成本，不是float32验证。三链同预算还比iid差约2倍。预注册variance ratio<5e-4和阶段5.779e-10必要门槛均失败，不做production，不扫系数或加采样量。

执行4.14s结束，无重计算并发。原报告 `results/phase2/tenfold_multilevel_gaussian_cv_pilot_r01.json`、身份补记录 `_r01_identity.json`；执行源码原样封存 `tenfold_multilevel_gaussian_cv_pilot_r01_snapshot.py` 且SHA与原报告一致。主脚本随后只增加显式身份校验与输出metadata，未覆盖原报告。数据/构造/精度/mean/control偏差边界全部记录；组件试验不计adjusted得分。

该负结果针对冻结expected-gain Gaussian-linear skeleton，不能证明一切控制变量失败。但十倍要求剩余条件方差进一步降低约千倍，任何后续控制路线需要新的深层latent结构与可准确求期望的proxy，不能仅调已有betas。

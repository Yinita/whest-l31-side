# WhestBench 开发日志（支线 `exp/l31-specialist`）

本 worktree：`/home/sti/Desktop/projects/whest-l31-side`  
策略（修订 2026-08-05）：floor 混合管线维护；**`pure-0.2.x` Hadamard 球面采样**已在 Mini 反超主线；线上确认优先。

## 当前最佳

| 版本 | Mini adjusted | 线上 adjusted | 备注 |
|------|---------------|---------------|------|
| **pure-0.2.1s** | **3.009e-07** | **3.202e-07** [#324164](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324164) | Hadamard+rad @ util≈0.40；复交 [#324157](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324157) |
| ~~pure-0.2.19s~~ | 2.889e-07* | **3.302e-07** [#324226](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324226) | *λ=0.97 gain；Mini↑线上↓ |
| ~~pure-0.2.22s~~ | 2.929e-07* | **3.301e-07** [#324236](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324236) | *util=0.39；Mini↑线上↓ |
| **opus-0.3.1s** | **3.836e-07** | **3.925e-07** [#323919](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/323919) | quad θ 字面量 |
| **v0.2.21c** | **3.873e-07** | **4.086e-07** [#324006](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324006) | selective-quad 24-dim；floor=0.093 |
| v0.2.20s | 3.905e-07 | 4.539e-07* [#321758](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321758) | *API 现读；曾记 4.316e-07 |
| v0.2.18s | 3.955e-07 | 4.527e-07* [#321747](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321747) | *API 现读 |
| ~~opus-0.3.0s~~ | 3.865e-07 | FAILED smoke [#321820](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321820) / [#323907](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/323907) | base64/`array` 解码在 grader smoke 挂 |
| v0.2.19s | 3.914e-07* | **4.372e-07** [#321750](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321750) | *λ=0.90 Mini↑线上↓ |
| v0.2.17s | 4.042e-07 | **4.436e-07** [#321643](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321643) | dual + dual-retrained corrector |
| v0.2.16s | 4.045e-07 | **4.487e-07** [#321621](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321621) | dual + single-stream θ |
| v0.2.15s | 4.097e-07 | **4.709e-07** [#321618](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321618) | dual independent RQMC+CV streams |
| v0.2.14s | 4.107e-07 | **4.841e-07** [#321616](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321616) | synth ridge corrector |
| v0.2.12s | 4.159e-07* | **4.945e-07** [#321321](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321321) | floor=0.094（*复测） |
| v0.2.13s | 4.130e-07 | **5.021e-07** [#321587](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321587) | floor=0.095；Mini↑线上↓ |
| v0.2.11s | 4.290e-07 | 4.953e-07 [#321307](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321307) | Σrelu² CV + λ=0.85 |
| v0.2.9s | 4.422e-07 | **4.957e-07** [#321300](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321300) | all-L1 multi-CV + λ=0.85 |
| v0.2.10s | 4.401e-07 | **4.963e-07** [#321291](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321291) | all-L1 + Σrelu² multi-CV |
| v0.2.8s | 4.456e-07 | **4.976e-07** [#321279](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321279) | all-L1 ridge multi-CV + blend |
| 主线 v0.1.3 | 8.762e-07 | 7.003e-07 | 预算感知 IID MC |

相对 opus 线上 **3.925e-07 → 3.202e-07**（**1.23×**，pure-0.2.1s）。相对公开 ~3.07e-7 约差 **4%**。

---

## 版本日志（支线）

### pure-0.2.22s — util=0.39（已提交，线上回退）

| 字段 | 值 |
|------|-----|
| 算法 | 同 pure-0.2.1s；`_UTIL` 0.40→0.39 |
| Mini | **2.929e-07**（vs u40 2.954e-07） |
| 线上 | **3.301e-07**（[#324236](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324236)）劣于 3.202e-07 |
| 结论 | **勿信 Mini 拧 util**；锁 0.40 |

### pure-0.2.19s — Hadamard+rad × gain λ=0.97（已提交，线上回退）

| 字段 | 值 |
|------|-----|
| 算法 | pure-0.2.1s Hadamard；`ŷ=0.97·μ_h+0.03·(0.992·gain)` |
| Mini | **2.889e-07**（优于纯 2.95–3.01e-07） |
| 线上 | **3.302e-07**（[#324226](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324226)）劣于 3.202e-07 |
| 结论 | **再次勿信 Mini 拧 λ**；已回退 `estimator.py`→pure-0.2.1s |

### pure-0.2.10s — Hadamard + Gaussian remainder（已提交，线上回退）

| 字段 | 值 |
|------|-----|
| 算法 | pure-0.2.1s；把 `n_forward - n_hada` 余量用 IID N(0,I) 填满 |
| Mini | **2.987e-07**（vs 3.009e-07，**1.007×**） |
| 线上 | **3.316e-07**（[#324221](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324221)）劣于 3.202e-07 |
| 备注 | Mini↑线上↓；已回退 pure-0.2.1s |

### 否决（pure-0.2.x 本轮）

| 变体 | Mini adj | 判定 |
|------|----------|------|
| util 0.50/0.55/0.60 | 3.20–3.45e-07 | 高 util 伤 adjusted |
| χ / χ_pair 半径 | 更差 | 固定 E[χ] 更好 |
| dual Hadamard | 3.477e-07 | BudgetContext 虚高；官方↓ |
| L1 CV on Hadamard | ~9e-07 | 崩（含 Σrelu²） |
| **a1-only multi-CV** (no Σrelu²) | 最佳 **3.061e-07** (u385) | raw↑但 util↑；全 util 扫仍劣于 3.009e-07 |
| a1cv beta-subsample 8k/2k | **3.121e-07+** | flopscope 仍抬 util；不交 |
| radmix 0.92/1/1.08 | ~1.2e-06 | 崩 |
| Gaussian fill | Mini↑ 线上↓ | #324221 |
| gain blend λ=0.97 | Mini 2.889e-07 / 线上 3.302e-07 | #324226 |
| util=0.39 | Mini↑ 线上 3.301e-07 | #324236 |
| util=0.41 / 0.50–0.60 | Mini adj↑ | 官方 |
| Hadamard-base θ synth | OOS ratio仅 1.009 | 不交 |
| Mini-retrain θ (hyb-0.3.0s) | Mini≈3.07e-07 仍劣于 2.95e-07 | 不交 |
| shared-dir radial 3pt | ~1.16e-06 | 崩 |
| Weyl automorphisms | 5.43e-07 | 崩 |
| radius scale ≠1 | 0.995 即崩 | E[χ] 极敏感 |
| partial Hadamard remainder | 3.143e-07 util↑ | 不交 |
| gain blend λ=0.92/0.95/0.97/0.99 | Mini 最佳 2.889e-07 | **线上 3.302e-07 回退** #324226 |
| row-sign rich / dual-rich | 3.83–4.15e-07 | 劣于 col-sign only |
| dense（无 within ±） | 7.92e-07 | antipodal 必须保留 |
| sphere-scaled L1 CV | 2.98e-07 | ≈持平，不值 |
| util 0.39 | Mini 2.93e-07 / 线上 **3.301e-07** | Mini↑线上↓ #324236；锁 0.40 |
| **gain-cov mid-layer recenter** | ~**3.6e-05** | 毁；目标不是 GT |
| **two-pass empirical recenter** last8/4 @p35/p50 | **7.5e-07–1.05e-06** | 官方 Mini；pilot 噪声毁路径 |
| hard-plug θ onto Hadamard | OOS ~1.004–1.009 | 不交 |
| **Haar R11 rotated Hadamard** | 官方 **3.620e-07** | Mini-probe 虚高；不交 |
| L30 linear DAgger residual | OOS L31 ratio **0.996** | 特征无力 |
| **pure-0.2.60s global fixed frames** | 官方 **3.553e-07** | 劣于 3.009；不交 |
| **pure-0.3.8s dual half-stream mean** | 官方 **3.514e-07** | 劣于 3.009；不交 |

**探针坑：** `seed_protocol=3.0` — 必须 `MLP.from_row(..., "3.0")`；直接用 `mlp_seed` 当 estimator seed → 离线 raw 虚高 ~1.24×。

### pure-0.2.1s — Hadamard + radius @ util≈0.40（P2 突破）

| 字段 | 值 |
|------|-----|
| 算法 | Sylvester Hadamard（±1/√d）× 随机置换/列符号 × antipodal × E[χ_d]；末层 mean；无 blend/CV/θ |
| Mini | **adj 3.009e-07**；raw 7.383e-07；util≈0.408 |
| **线上** | **3.202e-07**（[#324154](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324154)）；secondary raw≈8.075e-07 |
| vs opus 线上 | 3.925e-07 → 3.202e-07（**1.23×**） |
| 线上 | **3.202e-07**（[#324164](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324164)） |
| vs opus Mini | **1.27×**（3.836e-07 → 3.009e-07） |
| vs opus 线上 | **1.226×**（3.925e-07 → 3.202e-07） |
| vs Roberts+rad@0.40 | **1.91×**（5.648e-07 → 2.954e-07） |
| online/mini | **1.084**（迁移健康） |
| util 曲线 | 0.10→3.969e-07；0.20→3.639e-07；0.35→3.028e-07；**0.40→2.954e-07**；0.45→3.043e-07；0.55→3.142e-07；0.70→3.084e-07 |
| Ortho QR | @0.10 3.508e-07；@0.40 3.879e-07 |
| hyb→v0220 | 未重训 θ 变差（4.13–4.31e-07），否决硬接 |
| 文件 | `experiments/pure021s_hada_u40.py`（已拷到 `estimator.py`） |
| Full | 公共 prepared 仅 `mini`；parquet full 待下 |

### v0.2.21c — selective quadratic corrector（已提交）

| 字段 | 值 |
|------|-----|
| 算法 | v0.2.20s pooled dual；θ → 14 线性 + 10 OOS 选 pairwise；floor **0.093**；全 fnp |
| synth OOS | ratio ~1.059（scale=1）vs linear ~1.031 |
| 本地 Mini | adjusted **3.873e-07**；final **3.873e-06**；util≈0.097；失败 0 |
| 相对 v0.2.20s | Mini **3.905e-07 → 3.873e-07**（**1.008×**）；线上 **4.316e-07 → 4.086e-07**（**1.056×**） |
| 线上公开 adjusted | **4.086e-07**（[#324006](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324006)） |
| 备注 | #323996 因 `import numpy` smoke IMPORT_FAILED；corrector 不可逃出 flopscope |

### pure-0.1.x — 高利用率纯 RQMC（新前缀，本轮）

| 字段 | 值 |
|------|-----|
| 动机 | floor 混合 `0.85 μ_CV+0.15 μ_analytic` 有不可随 N 消失的 bias；公开采样 ~3.07e-7 说明纯采样仍有空间 |
| 算法 | Roberts+CP+antithetic+float32 前向+末层 mean；**无** gain/CV/corrector；`_UTIL_CALIB=1.036` |
| Mini util sweep | 见下表；日志 `experiments/results/pure010s_sweep/` |

| 版本 | 目标 util | 实测 util | Mini raw | Mini adj | 判定 |
|------|-----------|-----------|----------|----------|------|
| pure-0.1.0s | 0.10 | 0.1015 | 5.674e-06 | 5.768e-07 | 无 radius 基线 |
| pure-0.1.1s | 0.20 | 0.2024 | 3.262e-06 | 6.603e-07 | raw↓ adj↑ |
| pure-0.1.2s | 0.40 | 0.4086 | 3.782e-06 | 1.544e-06 | 无 radius 高 util 崩 |
| pure-0.1.3s | 0.70 | 0.7177 | 1.037e-05 | 7.437e-06 | **否决** |
| pure-0.1.4s | 0.10+radius | 0.1020 | 5.655e-06 | 5.757e-07 | 低 N 边际 |
| **pure-0.1.5s** | **0.40+radius** | **0.4095** | **1.377e-06** | **5.648e-07** | **纯采样最佳**；raw vs plain@0.40 **2.75×** |
| pure-0.1.6s | 0.20+radius | 0.2016 | 3.032e-06 | 6.114e-07 | 不如 rad@0.10/0.40 |

| 结论 | 高 util **必须**径向 RB；radius@0.40 恢复 p≈1；仍差 screen 3.9e-7 → **P2 换方向集** |
| 复现 | `experiments/pure0{10,14,15,16}s_*.py`；`experiments/results/pure010s_sweep/` |

### opus-0.3.1s — quad corrector 字面量内联（本刀）

| 字段 | 值 |
|------|-----|
| 算法 | 同 opus-0.3.0s（120 维二次交互 ridge）；θ 改为 `fnp.asarray([...])` 字面量，去掉 `array`/`base64`；`delta` clip + `relu` 输出 |
| Mini | **3.836e-07**（raw 3.808e-06，util≈0.100） |
| 线上 | **3.925e-07**（[#323919](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/323919)） |
| 相对 v0.2.20s | 线上约 **1.16×**（按 API 现读 4.539e-07） |
| 否决 | penult CV（raw ~9× 差）；opus-0.3.0s smoke 挂 |

### opus-0.3.0s — quadratic corrector（激进结构改进，本刀）

| 字段 | 值 |
|------|-----|
| 算法 | v0.2.20s pooled base 不变；corrector 从 14 维线性 → **15 维（+σ尺度 s/mean(s)）全两两交互 = 120 维** ridge |
| θ 训练 | 复用 500-MLP synth 缓存（N_GT=1.5e5，`corrector_cache_pooled_v0218.npz`），5-fold OOS 选 lam=1e-3、scale=1.0（无外推，利迁移） |
| **synth OOS** | mean base 5.063e-6 → corr **4.719e-6**（ratio **1.073** vs 线性 1.024）；median/p90/worst-10 **一致改善**；胜率 59.6% |
| **Mini** | final_layer_mse **3.808e-06**；util≈0.101 → **adj 3.865e-07**（vs v0.2.20s 3.905e-07，**0.990×**）；失败 0/100 |
| 迁移预期 | synth OOS −4.6% / mini raw −2.5% 方向一致；adj 因 util>0.1 打折，线上诚实预期 ~4.1–4.3e-7 |
| 实现 | 3 个 θ 数组以 base64(float64) 内联，估计器自包含（grader 只跑 estimator.py）；quad 增广 FLOPs 可忽略 |
| 复现 | `experiments/train_corrector_opus030.py`（用缓存，秒级）；提交 [#321820](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321820) |
| Huber | robust 回归无增益（1.076），说明尾部是结构性误差、二次项可吃，非离群点 |

### 否决记录（本轮）

| 尝试 | Mini adj | 原因 |
|------|----------|------|
| reserve 2.0e9→1.65e9 | 4.03e-07 | util 0.098→0.103，raw 几乎不涨 |
| poolmix 0.5（三路 CV） | 4.19e-07 | 额外 CV 推 util→0.106 |
| poolmix+降 floor | 更差 | 少样本伤害 > mix 收益 |
| penult-gain CV | raw ~8×差 | gain E[h] 有偏，CV 放大误差 |
| λ_ridge↓ / α>1 θ | ≤持平 | Mini 迁移不稳 |
| v0.2.21s 线性 θ@500 | 3.90e-07 | 仅 ~1.002×，被 quad 盖过 |
| corrector 叠 pure base | — | 门槛测试：pure@0.40 误差 **86.6% 方差 / 13.4% 偏差**，corrector 只吃偏差 → 上限 ~1-3%，否决 |
| pure util>0.40（u55/u70） | 3.14/3.08e-07 | raw×util U 形，min 在 0.40；p 在 >0.40 跌破 1（帧覆盖饱和） |
| cubature 诊断（径向/更多帧/Gaussian） | — | 3壳层径向 **差2×**（偷帧数）；2×帧 p≈0.19（饱和）；Gauss 差3×。Hadamard@0.40 已是方向积分前沿 |

### v0.2.20s — pooled + pooled-retrained θ

| 字段 | 值 |
|------|-----|
| 算法 | v0.2.18s 基线；θ 在 pooled 特征上重训（300 synth / N_GT=1.5e5；OOS ratio **1.031**） |
| Mini | **3.905e-07**；final **3.905e-06**；util≈0.098；vs 0.2.18s **0.987×** |
| 线上 | **4.316e-07**（[#321758](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321758)） |
| 相对 v0.2.18s | 线上 **4.329e-07 → 4.316e-07**（**1.003×**） |
| 约束 | **λ=0.85**（勿用 0.90） |
| probe022 | ridge 微调无增益；k=3 pooled 劣 |

### v0.2.19s — λ=0.90 误交（线上回退）

| 字段 | 值 |
|------|-----|
| 算法 | pooled + pooled-θ，但 `_BLEND_RQMC=0.90` |
| Mini | **3.914e-07**（优于 0.2.18s） |
| 线上 | **4.372e-07**（[#321750](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321750)）劣于 0.2.18s 的 4.329e-07 |
| 结论 | 再次印证 **勿信 Mini 拧 λ** |

### v0.2.18s — pooled dual multi-CV（已提交）

| 字段 | 值 |
|------|-----|
| 算法 | 双流 RQMC+anti 样本拼成一份再做 `[a₁\|Σa₁²]` multi-CV；θ 暂用 v0.2.17s |
| Mini | **3.955e-07**；final **3.903e-06**；util≈0.101 |
| 线上 | **4.329e-07**（[#321747](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321747)） |
| 相对 v0.2.17s | Mini **1.022×**；线上 **1.025×** |

### v0.2.4s — Roberts RQMC + antithetic L31（已提交）

| 字段 | 值 |
|------|-----|
| 算法 | Cranley–Patterson 移位 Roberts 格点 → `norm.ppf`；对偶 `(z,-z)`；只返回 L31；`0.098·B` |
| 本地 Mini | adjusted **6.162e-07**；final_mse **6.162e-06**；all_layers **0.791**；util≈0.096 |
| 线上 | **5.715e-07**（#321235）；online/mini **0.927** |
| vs 主线 | 线上 7.003e-07 → 5.715e-07（**~1.23×**） |

### v0.2.5s — RQMC × 缩放 gain 混合（已提交）

| 字段 | 值 |
|------|-----|
| 算法 | 同 v0.2.4s 格点+对偶；终层 `ŷ = 0.8·RQMC + 0.2·(0.992·gain-cov)`；为 gain 预留 ~2e9 FLOPs |
| Mini GT 拟合 | λ≈0.80、s≈0.992 最优（相对纯 RQMC raw ~1.25×） |
| Kronecker 试探 | 混合后仅略优于 Roberts 混合；定稿仍用 Roberts |
| 本地 Mini · local | adjusted **5.178e-07**；final_mse **5.178e-06**；all_layers **0.791**；flops≈**2.541e10**；eff≈**2.591e10**（util≈0.095）；失败 0；乘子 0.1 |
| 相对 v0.2.4s | Mini **6.162e-07 → 5.178e-07**（约 **1.19×**）；线上 **5.715e-07 → 5.448e-07**（约 **1.05×**） |
| 线上公开 adjusted | **5.448e-07**（[#321255](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321255)） |
| 线上 / Mini | **1.052** |
| 相对榜一 | 约 **52×**（线上） |

### v0.2.6s — ‖X‖² CV + blend（未提交）

| 字段 | 值 |
|------|-----|
| 算法 | 同 v0.2.5s，再对 RQMC 批做 ‖X‖² 控制变量后混合 |
| 本地 Mini · local | adjusted **5.215e-07**（劣于 5.178e-07） |
| 备注 | 离线探针虚高；格子上 ‖X‖² 均值已近 n，β̂ 噪声伤害混合。已回退 |

### v0.2.7s — L1-sum CV + blend（本刀）

| 字段 | 值 |
|------|-----|
| 算法 | 同 v0.2.5s；对 Σ_j relu(XW₀)_j 做标量 CV，E=Σ‖w_{·j}‖/√(2π) |
| Mini GT 探针 | L1-sum ≈1.04× vs plain；‖X‖² 单独/组合见 v0.2.6s 失败 |
| 本地 Mini · local | adjusted **4.724e-07**；final_mse **4.724e-06**；all_layers **0.791**；eff≈**2.613e10**（max 2.706e10 < 0.1·B）；失败 0 |
| 相对 v0.2.5s | Mini **5.178e-07 → 4.724e-07**（约 **1.10×**）；线上 **5.448e-07 → 5.101e-07**（约 **1.07×**） |
| 线上公开 adjusted | **5.101e-07**（[#321259](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321259)） |
| 线上 / Mini | **1.080** |
| 备注 | 不用 ‖X‖²；CV 预算中性 |

### v0.2.8s — all-L1 ridge multi-CV + blend（本刀）

| 字段 | 值 |
|------|-----|
| 算法 | 同 v0.2.5s；对全部 L1 ReLU 单元做多维 CV，`E[relu]=σ/√(2π)`；控制 Gram 相对 ridge `1e-4` |
| 本地 Mini · local | adjusted **4.456e-07**；final_mse **4.160e-06**；flops≈**2.694e10**；eff≈**2.906e10**（乘子均值≈0.107）；失败 0 |
| 相对 v0.2.7s | Mini **4.724e-07 → 4.456e-07**（约 **1.06×**）；线上 **5.101e-07 → 4.976e-07**（约 **1.025×**） |
| 线上公开 adjusted | **4.976e-07**（[#321279](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321279)） |
| 线上 / Mini | **1.117** |
| 备注 | 求解 256×256 Gram；略超 0.1 地板仍净赚。对角 ridge 变体 Mini **4.255e-07** / 线上 **4.996e-07**（[#321285](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321285)）未超过 #321279 |

### v0.2.9s — topk / block / z1（未采用）

| 变体 | Mini | 线上 | 备注 |
|------|------|------|------|
| top-k=128 | **4.844e-07** | — | 劣于 all-256 |
| z1+all-L1 λ=0.82 | — | **5.419e-07** [#321290](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321290) | 误提交；劣于 0.2.8s |

### v0.2.10s — all-L1 + Σrelu² multi-CV（本刀）

| 字段 | 值 |
|------|-----|
| 算法 | 同 v0.2.8s；控制再加 `q=Σ a₁²`，E[q]=Σσ²/2 |
| 本地 Mini · local | adjusted **4.401e-07**；final_mse **4.195e-06**；mult≈**0.105**；失败 0 |
| 相对 v0.2.8s | Mini **4.456e-07 → 4.401e-07**（约 **1.01×**）；线上 **4.976e-07 → 4.963e-07**（约 **1.003×**） |
| 线上公开 adjusted | **4.963e-07**（[#321291](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321291)） |
| 线上 / Mini | **1.128** |
| 备注 | 额外一列控制几乎不增 FLOPs；λ 重调到 0.825 仅微利，定稿仍 0.80 |

### v0.2.12s — floor=0.094（已提交）

| 字段 | 值 |
|------|-----|
| 算法 | 同 v0.2.11s；`_FLOOR_FRAC` **0.098→0.094** 压低 util |
| Mini | **4.176e-07**；final **4.155e-06**；mult≈0.1005 |
| 线上 | **4.945e-07**（[#321321](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321321)） |
| 备注 | 已是本刀最佳；cube/fourth/Kronecker@0.42 探针继续筛 |

### v0.2.19s — pooled + λ=0.90（待提交）

| 字段 | 值 |
|------|-----|
| 算法 | v0.2.18s pooled dual multi-CV；`_BLEND_RQMC` **0.85→0.90** |
| Mini | **3.932e-07**；final **3.905e-06**；mult≈0.1006 |
| 相对 v0.2.18s | Mini **3.955e-07 → 3.932e-07**（**1.006×**） |
| 相对 v0.2.17s | Mini **1.028×** |

### v0.2.18s — pooled dual multi-CV（打包未上线）

| 字段 | 值 |
|------|-----|
| 算法 | 双流样本 concat 后做一次 all-L1+sq multi-CV + dual-θ |
| Mini | **3.955e-07**；final **3.903e-06**；mult≈0.1013 |
| 线上 | AIcrowd TLS 受阻，未出分 |

### v0.2.17s — dual + dual-retrained corrector（本刀）

| 字段 | 值 |
|------|-----|
| 算法 | 双流 base 上重训 14 维 ridge（300 synth / N_GT=1.5e5）；OOS ratio **1.036** |
| Mini | **4.042e-07**（略优于 0.2.16s 的 4.045e-07） |
| 线上 | **4.436e-07**（[#321643](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321643)） |
| 相对 v0.2.16s | Mini ~持平；线上 **4.502e-07 → 4.436e-07**（**1.015×**） |

### v0.2.16s — dual + synth corrector（已提交）

| 字段 | 值 |
|------|-----|
| 算法 | v0.2.15s 双流 + v0.2.14s 正确器（θ 仍按单流特征训） |
| Mini | **4.045e-07**；final **4.043e-06**；mult≈0.100 |
| 线上 | **4.502e-07**（[#321621](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321621)） |
| 相对 v0.2.15s | Mini **1.013×**；线上 **1.046×** |

### v0.2.15s — dual RQMC streams（本刀）

| 字段 | 值 |
|------|-----|
| 算法 | 同 v0.2.12s；样本预算对半分给两个独立 CP-shift RQMC+CV 流，再平均 |
| Mini | **4.097e-07**；final **4.097e-06**；mult≈0.100 |
| 线上 | **4.709e-07**（[#321618](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321618)） |
| 相对 v0.2.12s | Mini **1.015×**；线上 **1.050×** |
| 相对 v0.2.14s | 线上 **4.841e-07 → 4.709e-07**（**1.028×**） |

### v0.2.14s — synth residual corrector（本刀）

| 字段 | 值 |
|------|-----|
| 算法 | v0.2.12s + radiant 式 14 维特征 ridge；θ 嵌进 `estimator.py`；200 synth / N_GT=1e5 |
| Synth OOS | ratio **1.020**（λ=0.01） |
| Mini | **4.107e-07**；final **4.107e-06**；mult≈0.100 |
| 线上 | **4.841e-07**（[#321616](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321616)） |
| 相对 v0.2.12s | Mini **1.013×**；线上 **1.021×** |

### v0.2.13s — floor=0.095（已提交，线上回退）

| 字段 | 值 |
|------|-----|
| 算法 | 同 v0.2.12s；`_FLOOR_FRAC` **0.094→0.095** |
| Mini | **4.130e-07**（优于 0.094 的 4.159e-07） |
| 线上 | **5.021e-07**（[#321587](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321587)）劣于 4.945e-07 |
| 结论 | Mini 过拟合；**保留 v0.2.12s / floor=0.094** |

### v0.2.13s — 采样结构试探（均已回退）

| 变体 | Mini adj | 备注 |
|------|----------|------|
| Kronecker 无 anti | **4.390e-07** | 劣于 0.2.12s |
| Roberts 无 anti | **4.348e-07** | 劣于 0.2.12s；离线子集误判 |
| Dual CP-shift | **4.346e-07** | 劣于 0.2.12s |
| Radiant-port（末层解析+错误 Edgeworth） | **4.055e-06** | 崩；κ₃/σ³ 量纲错 |
| **v0.2.12s（保留）** | **4.176e-07** | Roberts+anti；线上 4.945e-07 |

### 下一刀候选

- ~~pure-0.2.1s~~ 仍为线上最佳 **3.202e-07** [#324164](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324164)
- 本轮否决（续）：
  - ~~util=0.39~~ Mini↑线上↓ **3.301e-07** [#324236](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324236)
  - ~~k·E[χ]~~ |k−1|≥0.01 崩；锁 k=1
  - ~~dual-seed / col-perm~~ Mini 3.43–3.52e-07
  - ~~L1 recenter~~ Mini 3.24e-07
  - ~~hyb-0.3.1s θ·0.4~~ 全量 Mini 仅 **~1%**（2.96 vs 2.99）；Mini-fit 不交
  - ~~cheap Hadamard-only θ~~ OOS ratio **1.004**；无增益
- ~~two-pass empirical mid-layer recenter~~（pure-0.2.40s）：官方 Mini **7.5e-07–1.05e-06**，远劣于 3.009e-07；pilot≠GT
- ~~task-rotated ONB (`pure-0.3.6s`)~~ holdout ~1.20× 虚高；官方 Mini **3.609e-07** 劣于 2.95e-07
- Oracle：GT mean-recenter @ L30 alone → final MSE ~**1e-8**（~50–80×）；主要消方差，noisy pilot **不可**替代
- ~~L30 线性/MLP residual~~ OOS ≤1.02 或崩；~~cross-fit hard recenter~~ n=30 远劣于 base
- 偏差分解（订正）：两路**全 N** 独立 Hadamard → bias²≈0（甚至微负），MSE_bar≈½ MSE_single → **近无偏，误差几乎全是方差**；先前 half-N「bias≈½」是误用公式。消偏/θ/JS/Richardson/global-frames 均不应再作为主线
- Additive Φ-corr @ L31（不改路径）：GT L30 → **~52×**（≈ hard recenter）；但对目标噪声同样极脆（≥1% rms 即崩）；gain 目标崩
- 样本 μ30 已相对 GT ~0.1% rms — 值得学的是 **残差 (Y30−μ̂30)**，不是绝对值；现有线性/MLP 残差 ≤1.02
- L30 hard-recenter **极脆**：相对 rms 噪声 ≥5% 即从 ~50× 收益崩到远劣于 baseline（噪声敏感度探针）
- ~~synth hi/lo δ~~ 线性不可学；~~frame 权重~~ Mini OOS 崩；~~f64 accum~~ 无增益
- ~~Full×150 浅残差~~ OOS R²<0；Mini soft-recenter ratio **0.992**
- ~~a1-only multi-CV~~：E[relu] 匹配极好；BudgetContext raw ~1.06×；官方 Mini 最佳 adj **3.061e-07** 仍劣于 3.009e-07（CV FLOPs 抬 util）
- ~~odd L0/pre1 CV~~：对偶下均值≈0，修正恒零
- **下一刀**：非线性大模型（需更多 Full）或全新采样；停浅 θ / recenter / frame 重加权
- Full 分片缓存：`experiments/results/full_cache/full_shard_*.npz`（流式可读）
- **勿** `import numpy`；官方 Mini 门控后仍要线上验证


---

## 分数对照

```text
Version: v0.1.3 (mainline)
Local Mini:  8.762e-07
Online:      7.003e-07

Version: v0.2.4s (RQMC+anti L31)
Local Mini:  6.162e-07
Online:      5.715e-07  (#321235)
Ratio o/m:   0.927

Version: v0.2.5s (RQMC+anti+gain blend)
Local Mini:  5.178e-07
Online:      5.448e-07  (#321255)
Ratio o/m:   1.052

Version: v0.2.7s (RQMC+anti+L1-sum CV+gain blend)
Local Mini:  4.724e-07
Online:      5.101e-07  (#321259)
Ratio o/m:   1.080

Version: v0.2.8s (RQMC+anti+all-L1 multi-CV+gain blend)
Local Mini:  4.456e-07
Online:      4.976e-07  (#321279)
Ratio o/m:   1.117
vs 0.2.7s:   online ~1.025× ; Mini ~1.06×

Version: v0.2.10s (RQMC+anti+all-L1+sq multi-CV+gain blend)  ← current best
Local Mini:  4.401e-07
Online:      4.963e-07  (#321291)
Ratio o/m:   1.128
vs 0.2.8s:   online ~1.003× ; Mini ~1.01×
```

# WhestBench 开发日志（支线 `exp/l31-specialist`）

本 worktree：`/home/sti/Desktop/projects/whest-l31-side`  
策略：专打 L31、中间层可扔、贴/优化 0.1 乘子地板。

## 当前最佳

| 版本 | Mini adjusted | 线上 adjusted | 备注 |
|------|---------------|---------------|------|
| **opus-0.3.0s** | **3.865e-07**（raw 3.808e-06, util≈0.101） | 提交中 [#321820](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321820) | quad corrector（120 维）；synth OOS 1.073 vs 1.027 |
| **v0.2.20s** | **3.905e-07** | **4.316e-07** [#321758](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321758) | pooled + pooled-θ，λ=0.85 |
| v0.2.18s | 3.955e-07 | 4.329e-07 [#321747](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321747) | pooled dual multi-CV + dual-θ |
| v0.2.20s#321756 | 3.905e-07 | 4.330e-07 [#321756](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/321756) | 同算法复交；略差 |
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

相对 v0.2.17s：线上 **4.436e-07 → 4.316e-07**（**1.028×**，v0.2.20s）。相对 v0.2.12s 线上约 **1.146×**。

---

## 版本日志（支线）

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

- ~~pooled + 重训 corrector~~（v0.2.20s，#321758）
- ~~新特征（勿只降 ridge）~~ → **opus-0.3.0s 二次交互 corrector，synth OOS 1.024→1.073，已打包待提交**
- 提交 opus-0.3.0s 出分后：若线上确认 <4.316e-7，可再叠 **三次/选择性交互** 或**扩 synth 规模至 1000+** 继续压 corrector
- 攻 **base 误差**（占 5e-6 主体，重尾）：更强控制变量 / 减小 CV β̂ 估计噪声——但需 OOS 验证，勿被 mini 误导
- 勿回退到只拧 Mini util / λ

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

# WhestBench L31 实验方法总报告

> 支线 `exp/l31-specialist` · 专打末层（L31）均值 · Phase 1（宽 256 / 深 32 / B=2.72×10¹¹）  
> 数据截至 2026-08-05 · 原始流水日志见 [`DEVELOPMENT.md`](DEVELOPMENT.md)

---

## 一句话结论

**线上最佳仍是 `pure-0.2.1s`：3.202e-07**（[#324164](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324164)）。相对公开 ~3.07e-7 约差 **4%**。

本轮挤压尝试（rich/dual/CV/dense/gain-blend）**均未在线上超过 3.202e-07**；gain λ=0.97 Mini 更好（2.889e-07）但线上 **3.302e-07 回退**（[#324226](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324226)）——再次验证勿信 Mini 拧 λ。

| 参照 | Mini adj | 线上 |
|------|----------|------|
| **pure-0.2.1s Hadamard@0.40** | ~3.0e-07 | **3.202e-07** |
| ~~hyb-0.3.0s Mini-θ~~ | 3.07e-07 | — 不交 |
| ~~rad3 / κ-KQ~~ | ~1.15e-06 / n/a | — 否决 |
| ~~gain blend λ=0.97~~ | 2.889e-07 | 3.302e-07 ↓ |
| 公开采样 #323492 | — | ~3.07e-07 |
| opus-0.3.1s | 3.836e-07 | 3.925e-07 |---

## 0. 战略转向（2026-08-05）

### 为何停拧 floor 管线

当前定稿在 C/B≈0.094：

```text
ŷ = 0.85 · μ_CV + 0.15 · μ_analytic  (+ 14/120 维 residual θ)
```

混合项引入**不随 N 消失的 bias**。floor 上 raw≈4.3e-6 → adj≈4.3e-7 的饱和，**不能**外推成「纯 RQMC 也会停在同一 floor」。

公开 float32 采样已到 ~3.07e-7（相对历史 4.316e-7 约 **29%** 可实现差距），且不依赖高阶 cumulant。

### 新前缀 `pure-0.1.x`

| 版本 | 文件 | 内容 |
|------|------|------|
| pure-0.1.0s | `experiments/pure010s_u10.py` | 纯 RQMC，目标 C/B≈0.10 |
| pure-0.1.1s | `experiments/pure010s_u20.py` | 纯 RQMC，目标 C/B≈0.20 |
| pure-0.1.2s | `experiments/pure010s_u40.py` | 纯 RQMC，目标 C/B≈0.40 |
| pure-0.1.3s | `experiments/pure010s_u70.py` | 纯 RQMC，目标 C/B≈0.70 |
| pure-0.1.4s | `experiments/pure014s_rad_u10.py` | + 径向解析归一化 @0.10 |
| pure-0.1.5s | `experiments/pure015s_rad_u40.py` | + 径向 @0.40 |
| pure-0.1.6s | `experiments/pure016s_rad_u20.py` | + 径向 @0.20 |

### 新前缀 `pure-0.2.x` / `hyb-0.2.x`（P2）

| 版本 | 文件 | 内容 | Mini adj |
|------|------|------|----------|
| pure-0.2.0s | `experiments/pure020s_ortho_u40.py` | QR 正交 frame + rad @0.40 | 3.879e-07 |
| **pure-0.2.1s** | `experiments/pure021s_hada_u40.py` | **Hadamard ±frame + rad @0.40** | **2.954e-07** |
| pure-0.2.2s | `experiments/pure022s_ortho_u10.py` | ortho + rad @0.10 | 3.508e-07 |
| pure-0.2.3s | `experiments/pure023s_hada_u10.py` | Hadamard @0.10 | 3.969e-07 |
| pure-0.2.4s | `experiments/pure024s_hada_u20.py` | Hadamard @0.20 | 3.639e-07 |
| hyb-0.2.0s | `experiments/hyb020s_rad_floor.py` | v0.2.20s + rad（未重训 θ） | 4.307e-07 **劣** |
| hyb-0.2.1s | `experiments/hyb021s_ortho_floor.py` | v0.2.20s + ortho+rad | 4.128e-07 **劣** |

**只保留（纯采样线）：** 方向集 + antipodal + E[χ] 尺度、float32 前向、末层 sample mean。  
**去掉：** gain-cov、ridge CV、residual corrector。  

---

## 1. P0 结果：纯 RQMC × util sweep（Mini 100）

| 版本 | 目标 util | 实测 util | raw MSE | **adjusted** | 相对 u10 adj |
|------|-----------|-----------|---------|--------------|--------------|
| pure-0.1.0s | 0.10 | 0.1015 | 5.674e-06 | 5.768e-07 | 纯 RQMC 基线 |
| pure-0.1.1s | 0.20 | 0.2024 | 3.262e-06 | 6.603e-07 | raw↓ adj↑ |
| pure-0.1.2s | 0.40 | 0.4086 | 3.782e-06 | 1.544e-06 | 无 radius 时高 util 崩 |
| pure-0.1.3s | 0.70 | 0.7177 | 1.037e-05 | 7.437e-06 | **否决** |
| pure-0.1.4s | 0.10+rad | 0.1020 | 5.655e-06 | 5.757e-07 | 低 N 径向≈持平 |
| **pure-0.1.5s** | **0.40+rad** | **0.4095** | **1.377e-06** | **5.648e-07** | **系列最佳**；raw 相对 plain@0.40 **2.75×** |
| pure-0.1.6s | 0.20+rad | 0.2016 | 3.032e-06 | 6.114e-07 | 介于 u10/u40；不如 rad@0.40 |

对照：历史纯 RQMC v0.2.4s Mini adj **6.162e-07** → pure-0.1.5s **~1.09×**。

### 解读

1. **无 radius：** 抬 util 损害 adjusted；0.40/0.70 raw 甚至非单调（格点大 N + 半径噪声）。  
2. **有 radius @0.10：** 仅 ~0.2%（χ₂₅₆ 浓度，低 N 时半径方差本就不大）。  
3. **有 radius @0.40：** raw **3.78e-6 → 1.38e-6**（**2.75×**），adj **1.54e-6 → 5.65e-07**。相对 u10 的 raw（5.67e-6）约 **4.1×** / 样本约 4× → **p≈1.03**——半径 RB 后 RQMC 恢复近似 MC 率，adjusted 与 util≈0.10 **基本打平并略优**。  
4. **P0/P1 未过 screen**；换方向集（下节 P2）后反超。

### 晋级门槛

| 门 | 标准 | 状态 |
|----|------|------|
| quick screen | Mini adjusted **＜ 3.9e-7** | **通过（2.954e-07）** |
| Full 确认 | 相对 v0.2.20s 至少 **10%** | 公共 prepared 无 full；**线上替代确认** |
| 上线目标 | **~3.3e-7** | **线上 3.202e-07 已过** |
| 理想 | 逼近/超过公开 **3.07e-7** | 线上差约 **4.3%**（3.202 vs 3.07） |

---

## 1b. P2 结果：正交 / Hadamard 方向集（Mini 100）

| 版本 | 方法 | util | raw | **adjusted** | vs opus Mini |
|------|------|------|-----|--------------|--------------|
| pure-0.2.3s | Hadamard+rad | 0.095 | 3.969e-06 | 3.969e-07 | 0.97×（略差） |
| pure-0.2.4s | Hadamard+rad | 0.198 | 1.836e-06 | 3.639e-07 | **1.05×** |
| **pure-0.2.1s** | **Hadamard+rad** | **0.400** | **7.383e-07** | **2.954e-07** | **1.30×** |
| pure-0.2.5s | Hadamard+rad | 0.551 | 5.704e-07 | 3.142e-07 | 1.22×（劣于 0.40） |
| pure-0.2.6s | Hadamard+rad | 0.702 | 4.392e-07 | 3.084e-07 | 1.24×（劣于 0.40） |
| pure-0.2.2s | ortho QR+rad | 0.097 | 3.500e-06 | 3.508e-07 | **1.09×** |
| pure-0.2.0s | ortho QR+rad | 0.408 | 9.512e-07 | 3.879e-07 | ≈持平 |
| hyb-0.2.1s | v0220+ortho+rad | 0.094 | 4.128e-06 | 4.128e-07 | 劣（θ 未重训） |
| hyb-0.2.0s | v0220+rad | 0.097 | 4.307e-06 | 4.307e-07 | 劣 |

### 解读

1. **Hadamard @0.40 是当前全局 Mini 最佳**（2.954e-07），相对 Roberts+rad@0.40（5.65e-07）约 **1.91×**，相对 opus Mini **1.30×**。  
2. util 曲线：0.10→3.97e-7 → 0.20→3.64e-7 → **0.40→2.95e-7** → 0.55→3.14e-7 → 0.70→3.08e-7。**最优点 ≈0.40**（再高 raw 降不动、乘子吃掉）。  
3. Ortho QR @0.10 也过 screen（3.51e-07），但 @0.40 不如 Hadamard。  
4. **硬接进 floor 混合管线变差**：CV/θ 错配；纯采样单独上线。  
5. 已拷贝到 `estimator.py`；公共 HF prepared 暂无 `full` split → 优先线上确认。
---

## 2. 优先级路线图（更新）

| 优先级 | 方案 | 状态 | 判断 |
|--------|------|------|------|
| **P0** | 高利用率纯 RQMC util sweep | **完成** | 无 radius 时抬 util **否决** |
| **P1** | 径向解析积分（χ₂₅₆ 均值归一） | **完成** | 高 util 必要预处理 |
| **P2** | 正交 / Hadamard 球面 frame | **完成** | **Hadamard@0.40 → Mini 2.954e-07 / 线上 3.202e-07** |
| **P2b** | 深核 NNGP / 径向 quadrature | **否决（经典 κ）** | κ₃₂ 近平坦；κ₈ 能量优化≈正交≈Hadamard；见 §1c |
| **P3** | Hadamard 匹配 θ / 逐层 closure | **首刀否决** | Mini-fit θ 官方 Mini **回退**；analytic closure 太弱；见 §1c |
| **P4** | 低秩条件积分 suffix | 未开 | 高风险高上限 |
### P1 结构

无 bias ReLU MLP 正齐次：`f(rx)=r f(x)`（r≥0）。高斯 `X=RU`，`R∼χ₂₅₆` 与 `U` 独立 →

`E[f(X)] = E[R] E[f(U)]`，`E[R]=√2 Γ(128.5)/Γ(128) ≈ 15.984`。

对每个 RQMC 点：`x' = (E[R]/‖z‖) z`。保持 CP 移位下的无偏性；antithetic 仍成立。

### 建议实验顺序（修订）

1. ~~P0/P1 Roberts±radius~~  
2. ~~P2 ortho/Hadamard Mini~~ → **Hadamard@0.40 胜出**  
3. ~~P2 Mini + 线上~~ → **pure-0.2.1s 线上 3.202e-07**（[#324154](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/324154)）  
4. ~~util 微调 / dual / CV / gain-blend / Mini-fit θ~~ → 均未超 pure-0.2.1s  
5. ~~P2b 经典 κ-KQ~~ → 否决；仅当 **Mini-MSE 任务驱动** 节点/权重优化时再开  
6. ~~global frames / dual half-stream / a1cv 拧 util~~ → 官方均未超 pure-0.2.1s  
7. P3/采样下一刀：正确 `seed_protocol=3.0` 下的新方向结构，或 Full 监督非线性  
**不做：** 未重训 θ 硬接；Mini-only λ/θ；经典 κ₃₂；错误 `mlp_seed` 离线对比。  
**保留：** util≈0.40 + antipodal + E[χ]；`estimator.py` = pure-0.2.1s。
---

## 1c. P2b / P3 首轮（2026-08-05）

### P2b — 深核 quadrature

| 尝试 | 结果 |
|------|------|
| 深度 32 arc-cosine κ₃₂ | 近平坦 → 经典 NNGP-KQ **几乎无信息** |
| κ₈ 能量优化 512 球点 | energy≈0.8345 几乎不动；近正交 ≈ Hadamard |
| 产物 | `experiments/kq_nodes_d256_n512_L8.npz`；**不部署** |

结论：在 L=32 上用解析 κ 设计点集 **打不过** 随机化 Hadamard。若再开 P2b，必须 **经验目标（Mini/Full MSE）** 优化节点或权重。

径向 3 点（共享方向 × {μ±aσ, μ}，`pure-0.2.25/26s`）：Mini adj **1.15e-06**（a=1 与 a=0.5 同崩）≪ pure-0.2.1s。正齐次下 `E[f(RU)]=E[R]E[f(U)]`，单点 E[χ] 已充分；把预算换成多半径 **牺牲角向 frame 数** 得不偿失。**否决。**

### P3 — Hadamard 匹配学习

| 版本 | 方法 | 指标 | 判定 |
|------|------|------|------|
| train `corrector_hada_v030` | Mini 5-fold ridge on Hadamard+gain 特征 | OOS ratio **1.029** | 训练纸面 +3% |
| **hyb-0.3.0s** | 嵌入 θ → 官方 Mini | adj **3.07e-07**（raw 7.50e-07） | **劣于** pure-0.2.1s **2.954e-07**；**不交** |
| synth `corrector_hada021` | 250 synth / sample 特征 | OOS ratio **0.999** | 无增益 |
| `closure_layer_v030` | 对角 gain 逐层线性 TF | analytic final MSE 9.5e-4→7.6e-4（1.25×） | 绝对误差仍 ≫ Hadamard；单用无望 |

结论：末层线性 θ（无论 Mini/synth）**挤不动** Hadamard；逐层 closure 须接在采样轨迹上做 rollout，不能替代 sample mean。

### P3 续：DAgger-lite / 偏差分解（2026-08-05 晚）

| 诊断 | 结果 |
|------|------|
| Oracle：L30 GT mean-recenter（**不含**末层） | final MSE ~**50×** ↓（主要是用 GT **消掉 MC 方差**） |
| 两独立 Hadamard（half-N 公式） | 曾记 bias_frac≈0.49（**不可靠**） |
| 两独立 **全 N**（订正） | bias²≈0，MSE_bar≈½ MSE → **近无偏，主方差**；消偏地板不存在 |
| L30 线性 residual + recenter | OOS ratio **0.99** |
| L30 → diag-gain 混合 recenter | 全面变差 |
| 独立流目标线性 θ（Y−μ_B） | OOS **~1.021**（~2%，不够 4% 缺口） |
| tiny MLP bias θ | OOS 崩（训练不稳） |
| Bias SVD top-16 | 仅 ~54% 能量；非强低秩 |
| cross-fit L30 recenter（pilot→path） | n=30 上 **劣于** base（cf60≈6.0e-7 vs base≈2.8e-7） |
| L30 差作 CV（不 recenter） | OOS 劣于单流 |

**含义：** GT recenter 的 oracle 不可直接部署——换成 noisy pilot 会把方差灌进路径。浅学习抓不住 ~50% 系统偏差。要打公开 4% 缺口，需要 **更好的 L30/末层目标（强逼近/Full）** 或 **真正降方差的设计**，不是 Mini ridge / 硬 recenter。

### P3/P2b 再续（synth hi/lo + 经验 frame 权重）

| 尝试 | 结果 |
|------|------|
| Mini：hi×8 L30 vs GT | lo 1.2e-6 → hi8 **1.6e-7**（~7×）；oracle recenter L31 **2.2e-8** |
| Synth 200× hi/lo δ=μ_hi−μ_lo 线性 | OOS **无法**预测 δ（shrink→0 最优；rδ≪1） |
| float64 累加均值 | ratio ≈1.000（无增益） |
| Mini 经验 Hadamard frame 权重 | OOS **崩**（3.5e-5 vs uniform 9.2e-7）；top-K 也劣 |
| 并行 quad L30 soft recenter | OOS L31 ≤**1.02×** |

**结论：** hi-N 是好老师，但 δ 几乎是不可学噪声；frame 重加权在 Mini-100 严重过拟合。定稿仍 **pure-0.2.1s**。下步优先 **Full 监督大模型** 或全新采样结构，停浅正确器。

### Full 监督浅残差（150 MLPs，进行中扩到 400）

| 尝试 | 结果 |
|------|------|
| Full×150：Y30−μ̂30 线性+权重特征 | OOS **R² < 0**（shrink→0.05 仍略伤） |
| Mini soft-recenter 迁移 | corr **劣**（ratio **0.992**；adj 3.66→3.69e-7） |
| soft-α × gain pushforward | 任意 α>0 崩 |

浅线性/权重特征在 Full 上同样学不动 L30 残差 → 需要 **非线性大模型** 或放弃 residual、改采样。

### Full 监督（protocol 3.0 重跑）

| 尝试 | 结果 |
|------|------|
| Full×250 特征缓存 `l30_feat_cache_v3.npz`（正确 estimator seed） | base L30≈9.74e-7 |
| 线性 ridge 预测 Y30−μ̂30 | 最优 **shrink=0**；任意 sh>0 → R² **<0** |
| 小 MLP（H=32）同目标 | 任意 sh>0 → R² **≪0**（最佳仍 sh=0） |

**结论：** 在正确种子下浅/小非线性残差仍 **学不动**；停 L30 residual / soft-recenter 线。定稿仍 pure-0.2.1s。

### P2b 旁路续（global frames / dual-mean / 探针坑）

| 尝试 | 官方 Mini | 判定 |
|------|-----------|------|
| **pure-0.2.60s** 固定 global Hadamard（seed 12345） | adj **3.553e-07**（raw 8.54e-07） | **否决**（劣于 3.009e-07） |
| **pure-0.3.8s** 双独立半预算流均值 | adj **3.514e-07**（raw 8.70e-07） | **否决** |
| a1cv 族（diag/PCA/subsample β，多 util） | 最佳 ~**3.06e-07** | 未超 base |
| a1cv sub2k | 3.063e-07 | 未超 |
| Weyl/Halton 替换列符号（protocol 3.0） | proxy_adj 更差（r≪1） | **否决** |
| Full×250 线性 residual（protocol 3.0） | OOS 最优 shrink→0（R²≤0） | **否决** |
| Full×250 浅 MLP residual（H=64） | OOS R² **-0.04～-8** | **否决**；门控失败 |
| Full→Mini 全局 bias / scale | r=0.975–0.995 | **否决** |
| Full×40 搜 K 个全局 Hadamard 设计 | 均劣于 per-MLP（r≤0.98） | 再确认 fixed global 无益 |
| gain-init + diag-Gauss suffix | raw ~1e-3 | **崩** |
| frame RNG：np SeedSequence.spawn / fnp stride / pair-xor | Mini r=0.83–0.90 | **否决**（均劣于 sequential fnp） |
| frame 逆方差加权 / JS-on-frames | Mini r=0.94–0.98 | **否决** |

**探针坑（重要）：** Phase-1 数据集 `seed_protocol=3.0`。`MLP.from_row(..., "3.0")` 才得到官方 `mlp.seed`；直接用 parquet `mlp_seed` 当 estimator seed 会使 Hadamard 方向错位，离线 raw 虚高约 **1.24×**（9.15e-7 vs 官方 7.38e-7）。此后离线对比必须 `from_row` + protocol 3.0，或只信 `whest run`。

**保留：** `estimator.py` = pure-0.2.1s。残差/浅学习/全局偏置/换 RNG/frame 加权已死；进行中：Full 上更深 MLP 直接拟合 L31 残差（`train_l31_mlp_direct.py`，H=128）。
---

## 3. Floor 管线回顾（线上回退族；Mini 已被 pure-0.2.1s 超过）

```text
Roberts RQMC + CP 移位 → antithetic (z,-z)
        ↓
双独立流，样本 concat（pooled）
        ↓
all-L1 ReLU + Σrelu² 多维 ridge CV（ridge_rel=1e-4）
        ↓
ŷ = 0.85 · μ_CV + 0.15 · (0.992 · gain-cov)
        ↓
residual corrector（14 维线性 / opus 120 维字面量 θ）
```

| 旋钮 | 定稿 | 备注 |
|------|------|------|
| `_BLEND_RQMC` | 0.85 | 0.90 线上回退 |
| `_FLOOR_FRAC` | 0.094 | 0.095 线上回退 |
| corrector | 14 维 / opus-0.3.1s 120 维 lit | 0.3.0s smoke FAILED |

有效里程碑（线上）：RQMC+anti → gain 混合 → L1 CV → Σrelu² → floor 0.094 → θ → dual → pooled → **opus-0.3.1s**。

---

## 4. 版本排行榜（Mini + 有线上分）

| 排名 | 版本 | Mini | 线上 | 要点 |
|------|------|------|------|------|
| **1** | **pure-0.2.1s** | **2.954e-07** | **3.202e-07** | **Hadamard+rad @0.40** [#324154] |
| 2 | pure-0.2.2s | 3.508e-07 | — | ortho+rad @0.10 |
| 3 | pure-0.2.4s | 3.639e-07 | — | Hadamard @0.20 |
| 4 | opus-0.3.1s | 3.836e-07 | **3.925e-07** | 原线上最佳 |
| 5 | v0.2.21c | 3.873e-07 | 4.086e-07 | selective-quad |
| 6 | pure-0.2.0s | 3.879e-07 | — | ortho @0.40 |
| 7 | v0.2.20s | 3.905e-07 | 4.316e-07† | floor 硬回退 |
| — | pure-0.1.5s | 5.648e-07 | — | Roberts+rad @0.40 |

---

## 5. 操作纪律

1. **信任线上 / Full，不信任 Mini 拧 λ·floor**（历史多次回退）。  
2. **纯采样与混合管线分开记账**——混合 bias floor ≠ 采样方差地板。  
3. **util sweep 先看 raw 的 N 指数 p，再决定是否抬 C/B**；p≤1 时抬 util 必亏 adjusted。  
4. **Warm-up（L7）榜不能直接比 Phase 1（L31）**。  
5. 硬回退：`experiments/CANONICAL_v0220s.py` / opus-0.3.1s。

---

## 附录：关键文件

| 路径 | 说明 |
|------|------|
| `experiments/opus031s_quad_lit.py` | 当前线上最佳 |
| `experiments/CANONICAL_v0220s.py` | floor 管线硬回退 |
| `experiments/pure010s_u{10,20,40,70}.py` | P0 纯 RQMC util 点 |
| `experiments/pure014s_rad_u10.py` | P1 径向归一 @0.10 |
| `experiments/pure016s_rad_u20.py` | P1 径向归一 @0.20 |
| `experiments/pure015s_rad_u40.py` | P1 径向归一 @0.40（Mini 纯采样最佳） |
| `experiments/sweep_pure010s_mini.sh` | Mini util sweep |
| `experiments/results/pure010s_sweep/` | JSON 结果 |
| `DEVELOPMENT.md` | 逐版本原始日志 |

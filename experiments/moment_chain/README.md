# 历史矩链探针

> 2026-10-10 状态：以下结果来自 2026-08 的旧阶段网络；不代表 Phase 2 强基线复现。旧 Next 建议已被根目录项目报告及 DEVELOPMENT.md 取代。此目录使用 NumPy 的程序仅供研发参考，不能直接用于 Phase 2 提交。

experiments/moment_chain — trajectory-calibrated Hermite moment chain.

Port of #314695 / jamesrahenry arc-whitebox-replication `moment_chain/`.

- `portable.py` — numpy Hermite K=10 bivariate ReLU step + features
- `train_dagger_mini.py` — DAgger fit on Mini `all_layer_means` (mean-only default)
- `eval_mini_gate.py` — raw MSE vs Hadamard pure-0.2.1s
- `port_coefs.b64.txt` — baked #314695 coefficients (512 doubles)

## Local Mini / atlas ledger (2026-08-07)

| Variant | Val / gate raw MSE | Notes |
|---|---|---|
| Uncorrected Hermite K=10 | ~5–9e-5 | |
| Baked #314695 coefs | **4.4–7.1e-6** | Full mean+var+offdiag |
| DAgger mean-only (Mini alm) | 1.55–2.0e-5 | Cov corrections matter |
| DAgger + MC cov teacher 8k | **~6.7–9.9e-6** | Near baked |
| Atlas true-M11 DAgger 24/6 | **4.62e-6** | Matches baked order |
| **Two-pass residual/abs** | **≡ pass-1** | **KILLED** (redundant) |
| True cumulant ceiling (atlas κ) | **2.92e-6** | vs MC probe 4.36e-6 (~1.5×) |
| Hadamard N≈25.6k (gate) | **1.14e-6** | Still better than best chain |

White-box Phase-2 axis; **not** a Phase-1 leaderboard challenger. Two-pass and naive capacity bumps closed.

## P2-1 n³ split (2026-08-10, 10 Full nets)

`probe_n3_blocks.py` → `artifacts/n3_blocks_summary.json`

| Block | Mean energy share of next pre-κ₃ |
|-------|----------------------------------|
| (3) diagonal b3 | ~2% |
| (2,1) b21 | ~79% |
| **(1,1,1) b111** | **~15%** |

`rms(b111)/rms(full)≈0.37`. Omitted triple-index structure is material.

## P2-1b gates (2026-08-10)

| Test | Result | Verdict |
|------|--------|---------|
| DAgger + oracle b111 feature | 4.57e-6 vs 4.62e-6 (~1%) | **KILL** stabilizer inject |
| Proxy corr (eigh8 best) | 0.36 | too weak |
| κ3 = b3+b21 only (baked coefs) | **4.29e-5** (~10× vs probe) | b111 required for analytic κ3 |
| κ3 = b3+b21+eigh8 | still ~4.2e-5 | **KILL** weak proxies |

Next: **P2-3** hybrid mean re-anchor (or deferred high-quality b111 sketch).

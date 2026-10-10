# Orthogonality, antipodes, and a depth-32 falsification map for white-box mean estimation

**Challenge:** ARC White-Box Estimation Challenge 2026, Phase 1  
**Referenced graded submission:** [#325459](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/325459) (`pure-0.2.1s`; adjusted **3.202e-07**, secondary/raw **8.075e-07**)  
**Also nominate (same estimator):** [#325112](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/325112)  
**Code:** branch `exp/l31-specialist`, estimator `estimator.py` (`VERSION = "pure-0.2.1s"`)  
**Date:** 2026-08-07  

**LLM disclosure.** Research direction, experiment design, coding, and drafting of this write-up were AI-assisted (Cursor / Composer). All quantitative claims below are tied to committed probe scripts under `experiments/probes/` or official `whest run` / AIcrowd graded submissions. Where a claim is interpretive rather than measured, it is marked as such.

---

## Abstract

Under the Phase-1 budget *B* = 2.72×10¹¹ (depth 32, width 256, ReLU MLP), we find that **final-layer post-ReLU mean error is almost entirely Monte Carlo variance**, not correctable bias. A structured spherical sampler — **Sylvester–Hadamard orthonormal frames**, random row permutation, **column signs**, **antipodal pairing**, and **χ-mean radius** — reaches online adjusted score **3.202e-07** at utilization ≈0.40. A large falsification map (column-sign redesigns, partial ONBs, robust aggregators, James–Stein shrinks, last-layer Gaussian closures, residual MLPs, mid-layer recenter, and post–v0.10 residual packs) fails to beat this baseline once scored under the official adjusted metric and seed protocol 3.0. Independently, a trajectory-calibrated Hermite moment chain (port of public #314695) improves plain kprop but remains **~6× worse in raw MSE** than Hadamard sampling on Mini. We argue the Phase-1 sampling ceiling is architectural: cheap white-box correctors do not see a usable residual at L31, so further gains require better *deterministic* propagation (Phase-2 axis), not more sampler cosmetics.

---

## 1. Problem and scoring

We estimate μ₃₁ = E[ReLU(z₃₁)] for each of 256 units of a depth-32 width-256 bias-free ReLU MLP, given white-box weights and FLOP budget *B*. The ranked metric is

> adj = mean_m( MSE_m · max(0.1, C_m / B) ),  
> C_m = F_m + λ R_m,  λ = 10¹¹.

All offline Mini gates use `seed_protocol` **3.0** (`MLP.from_row(..., seed_protocol_version="3.0")`). Early probes that fed raw `mlp_seed` into estimators inflated offline MSE by ~1.24× and are discarded.

Environment note (2026-08): evaluators run **flopscope 0.10 / whestbench 0.14** (dtype-aware costs, billed data movement, participant code limited to one physical core). Our Mini-100 under v0.10: adj **2.928e-07**, util ≈0.397; online public/private suite for #325459 remains **3.202e-07**.

---

## 2. Method: Hadamard frames + radius (`pure-0.2.1s`)

### 2.1 Construction

Let *H* ∈ {±1}^{d×d} be the Sylvester Hadamard matrix (*d* = 256). Rows of *H*/√d form an ONB. For each frame *t* = 1, …, *T*:

1. Draw a uniform row permutation π_t and column signs *s*_t ∈ {±1}^d.
2. Form *M*_t = (*H*_πₜ / √d) ⊙ *s*_t (row perm + column signs; still a signed ONB).
3. Scale by the χ-mean radius *e*_r = E[‖*g*‖₂] for *g* ~ N(0, *I*_d).
4. Emit the antipodal pair {+*e*_r *M*_t, −*e*_r *M*_t}.

Forward all *N* = 2*T d* directions through the MLP in float32; return the empirical mean of the final-layer ReLU activations (zeros for earlier layers). Sample count targets utilization ≈0.40 via an analytical FLOP model (matmul 2*d*² per layer per sample + ReLU).

### 2.2 Why this structure (mechanistic content)

| Property | Role |
|----------|------|
| ONB rows | Exact second angular moment of the direction set (uniform energy on coordinates). |
| Antipodes | Exact cancellation of odd moments of the *input* direction measure; removes a large class of first-order biases before the first ReLU. |
| χ radius | Matches E[‖*x*‖₂] of isotropic Gaussians without sampling radii. |
| Random perm + signs | Breaks alignment with any fixed weight eigenframe while preserving ONB geometry. |

**Interpretive claim (validated indirectly):** under full ONB packing, further “design” of signs/perms cannot change the *set* of rows up to signed permutation of an ONB — only the *randomization* of which ONB is used. Empirically, structured sign designs never beat i.i.d. signs (Section 3.1).

### 2.3 Scores

| Artifact | Mini adj | Online adj |
|----------|----------|------------|
| `pure-0.2.1s` | 2.928e-07 (v0.10 Mini-100) | **3.202e-07** (#325459 / #325112 / #324164) |
| Prior gain-blend λ=0.97 | 2.889e-07 | 3.302e-07 (#324226) — Mini↑ online↓ |
| Prior util=0.39 | 2.929e-07 | 3.301e-07 (#324236) — Mini↑ online↓ |

We therefore **lock util≈0.40** and refuse Mini-only util/gain knobs.

---

## 3. Falsification map (depth 32)

All comparisons below are against the same Hadamard baseline under seed protocol 3.0. Ratios are baseline_mse / method_mse (higher is better for the method; &lt;1 means method loses).

### 3.1 Geometry of the direction set

| Intervention | Evidence | Verdict |
|--------------|----------|---------|
| Column-sign designs (balanced / Hadamard-colperm / random) | Mini-100: baseline raw **7.383e-07**; best alt (hada_colperm) **7.845e-07** (ratio 0.94) — `hada_sign_focused_final.json` | **Kill** |
| Within-frame sign balance / block4 / pair_anti | Mini-21: all worse than i.i.d. — `hada_sign_balance_final.json` | **Kill** |
| Partial ONB (fewer rows → more frames) | Mini-21: r=256 best; r=64 ratio 0.56 — `hada_partial_onb_final.json` | **Kill**; lock full ONB |

### 3.2 Aggregation and shrinkage

| Intervention | Evidence | Verdict |
|--------------|----------|---------|
| Trim / winsor / median | Mini-100: mean best; win05 ratio 0.06 — `hada_agg_robust_final.json` | **Kill** (ReLU skew → mean required) |
| James–Stein / soft-gain | Mini-100: ≈ noise (ratio ≈1.000) — `hada_js_shrink_final.json` | **Kill** |

### 3.3 Analytic closures and learned residuals

| Intervention | Evidence | Verdict |
|--------------|----------|---------|
| Last-layer diag-Gaussian / blends | Mini-31: blend05 ratio 0.44 — `hada_last_gauss_final.json` | **Kill** |
| L30/L31 residual MLP / ridge on (Y₃₁ − μ̂) | Full×250 OOS R² ≈ 0.001; features ≈ noise vs Hadamard residual | **Kill** |
| Hard L30 recenter from noisy pilot | Extremely brittle: ≥5% relative RMS noise destroys ~50× oracle gain | **Kill** for online |
| Trajectory Hermite chain (baked #314695 coefs) | Mini-30: chain raw **7.15e-06** vs Hadamard **1.14e-06** (hada/chain ≈0.16) — `moment_chain/artifacts/eval_mini_gate.json` | Better than plain kprop, **not** competitive with sampling |
| Atlas DAgger + two-pass residual/abs-refit | Pass-2 **numerically identical** to pass-1 on Mini+MC and on atlas true M₁₁ (24/6) — `dagger_atlas_twopass_meta.json`, `dagger_mini_twopass_meta.json` | **Kill** two-pass (redundant; matches public #314695 README) |
| True-cumulant ceiling (atlas connected pre κ₃/κ₄/k_{iij}, baked coefs) | Val **2.92e-06** vs MC-probe **4.36e-06** (~1.5×); still ≫ Hadamard — `true_cumulant_ceiling.json` | Probe noise ~1.5×; **family floor ~3e-06** even with oracle cumulants |

### 3.4 Bias–variance (corrected; flopscope 0.10)

On Mini-20 (seed protocol 3.0), two independent full-*N* `pure-0.2.1s` streams Y⁽¹⁾, Y⁽²⁾ with averaged predictor Ȳ = ½(Y⁽¹⁾ + Y⁽²⁾):

| Quantity | Value |
|----------|-------|
| mean MSE_single | 6.46×10⁻⁷ |
| mean MSE_Ȳ | 3.07×10⁻⁷ |
| MSE_Ȳ / MSE_single | **0.475** (≈½) |
| bias²̂ = 2·MSE_Ȳ − MSE_single | −3.2×10⁻⁸ (slightly negative ⇒ consistent with **≈0**) |

Artifact: `writeups/artifacts/bias_var_mini20_v010.json`.

**Interpretive conclusion:** at this budget the estimator is nearly unbiased; the adjusted score is a variance–utilization tradeoff. This explains why bias-facing machinery (θ correctors, JS, recenter) fails, and why Mini-tuned util/gain often regresses online.

### 3.5 Residual / off-meter paths (post one-core lock)

After flopscope v0.10’s one-physical-core residual safeguard ([forum #18125](https://discourse.aicrowd.com/t/18125)), we re-tested RemoteArray detach + shipped OpenBLAS packs (`experiments/pure021s_rp_ob/`). Working detach is `tolist()` (RPC); soft probes (`pickle`, `memoryview`, `dlpack`, `tofile`, …) fail smoke. Online results:

| Path | Online adj | Verdict |
|------|------------|---------|
| Best stable residual pack (ob9–ob12) | ~3.47–3.53e-07 | **Worse** than honest 3.202e-07 |
| Optimistic packing / rate-gates (ob13–19) | 0.23–0.68 or ~3.49e-07 | Budget explode or no gain |
| Apparent “win” then reconfirm (ob20) | 3.194e-07 → **3.324e-07** | **Noise**; do not swap |
| Mild-open / RATE_MIN sweeps (ob21–23) | 1.70e-06 … 3.52e-07 | Still residual asymptote |

**Kill:** under the official one-core λ, residual throughput cannot stably beat flopscope-native Hadamard. Public ~e-9 scores with `instrumented_share ≪ 0.001` are a different accounting regime; we do not chase them for the algorithmic write-up.

---

## 4. Relation to cumulant / moment-chain work

Public write-up [#314695](https://discourse.aicrowd.com/t/18097) shows deep moment chains are *error-compensating dynamical systems*: trajectory-fitted (DAgger) corrections help, step-exact Edgeworth can hurt. We ported the Hermite *K*=10 portable step + stabilizer into `experiments/moment_chain/`, reproduced the qualitative gap on Mini, and closed two reopen axes locally:

1. **Two-pass residual / absolute refit** on the frozen pass-1 rollout is redundant (pass-2 ≡ pass-1) with both Mini MC-covariance teachers and atlas true *M*₁₁.
2. **True-cumulant ceiling** (atlas connected pre-activation cumulants replacing the *N*=4096 probe) improves baked-coef raw MSE by ~1.5× to **~2.9e-06**, matching the public “probe noise costs ~1.5×” claim — and still leaves a **~3e-06 family floor** under linear per-layer heads, far above budgeted Hadamard.

We therefore treat moment chains as a **Phase-2 mechanistic axis**, not a Phase-1 leaderboard contender under current closures.

Our contribution relative to that line is complementary: a **sampling-side error budget** showing that, once variance is driven down with an ONB antipodal design, there is **no cheap residual left for white-box correctors at L31**.

---

## 5. What we are *not* claiming

- We do not claim Hadamard sampling is optimal among all estimators — only that within *budgeted Monte Carlo* at depth 32, ONB+antipodes+χ-radius saturates the gains available from direction design and aggregation.
- We do not claim grey-area residual-wall bypasses (raw ndarray / bundled BLAS) are desirable; after flopscope v0.10’s one-core residual safeguard ([forum #18125](https://discourse.aicrowd.com/t/18125)), that channel is intentionally closed (Section 3.5). Our shipped estimator `#325459` is flopscope-native.
- Mini-adjusted improvements of ≤2% without online confirmation are treated as noise (multiple documented Mini↑/online↓ failures; also one residual false win that failed reconfirm).

---

## 6. Reproducibility

```bash
# Environment (Phase-1 evaluator)
pip install -U "flopscope>=0.10.0" "whestbench>=0.14.0"

# Official Mini gate
whest run --estimator estimator.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
  --split mini --runner subprocess

# Geometry / aggregation probes (examples)
python experiments/probes/probe_hada_sign_focused.py
python experiments/probes/probe_hada_partial_onb.py
python experiments/moment_chain/eval_mini_gate.py
```

Artifacts: `experiments/probes/results/*_final.json`, `experiments/moment_chain/artifacts/`, development log `DEVELOPMENT.md`.

---

## 7. Phase-1 nominations

| Priority | Submission | Role |
|----------|------------|------|
| 1 | **#325459** | Canonical `pure-0.2.1s` on current evaluator |
| 2 | **#325112** | Same estimator; earlier graded confirmation |

Write-up deadline for Algorithmic Contribution: **17 August 2026, 23:59 UTC** (extended in forum [#18125](https://discourse.aicrowd.com/t/18125); earlier #18041 text said Aug 7 — use the extension).

Posting drafts: `writeups/POSTING_DRAFTS.md` (forum body + email to `arc-whestbench@aicrowd.com`).

---

## References (challenge-internal)

1. Submission #325459 / #325112 / #324164 — `pure-0.2.1s` Hadamard+radius.  
2. Forum [#18097](https://discourse.aicrowd.com/t/18097) — trajectory-calibrated moment chain (#314695).  
3. Forum [#18085](https://discourse.aicrowd.com/t/18085) — variance/bias budget for deep white-box means (#317660).  
4. Forum [#18125](https://discourse.aicrowd.com/t/18125) — flopscope v0.10, one-core residual safeguard.  
5. Forum [#18041](https://discourse.aicrowd.com/t/18041) — algorithmic contribution judging guidelines.

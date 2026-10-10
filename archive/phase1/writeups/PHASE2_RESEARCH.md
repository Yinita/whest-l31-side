# Phase 2 research outline (pre-announcement)

**Status:** Phase 1 locked (`pure-0.2.1s` / #325459). Grey residual path dead under flopscope 0.10 one-core + prize diligence.  
**Date:** 2026-08-10  
**Local assets:** `experiments/moment_chain/`, atlas cache (~30 nets), Full shards `experiments/results/full_cache/`

Wait for the official Phase 2 announcement (budget, architecture, whether residual/λ remains). Below is what we can prepare **without** that policy.

---

## 1. Goal shift

| Phase 1 | Phase 2 (intended) |
|---------|-------------------|
| Budgeted MC variance at L31 | Deterministic / hybrid white-box that **beats** sampling under the new budget |
| Ship: Hadamard ONB+antipodes | Ship: trajectory-stable moment (or richer) propagation |
| Residual FLOPs arbitrage | Assume **flopscope-native only** (forum #18125 / #18132: may require all numerics through flopscope) |

Success bar (provisional): raw MSE **< Hadamard at matched F**, and adj competitive after util. Local Phase-1 reference: Hadamard gate ~**1.1e-6** raw; best linear chain ceiling ~**2.9e-6** (true κ) / ~**4–5e-6** (MC probe).

---

## 2. What is already falsified (do not reopen)

- Linear per-layer stabilizer capacity alone → floor ~3e-6 even with oracle cumulants  
- Two-pass residual / abs-refit on same feature family → redundant  
- Teacher-forcing (non-rollout) closures → understate drift  
- Step-exact Edgeworth injected into compensating chains → often hurts  
- Sampling cosmetics / Mini util-gain / cheap L31 residual learners → no Phase-1 win  
- Bundled BLAS / tolist residual packs → online worse + policy risk  

---

## 3. Open scientific axes (priority)

### A. State that carries compensating errors (highest EV)

Public #314695: depth failure is **error dynamics**, not one-step closure order.  
Need a state / corrector that does **not** destroy the chain’s internal error compensation while still reducing coherent bias.

Candidates (gate each on Mini raw vs baked 4–5e-6 and vs Hadamard 1e-6):

1. **Approximate (1,1,1) / n³ block** — P2-1 energy ~15%. **P2-1b gates (2026-08-10):**
   - Stabilizer extra-feature inject (oracle b111): val 4.57e-6 vs 4.62e-6 (**~1%**) → **KILL inject**
   - Cheap proxies: best corr(eigh8,b111)=**0.36**; proxy inject slightly worse
   - Analytic κ3 transport under baked coefs: probe 4.36e-6; atlas full κ3 5.31e-6; **b3+b21 only 4.29e-5 (~10×)**; eigh8 proxy does not recover
   - **Keep** only if a near-oracle b111 approx appears (corr≫0.9); else probe-free analytic κ3 is blocked. Artifact: `p21b_b111_gate.json`, `p21b_k3_transport_gate.json`
2. **Low-rank / eigenbasis corrections with closure-derived features** (naive rank-32 already killed in replication; try features tied to b111 residual, not probe-κ alone).  
3. **Layer-coupled heads** (share coefs across depth bands; or predict *deltas of deltas* along trajectory).  
4. **Hybrid reset**: every *k* layers, spend small MC budget to re-anchor **mean only** (cov stays chain) — oracle mean-reset was strong; noisy reset was brittle in Phase 1, so test with Hadamard pilot budget accounting.

### B. Richer closures without breaking compensation

1. Use atlas `pre_*` connected moments to measure **how much** omitted n³ structure feeds next-layer marginal means (dataset README identity). If large → justify approximate n³ contraction; if tiny → look elsewhere.  
2. Pair-copula / vine light models for ReLU bivariate tails (only if FLOP model allows cheap CDF evals).

### C. Probe efficiency

True κ beats N=4096 MC by ~1.5×. Options: larger probe under Phase-2 budget; antithetic / control-variate probes for κ3/κ4 only (not means); reuse forward activations from a cheap Hadamard pilot as the probe (shared FLOPs).

### D. Honest hybrid (only if Phase 2 still scores adj)

Spend fraction α of budget on chain (low util, low variance? No — chain has high bias floor) vs Hadamard. Phase-1 CV corr(hada, gain)≈0 → blend unlikely unless chain bias is *orthogonal and large*. Gate: OOS MSE of `λ·chain+(1-λ)·hada` must beat hada alone by **>5%** on Mini before any submit.

---

## 4. Local workstreams (can start now)

| ID | Task | Effort | Gate |
|----|------|--------|------|
| P2-0 | Freeze Phase-1 ship + post write-up #325459 | human | done locally |
| P2-1 | Atlas n³ / (1,1,1) energy split | **done** | ~15% energy — material |
| P2-1b | b111 approx → stabilizer / κ3 transport | **done** | **KILL** cheap inject+eigh8; κ3 needs near-oracle b111 |
| P2-1c | High-quality b111 sketch (deferred) | — | only if corr≫0.9 on held-out |
| P2-2 | Band-shared DAgger | parked | after P2-3 |
| P2-3 | Mean-only MC re-anchor every 4/8 layers + chain cov | **next** | Mini raw < 1.5e-6 vs hada |

Default next: **P2-3** (hybrid reset). Do not spend more on weak b111 proxies.

---

## 5. Policy watchlist

From forum #18125 / #18132 (not final):

- All numerical work through `flopscope`?  
- Cap residual wall time / residual share?  
- New width/depth/budget?  
- Private re-eval instrumented-share diligence (Phase 1 prizes)

Do **not** design around residual arbitrage for Phase 2.

---

## 6. Nomination / write-up (Phase 1 still)

- Write-up due **17 Aug 2026 23:59 UTC** → `writeups/dist/`  
- Nominate when emailed: **#325459**, **#325112**

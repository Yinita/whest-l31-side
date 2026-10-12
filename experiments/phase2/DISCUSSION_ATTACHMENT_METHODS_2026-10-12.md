# 附件方法定位

下列章节从全文抽取，供后续复现定位；不代表章节结论已经本项目验证。

## Topic 18241

- Auditing flopscope’s cost model: issues, reproductions and
- Associated submission: 335434. Algorithmic write-up: submitted separately for the same submission.
- 3. Method
- one. multivariate_normal(method=...) and hermitian=True in svd, pinv and matrix_rank are
- method= and
- edits the same matrix_rank lines as #291 and only changes how method= in multivariate_normal and
- multivariate_normal(method=...)
- costs 0 there and k * (7n - 1) through

## Topic 18145

- tracked depth-5 Strassen–Winograd kernel that costs 75 = 16,807 products where the conventional recursion charges 323 = 32,768.
- cost side moved without the accuracy side moving at all. On one fixed 100-network development cohort, holding the statistical
- charged-cost term. On ten held-out selection networks the two arms agree to a maximum final-mean drift of 7.551e-6 (dense raw
- 1.715489347e-7, Winograd raw 1.715362009e-7), confirming the change is cost-only rather than a different estimator. This is the
- 3. Method
- are the dominant cost. They are evaluated by an exact depth-5 Strassen–Winograd recursion: five levels of the 7-multiplication
- 5. Cost accounting, and the mistake it prevented
- selection criterion. This is worth stating plainly because it cost me a promotion decision. A streaming variant of the kernel
- Transferable warning. On this benchmark, any candidate must be priced on complete measured subprocess cost, not on tracked
- difference exactly 0.000e+00 — so this is purely a cost choice:
- 7. Negative results worth other people's time
- • Shared-reference Taylor evaluation of a heteroscedastic mixture state. Expanding component covariance maps around the
- pooled-within covariance shrank covariance offsets substantially (at K=64, layer 29: 0.574 to 0.357) while approximation error
- • Low-rank Hermite / direct-diagonal extraction. Truncating the component covariance to rank r gave relative errors 2.16e-1,

## Topic 18216

- 03_covariance_propagation
- v2 exact ReLU covariance +           64               none         1.230525e-06      3.37x         2.36%      1.23e-07
- v6 inherited-cumulant CP             0                none         1.134226e-06      3.65x         9.89%      1.13e-07
- the same two-part construction as this document’s v2 — an exact bivariate ReLU covariance plus
- than fed back into the loop — and four negative results on sampling. baghyeonsig (submission
- 328274, 3.04e-07) blends gain-covariance propagation with whitened antithetic Monte Carlo.
- What I believed. Plain Monte Carlo has error sigma^2/k at cost proportional to k, so the product
- What it cost. The leaderboard shows it directly. Leading Phase 2 submissions sit at a compute
- 2. The bundled ReLU covariance is the first term of a series
- The bundled estimator approximates the post-ReLU covariance by a gain factor, cov_post[i,j]
- k=1 form understates the covariance by 4–26% there, and the series has converged by K=3 for
- rho up to about 0.8 (at rho = 0.9 K=3 is still 2% short and K=6 is needed for 0.1%). K=4 costs
- (data/mlp0.npz), and the “true” mean and covariance rows are themselves 10^6-
- The propagated covariance is wrong by 28% in relative off-diagonal terms and it does not matter.

## Topic 18200

- A Fixed Gain-Covariance and Whitened Antithetic Monte Carlo Blend for WhesTBench
- Submitted method         0.75 gain-covariance propagation + 0.25 whitened antithetic Monte Carlo
- The algorithmic-contribution rules do not require placement in the score-based rankings. This report evaluates
- the frozen artifact through preregistered experiments, negative-result records, and measured comparisons; it
- random multilayer perceptrons. One branch is the oﬃcial full-covariance Gaussian closure: it propagates means
- and covariances analytically, applies exact marginal ReLU moments, and approximates oﬀ-diagonal post-ReLU
- covariance by a gain product. The second branch is a whitened, antithetic Monte Carlo estimator adapted from
- 4.050381542 × 10−7 for covariance propagation to 3.044010045 × 10−7 , a 24.846% reduction, with zero failures
- The closed-form MSE-minimising covariance weight was 0.7341 and the measured grid optimum was 0.735, while
- the preregistered shipped weight was 0.750 and cost only 0.118% relative MSE versus the measured optimum.
- The Monte Carlo branch was 2.652 times worse than covariance propagation alone, yet its low error alignment
- The development record includes one scientiﬁc negative and three operational failures directly relevant to
- the method. A full-strength second-order Gaussian Hermite correction worsened paired ﬁnal-layer MSE by
- Contribution. Covariance propagation, whitening, antithetic sampling, and convex averaging are established

## Topic 18183

- probability ½                   probability ½                                negative half accumulates at zero
- ReLU leaves every positive coordinate untouched and replaces every negative
- both positive                                         one negative                                               both negative
- have one negative coordinate; dark-purple points have two. Correlation between real first-layer coordinates leaves this
- their joint arrangement. Covariance would distinguish this toy pair; deeper networks can require still richer information than
- covariance alone.
- A standard Gaussian coordinate has ordinary average zero. Positive and negative draws cancel in that
- All 256 squared contributions are nonnegative, so they add without cancellation and remain fairly
- whenever r is nonnegative.
- negative coordinate remains on the zero side of its gate.
- Changing U to −U is also a change of direction, since negative scaling can reverse many preactivation signs.
- produce positive coordinate products, their disagreements produce negative products, and the complete
- read geometrically by turning their two symbols into positive and negative signs.
- product may be positive or negative, while its absolute value is always 1/16. Its squared overlap with each old

## Topic 18188

- When compared to the classical methods of quadrature in one dimension, this error scal-
- (i.e. trapezoid, Simpsons) rules. More sophisticated methods like Gaussian or Clenshaw-
- The generally outstanding problem is that the naive transfer of these methods to dimen-
- simple method seems attractive again.
- The naive Monte Carlo quadrature is essentially the ”method of complete ignorance”.
- own, but they may yet find a place as one part of an algorithm that does. While composite
- Rℓ )F ] which could be approximated by this method, or something similar.
- and covariance of ⃗xℓ , the variable ⃗zℓ has mean ⃗ηℓ = Wℓ µ⃗ ℓ and covariance Sℓ = Wℓ Cℓ Wℓ⊺ , and
- the mean and covariance through the nonlinear layer functions.
- where σα =        Sαα and rα = σηαα . The covariance of the next layer is then, in terms of
- derivation is described in Appendix A along with definitions and details.
- The Gaussian closure is biased, but relative to its cost and simplicity it is a remarkably good
- sampling? Essentially, use the Gaussian approximation as a control variate. If there is
- In any case, I don’t think any control variate that tries to cut directly through 32 layers

## Topic 18157

- cumulant propagation, found partial k=3 corrections can be worse than the Gaussian
- nel, every fixed nonnegative spherical rule with ≤66,048 support points has normalized
- 0.2351% against a 4.05% cost surcharge — dead. The explicit carve-outs matter as much
- discipline: α = 0.035 saved 2.3% arithmetic but cost 13.3% MSE (product 10.6%
- development-only gains that reversed held-out; 0.020 retained. Their negative-results
- lines: ratio 1.0000203, with the optimizer returning no negative weights; shared-
- samples, sample count sized against a conservative no-pruning worst-case cost model,
- (1.7292e-7 vs 1.7275e-7). “A lower local cost proxy did not transfer to a better official
- whitening: cheaper and plausible but unshipped), and control variates, higher-order clo-
- • 5c/5d (the mid-network control variate, now fully documented at the published
- ance, marginal third/fourth cumulants, a paired (2,2) fourth, and an eight-generation
- unbiased control variate,” with the error identity ‖e_CV‖² = ‖e₃₁‖² − 2ρ·(alignment) +
- score is S = ē·q̄ + Cov(e,q) — the error–multiplier covariance is a real score channel
- al. K=3 cumulant backbone but replaces the fourth-cumulant projection: κ₄ is regener-

## Topic 18185

- The useful way to think about our final method is as a combination of two descriptions of the
- This suggests a deterministic algorithm: carry the mean and covariance through each linear
- ReLU formula. We use a simple gain-×-Φ rule for the off-diagonal covariance.
- Gaussian. This gives a useful control variate.
- keep mean and covariance;
- covariance.
- Korobov inputs throughexact-L1 control variate
- D            vector layer-1 control variate               5.287 × 10−7
- The negative results matter because they tell us which object is wrong. We used small canaries
- and held-out gates before promoting a method.
- Can we transport κ3 and               Oracle cumulants contained in-      Higher moments were not useful un-

## Topic 18177

- A Learned Cumulant-Transport Propagator and Its
- Measured Ceiling
- Algorithmic contribution entry — submission ID 327838
- cumulant plane state, and a recurrent per-neuron latent through all 32 layers, with a learned
- why. Four results: (i) the exact O(N 3 )-state cumulant chain bills 1.4–3.4× the entire compute
- of what a 0.26B dense learned head buys at ∼0 cost; (iii) no learned corrector family we
- means and covariance injected at every layer the family bottoms out at 8.1 × 10−7 raw — which
- adjusted metric, because both methods already sit near the compute-multiplier floor where
- two, and the walls above say it is not reachable. We also report, as a negative result about our
- numerically inert at this operating point while consuming 61% of the billed cost; deleting it is
- ablation, cost attribution and lean rebuild of §4–§5, were produced by Claude Opus 5 from the shipped artifact.
- multiplier floor at 0.1 means a biased but cheap deterministic method competes on even terms with
- [9] recently showed that transporting cumulants with Hermite expansions estimates the expected
- standing in for the O(N 3 ) third-cumulant tensor whose exact transport bills 1.4–3.4B.

## Topic 18182

- trim_qewas · companion to “Measuring the Ceiling of a Method Class” · submission 326725
- II.E Learned and amortised methods — closed at seven students plus a diagnosis   9
- II.G Budget and cost tricks — all empty                                          10
- III.2 Two traps κ cost us, both repeatable                                       11
- A. Ceilings measured (26)                                                    17
- F. Budget, cost, competition (30)                                            22
- G. Process & methodology (1)                                                 23
- Companion volume to Measuring the Ceiling of a Method Class. That document argues a thesis; this one is the
- ~5.5 MB of accumulated notes across 109 working sessions. The great majority of it is negative results,
- and that is the point: we spent the project mapping the boundary of a method class rather than tuning
- The central instrument of the project. Rather than build a better method and see, we substituted ground
- the step formula (exact per-step error)            1.19       1.032    ceiling of a whole class
- linear correction in the closure's own basis       1.027      1.001    ceiling of a whole class
- true OFF-DIAGONAL of C                             1.149      1.012    the covariance channel is nearly empty

## Topic 18182

- Measuring the Ceiling of a Method Class
- A family of estimators has a ceiling, and you can measure it without building a
- One such measurement costs about fifteen minutes. We ran the protocol 54 times
- random ReLU networks — and it closed 54 method families, most of which we would
- substitute the exact …           ceiling    what it closes
- mean and covariance               ×3.36    the entire two-moment class
- covariance alone                  ×1.10    every "better covariance" idea
- chose does not contain a winning method, and we can say that with a number rather
- above is the reason: we spent Phase 1 inside a class whose ceiling we eventually
- The report is organised around the protocol (§2), the ceilings it produced (§3), the
- functions integrate exactly, and the ceiling of any additive one-dimensional class is
- than decorate it, and the classification is the point: most of what follows is a negative
- result, and a negative result is only as good as the check behind it.
- about it first. Closed. — a branch closed, with the ceiling and the pre-registered

## Topic 18181

- ARC White-Box Estimation Challenge 2026 - Phase 1 Algorithmic Contribution
- Antithetic pairing, an alternative Roberts generating vector, and an expected-gain control variate all worsened the
- The contribution here is the controlled characterization of this estimator under the current deep-MLP grader and cost
- model, together with negative results that delimit the useful operating regime.
- The final method was reached through 28 submissions in the Phase 1 submission history. The campaign began with
- analytic mean/variance propagation and full-covariance Gaussian closures, then moved through Hermite/Price
- covariance corrections and hybrid analytic-RQMC estimators. Those hybrids reduced error substantially but retained
- The most useful lesson from the campaign is methodological: the official metric must be treated as an objective in its
- computation must justify its cost under the grader rather than only improve MSE.
- 5. Structural ablations and negative results
- control variate, 20k
- covariance
- construction. This was one of the largest negative results in the campaign and shows that rank-1 lattice quality is
- 5.3 Layer-1 control variate

## Topic 18175

- n = 256. ARC’s cumulant propagation algorithm is one such method. It belongs to the general class
- the K = 3 cumulant propagation algorithm, we carry a mean, a covariance, and a structured third
- cumulant; in addition, we regenerate the fourth cumulant from at most four covariance-response modes,
- model is a Gaussian whose covariance is modulated by background fluctuations, following the idea of
- super-sample covariance in cosmology. The submitted estimator also includes corrections for third-
- 77.2% relative to ARC’s factorized K = 3 algorithm and by 48.2% relative to factorized K = 4;
- the submitted method therefore remains less accurate than the sampling references used here.
- covariance-propagation methods (Frey & Hinton 1999; Wright et al. 2024), but every ReLU creates non-Gaussian
- dependence. Wu et al. (2026) address this with cumulants and Hermite expansions. Linear layers transport the repre-
- sented cumulants exactly; nonlinear updates create new terms; and a structured projection keeps the state tractable.
- The cumulant estimator of Wu et al. (2026) degrades as network depth increases. Each ReLU reopens the cumulant
- mean and other cumulants, after which the remaining layers transport and compound the local errors. Wu et al. (2026)
- conjecture error of order cK (L/n)K . Their Appendix D tests the depth dependence by overlaying predicted LK slopes
- plain Monte Carlo by more than an order of magnitude. A dense fourth cumulant has O(n4 ) entries, while factorized

## Topic 18175

- integration, a shallow analytical reference, a fitted inter-layer control variate, and compute-aware
- execution. Technical details and evidence limits are in the Appendix.
- • A fitted inter-layer control variate that transports the layer-9 sampling discrepancy to correct the final-layer
- Section 2 gives the measured results. The Appendix gives the construction and statistical qualifications.
- accuracy. The adjusted score combines final-layer MSE with effective compute; lower values are better. Appendix B
- a layer-9 discrepancy. A fitted inter-layer control variate transports that discrepancy to correct the final-layer sample
- ancy that reaches the final layer. Appendix A.8 gives the ridge fit, the submitted terminal-bypass branch, and the
- limitations of this model-assisted control variate.
- Appendix A.3 and Table 2 give the exact differences.
- Appendix B.1 gives the protocol and qualifications.
- 3. CONCLUSION AND FUTURE WORK
- before evaluation and tested on untouched networks. Appendix B.2 gives the relevant qualifications.
- APPENDIX
- This Appendix describes the shared numerical estimator and the configuration differences between #327950 and

## Topic 18176

- ment and measured cost/error ladder explain downstream com-                                  2𝑁 𝑖=1 𝑊 𝑖
- Every coefficient is nonnegative. Composition preserves this               3. Controlled finite-width study
- negative mixture at every depth.
- matched raw-cost product.
- width; and the live-anchored cost ratio was 0.999895.
- Full covariance whitening moved the cloud and worsened raw
- The dominant cost is propagating 65,536 states through 31 non-
- path-compatible exact identities                            support-certified cost removal
- Algorithm 1 · Structured angular mean
- H1 covariance whitening/recoloring exact H1 mean/covariance geometry                                    2.65× worse raw MSE
- Four/eight-component gate cubature conditional means and covariances                                    about 26× worse raw MSE
- The second is matched-width support versus magnitude/RMS                               10. Conclusion

## Topic 18106

- 3   Method: A Structure-Aware Hybrid Estimator                                                                    5
- fall on either side of the ReLU threshold at zero. Neurons that are negative for nearly all inputs (“dead”)
- covariance mass concentrates into a small, tightly coupled core of neurons, so the late layers are far more
- These observations motivate a hybrid estimator that combines analytical methods with strategic
- asking early how much of the method is tied to the particular architecture it was tuned on. The challenge
- 1. A sampling method that follows the structure of a ReLU network. Section 3 presents the estimator
- 3     Method: A Structure-Aware Hybrid Estimator
- Before choosing an algorithm, we selected a few example MLPs and drew a large number of Monte Carlo
- pass consumes, so every probe sample is reused and classification costs no extra draws.
- The estimator is not a pure cumulant-propagation or covariance-propagation method [26, 27]. Instead, it
- products are evaluated. We hope our methods can serve as a sampling baseline for more sophisticated
- Both ablations report the effective rank of the activation covariance, which we define here. Let
- Σℓ ∈ Rn×n be the empirical covariance of the layer-ℓ activations over the sampled inputs, with eigenvalues
- their inputs. This distinction is important when designing diagnostics. A method that declares neurons

## Topic 18174

- Phase 1 Algorithmic Contribution — Submission 326960
- 7. The cost model assumes the worst case—no pruning—so the production path remains budget-safe.
- The principal lesson is that variance reduction and cost reduction must be designed together. Exact identities
- effective cost = instrumented FLOPs + lambda * residual seconds
- compatible with a worst-case cost bound. The implementation does not use MLP names, public instance
- positive margin is evaluated linearly; a column with a conservative negative margin is set to zero; all
- result with an ordinary product and found identical values. This is a cost-model reduction of the same
- The sample count is sized against a conservative no-pruning cost model. Numerical safeguards retain a finite
- The estimator separates value-preserving identities from decisions that trade a certified risk for lower cost:
- The cutoff-80 result is an important negative result: a lower local cost proxy did not transfer to a better official
- Earlier structural split            Small cost saving but worse held-out accuracy                                      Rejected
- because it improved the cost/runtime tradeoff without changing the estimator’s structural decisions.
- Other exploratory directions—including control variates, higher-order analytic closures, and alternative
- pilot and can therefore trade a small amount of bias against cost. Strassen is beneficial only in sufficiently large

## Topic 18173

- Phase 1 Algorithmic Contribution                                                                 Submission 327651
- Hadamard identities, reducing cost without changing the estimator.
- Phase 1 Algorithmic Contribution                                                             Submission 327651
- This is a cheap, known correction. Exact covariance transport helped an earlier sampling chassis but hurt the
- Phase 1 Algorithmic Contribution                                                             Submission 327651
- Apply the BCH structure with exact counted algorithms
- ucts uses the public flopscope.numpy interface and is charged by the same official cost model as any other
- Phase 1 Algorithmic Contribution                                                             Submission 327651
- sponding negative rows.
- Fused L0 covariance + cost prune             1.69e-7         3.36e-7 transport exact L0 mean/covariance;
- Phase 1 Algorithmic Contribution                                                              Submission 327651
- The initial direction was analytic cumulant propagation. K=2 covariance propagation was useful, and higher-
- order diagnostics taught me why it failed at depth: important off-diagonal cumulants grow through the ReLU
- I then tried Sobol and orthogonal QMC tables, radial rules, learned residuals, control variates, restart estima-

## Topic 18171



## Topic 18170

- double sample work. The sample count cancels, leaving only the variance per input, the cost per input,
- roughly stable bias, linear marginal cost, and operation above the utilization ﬂoor. It is not speciﬁc to
- cumulant-aware moment closure or to this particular anchor.
- setup cost diluted by samples
- Increasing γ earns a linear reward by removing shared sampling error, but pays two quadratic costs:
- Extending the deterministic cumulant-aware moment pass described in Section 4.1 through all 32 nonlin-
- matched low-order Gaussian moments, paired inputs with their negatives, and replaced the sampled
- antithetic sampling [2, 3]. Covariance matching produced the largest early gain. Sign pairing and exact
- population mean. A layer-8 control variate improved an archived adjusted score from roughly 3.42×10−7
- control variates [4].
- with control strength 0.6 was the better accuracy–cost compromise.
- cumulant-aware moment pass described in Section 4.1, then adds the frozen recurrent model’s correc-
- tion. It chooses between 6,000 and 8,600 inputs, constructs a covariance-matched antithetic cloud, pins
- cost of the sampled matrix products [5].

## Topic 18166

- 8.55 × 10−8 ); the best published, method-disclosed competitors score 3.7× and 16× worse. We
- Finally, we report cost-model findings we believe are of independent value to the organizers: a
- dtype-billing interaction worth a deterministic 2.00× on the score, a residual-time law (cost ∝
- 3 Method                                                                                                5
- 5.3 Calibration against other methods . . . . . . . . . . . . . . . . . . . . . . . . . . . .        12
- 6.3 Design growth: capped by the budget, correctly costed . . . . . . . . . . . . . . . . .          15
- 6.5 Negative results with mechanisms . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       15
- 8 Observations on the evaluation methodology                                                           17
- Our submission is a cubature estimator: a sampling method whose sample locations are chosen
- radial dimension of the problem costs nothing.
- operation-level cost model (flopscope), which turns implementation into an optimisation problem
- under heavy host contention, analysed in Section 7). The best published, method-disclosed
- cumulant chain) behind (Section 5).
- partner MSE M ≤ 9V ≈ 1.85 × 10−6 at negligible cost; the organizer-referenced cumulant

## Topic 18149

- Private Phase 1 algorithmic-contribution packet
- • https://discourse.aicrowd.com/t/phase-1-update-flopscope-v0-10-0-cost-model-f
- • https://discourse.aicrowd.com/t/algorithmic-contribution-prize-guidelines-how-a
- 4. measured evidence for the combination, including negative results that bound what the mecha-
- 6     Negative evidence and falsiﬁable limits
- sures on a fresh MLP                                          ceeded the rank-ﬁve raw ceiling
- Table 4: Selected negative results after e141. Each killed family remained unpackaged and unsub-

## Topic 18154

- Working draft for the ARC White-Box Estimation Challenge 2026 — Best Algorithmic Contribution.
- (dtype-aware cost model), and again with the 2026-08-03 update (flopscope 0.10.0 re-evaluation + one-core
- Companion artifacts: method-deep-dive.md (mechanism + full derivations), research-note.md (chronological log),
- pass, which is in turn the §9 intra-network control variate plus the §9 compute-composition bundle, and
- exp_*.py / _d_*.py (every ceiling and frontier experiment).
- cumulant series diverges, contraction is impossible in a critical network, truth-injection is variance-
- bounded). Each of ~20 measured-dead methods is one of these three exits closing.
- activation — already produced by the Monte-Carlo pass — is an intra-network control variate whose true
- mean the analytic propagation supplies for free; it costs no second forward pass and, unlike input-space
- stratification, survives depth. This is exactly the "clever sampling method resting on an interesting
- unreachable. The public board proves a low-bias method is findable; we locate it outside the per-
- confirms for free, one that survived two cost-model re-evaluations and a runtime migration, and one that
- propagates the first three moments of each layer's pre-activation distribution (mean, full covariance, third
- cumulant), closes each ReLU analytically with a Gram-Charlier expansion, augments the third cumulant with a

## Topic 18097

- Stabilizing Cumulant Propagation at Depth 32: A Trajectory-
- observations: step-exact Edgeworth corrections computed from true cumulants de-
- ARC’s published cumulant-propagation algorithm (“kprop” [1]) is accurate at depth
- layer while its covariance errors partially compensate: re-anchoring the covariance
- cumulants degrade the chain. We argue this compensation structure is the reason
- C2 — Method: trajectory-fitted stabilization (§2). Fitting per-layer linear cor-
- C3 — Ground-truth joint-cumulant tests (§4). Against keenanpepper’s published
- full third-cumulant tensors [2] (Hugging Face dataset keenanpepper/whest-k3-
- still 4.5× worse than the low-cost MC probe already feeding the estimator, on 10 of
- alone makes the result worse), with a measured denoising ceiling (0.48–0.58×) and
- C5 — Negative results with mechanisms, and an open problem (§§5–7). Each
- Sampling methods win Phase 1’s raw metric, and this document says so plainly. The
- from deployment (§8). The diagnosis, the stabilization method, and the error budget
- tor_chain.py, a single-file fnp estimator implementing the method of §2, i.e. a

## Topic 18097

- Stabilizing Cumulant Propagation at Depth 32: A Trajectory-
- Erratum (2026-08-04): grader cost-model repricing under FLOPScope
- graded all affected submissions in place. The cost model is now dtype-aware — float64
- fore none of the claimed contributions — the diagnosis (C1), the stabilization method
- (C2), the ground-truth cumulant tests (C3), the error budget (C4), or the negative
- the contribution claimed for this artifact is mechanistic, not positional, and the cost
- conclusion — well short of the sampling frontier — is unchanged, and indeed strength-
- while the frontier entries sit at the multiplier floor and barely moved. The §5 ceiling
- observations: step-exact Edgeworth corrections computed from true cumulants de-
- ARC’s published cumulant-propagation algorithm (“kprop” [1]) is accurate at depth
- layer while its covariance errors partially compensate: re-anchoring the covariance
- cumulants degrade the chain. We argue this compensation structure is the reason
- C2 — Method: trajectory-fitted stabilization (§2). Fitting per-layer linear cor-
- C3 — Ground-truth joint-cumulant tests (§4). Against keenanpepper’s published

## Topic 18097

- Stabilizing Cumulant Propagation at Depth 32: A Trajectory-
- Erratum (2026-08-04): grader cost-model repricing under FLOPScope
- graded all affected submissions in place. The cost model is now dtype-aware — float64
- fore none of the claimed contributions — the diagnosis (C1), the stabilization method
- (C2), the ground-truth cumulant tests (C3), the error budget (C4), or the negative
- the contribution claimed for this artifact is mechanistic, not positional, and the cost
- conclusion — well short of the sampling frontier — is unchanged, and indeed strength-
- while the frontier entries sit at the multiplier floor and barely moved. The §5 ceiling
- observations: step-exact Edgeworth corrections computed from true cumulants de-
- ARC’s published cumulant-propagation algorithm (“kprop” [1]) is accurate at depth
- layer while its covariance errors partially compensate: re-anchoring the covariance
- cumulants degrade the chain. We argue this compensation structure is the reason
- C2 — Method: trajectory-fitted stabilization (§2). Fitting per-layer linear cor-
- C3 — Ground-truth joint-cumulant tests (§4). Against keenanpepper’s published

## Topic 18151

- the residual term costs per call
- The v0.10.0 announcement asked for feedback on the cost model, and left one question explicitly open
- method, aimed at both.
- methods with one measurement and no submission.
- Gaussian-closure estimator (t/18063) is the clearest example, and moment/cumulant propagation is
- the obvious mechanistic approach to a problem that hands you the weights. Every such method,
- whatever order of cumulant it carries through the depth, has to end in the same place: a readout that
- the same, plus the analytic third-cumulant Edgeworth term                  3.38e-06
- compute spent propagating them more accurately moves it. A closure method cannot reach 8.8e-07
- from above by working harder; that is the ceiling of the whole approach.
- cumulant term multiplies the error by 3.8x. The skew correction is not small relative to the base error,
- natural "just carry one more cumulant" repair fails here, and it fails in the direction that looks like
- 2. The cost model has a second term that penalises exactly what the first
- Both of those are about residual as a channel to be exploited or a cost to be avoided. Our contribution is

## Topic 18151

- the grader, against 8.37e-6 for the bundled covariance-propagation baseline — a factor of 40, obtained
- Everything else we built is worth nothing. Measured at equal cost — 6,178 forward passes per MLP
- expected to win, and we report every conclusion we got wrong the first time together with the bug or the
- budget axis is flat and the cost axis is not. We priced twenty-six operations under flopscope on one core
- last digit; billing follows the output dtype and bottoms out at float32, so float16 costs exactly what
- float32 costs while running 110x slower; and the symmetry system prices a declared symmetric operand
- last line is a standing invitation to moment-propagation methods and worth nothing to a sampler, and
- it is the clearest statement we can make of where this cost model wants the field to go.
- We derived a ceiling on everything in this note, withdrew it, and then found the
- three networks that came to 37.3%, an equal-cost bound of 1.34x — which our own deployed estimator
- 26.1%. The equal-cost bound is 1/(2 x 0.2611) = 1.915x and the deployed estimator is at 1.731x against
- the same baseline: the ceiling is real and we are at 90.4% of it.
- prize would be 1.286x, against an equal-cost gap to the field of 1.73x to 2.61x . Korobov moduli, CBC
- buying cost instead of variance by carrying a rank- r factor needs rank 196 at layer 24 to hold 1 - 1e-6

## Topic 18063

- falsification experiments (§6) are the ablations and negative
- second-order Gaussian-closure estimator (covariance propagation,
- additional FLOP cost — reduces final-layer MSE by ~3× over the
- covariance baseline. Directly against the organizers’ own
- reference cumulant-propagation code, this zero-cost correction
- matches or beats full third-cumulant (k=3) propagation at the
- benchmark’s depth (L=32), at 1/500th the cost.
- the covariance baseline, but it does not beat the challenge’s
- the finding — that a large, clean, zero-cost-correctable component
- covariance Σ layer by layer, applying exact per-neuron post-ReLU
- standard covariance-propagation baseline.
- 2.66e-6 vs 7.52e-6 for the uncorrected covariance baseline on the
- same networks (2.83× improvement), at identical FLOP cost. The
- (+0.005, symmetric — 47% of neurons negative, 53% positive), so


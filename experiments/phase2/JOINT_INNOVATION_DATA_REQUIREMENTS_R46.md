# Joint state innovation: what the corpus can and cannot identify

2026-10-13. H18 remains the anchor; no tenfold stage has been reached. Fixed
learned corrections of total mean/variance/K3/K4 state error worsen actual
closed final MSE. Training-CV marginal improvements do not validate a rollout.

The train24 mean innovation decomposition r43 and independent OOF replay r45
identify one cause. Given post error e_l and frozen baseline mean Jacobian
J_l=diag(Phi_l)W_l.T, define z_l=e_l−J_l e_(l−1). Total state error contains
inherited history. Adding its prediction at every layer repeats that history.
R45 train-CV actual-weight replay at layer14 yields local-innovation ratio
.99420693 versus repeated-total-error ratio1.12483026. These are a frozen
first-order diagnostic, not a nonlinear final-layer candidate score. The
local-innovation gain is too small to justify further full mean-only rollout.

## Full joint innovation cannot be recovered from marginal MC labels

For variance after a linear layer, true pre covariance is W.T C_true W.
The diagonal variance transport error is

```
diag(W.T ΔC W)_i
  = sum_j W_ji² ΔC_jj + sum_(j!=k) W_ji W_ki ΔC_jk.
```

The 14k MC corpus contains post univariate raw powers only. It provides the
first term, but no true off-diagonal ΔC. Using `(W*W).T @ delta_variance`
and calling the result full variance innovation would incorrectly attribute
unknown transported covariance error to a local closure. Teacher k22 factors
are approximate teacher states, not true covariance or cumulant labels.

Likewise third/fourth pre cumulants depend on all joint entries contracted by
W**⊗3/W**⊗4. MC K3/K4 diagonal errors alone do not identify inherited joint
transport. Parent low-rank source representations contain an approximation to
those entries; plugging them in yields a defined approximate decomposition,
not observed true local innovation. A learned model trained on that residual
must retain this attribution limitation.

## Three scientific next gates

| Gate | Actual information required | Before → after hypothesis | Main limitation |
|---|---|---|---|
| Target-free directional graph features | Existing baseline C_off/D21/wk4m/g4row and weights, no labels in feature construction | Original22 scalar features → fixed cov-row contractions and source-coupled descriptors | Needs a new actual fullstate capture on train networks; marginal truth permits error prediction, not exact joint innovation |
| Independent probe-coordinate MC | New seed-disjoint training networks, MC pre/post moments for a preregistered small neuron subset and contracted next-layer projections | Unidentified covariance transport → directly measured projected true variance/K3/K4 along actual W columns | MC bake needed; finite noise/cost must be controlled and held independently; no benchmark fitting |
| State-direction tangent JVP | Fully frozen H18 augmented state, permitted runtime graph, deterministic test perturbations of C/source and mean | Mean-only Phi*W tangent → actual joint algorithm derivative linking trained corrections | Derivative is of the approximate algorithm; it does not supply missing true full-state labels |

For the first gate, choose a small fixed feature set before any held network
score. Examples are `sum_j C_off_ij²`, signed and squared D21 row contractions,
standardized C_off-to-next-layer-mean projection, and existing wk4m/g4row
trace mismatch. Keep actual dimensions, canonical permutation and birth age
semantics. Values from current training labels must not be used to decide which
features to extract. A train-network CV gate checks residual predictability;
zero or very small signal stops the model before cloud rollout. R41/R42 show
that richer static regression alone does not establish closed-loop benefit.

For a projected MC gate, measure the same actual scalar linear projections
the closure needs rather than storing all n²/n³/n⁴ truth. If h=W_next[:,i],
samples of h·post_activation identify pre mean, variance and higher cumulants
along that direction, including off-diagonal structure. Use separate streams
to estimate target noise and the cross-corrected diagnostic, and enough
networks that validation is by whole network. Preserve reference identity and
precision. MC data may form offline labels only; the final runtime candidate
must regenerate its correction from weights/learned data with flopscope.

No further expensive full rollout or MC bake is launched by this design file.
It records why diagonal marginal innovation should not be mislabeled exact,
and the minimum missing evidence needed for the next mechanism.

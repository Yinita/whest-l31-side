# R37 canonical joint diagonal/trace projection

Frozen H18 parent `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`; R34 base `83d81b99c7bd38db184686d63ddbe5b39402c3bff0597289ab10c34aa6665799`.
Candidate SHA256 `637212f82ca4906a3b6524623c64bb0bb0eff5712ae592e25708db93fd48a469`.

R34 kept the parent harmonic projection and repaired the combined D4 with a diagonal-support residual. This changed the trace, since parent G also carried some D4 contribution into its cross support. R37 makes one model-internal algebraic change: solve both produced ordinary D4 and produced K22 row sums jointly at every birth. No data-dependent rank, age, layer or scale scan; no reference access.

Let Eoldpost=Σ c qpost^4 be the inherited exact residual. Its cross-row trace is `crossOld=qpost²@(c*colsum(qpost²))−rowsum(c*qpost^4)`, evaluated with metered O(n*columns) work. The required proxy cross-row target is `s=producedK22@ones−crossOld`. In the parent postbirth identity metric S(G,I), cross-row sums are `((n−2)diagG+trG)/6`; hence

```
g_prev = 6/(n−2)*s − 3/((n−2)(n−1))*sum(s)
cnew = ordinaryK4v − g_prev − D4(Eoldpost)
```

The new source is diagonal support only. The combined postbirth H=S(G,I)+Eoldpost+diag(cnew) exactly preserves both ordinary D4 and K22 row trace. Parent off-diagonal lambda covariance component remains; it does not affect postbirth D22 when the metric is I. This preserves only the trace constraint, not every K22 entry or all-distinct tensor. Parent USE and approximate feed plus exact residual USE/fullfeed share the same lifecycle and M-residual repeated-support subtraction as audited R34.

Suite riders branch only; generic parent fallback unchanged. Candidate arithmetic uses permitted flopscope operators, no truth correction, counter manipulation, parallel prediction or compiled kernels. Full residual histories retain neuron rows; last reads D4/feed D3 only.

`results/phase2/tenfold_joint_trace_d4_production_contract_r37_r01.json`: saved n32/depth4 diagnostic body copy with only riders=True literal flag, inverse replacement proves body unchanged; original/snapshot hashes separately recorded. Residual USE<=7.51e-8, complete residual feed/marginal<=1.85e-8, combined post D4<=3.07e-8, K22 row trace<=2.05e-7. Diagnostic copy is not the submitted estimator or 1024 resource acceptance. R34 has negative strict score, so R37 must establish whether trace inconsistency explains it; no accepted score claim yet.

The new g_prev is the fixed affine constrained split, not the former orthogonal harmonic projection. Frozen parent LAM/REF_R and its adaptive lambda formula remain; their automatic response to changed g_prev is inherited model behavior, not a fitted new coefficient.

Independent production-body ledger PASS: `results/phase2/tenfold_joint_trace_d4_r37_ledger_audit.json`, n32/depth4 diagnostic flag-only copy, frozen candidate hash and reversed-body equality. Includes both-constraint recovery, oldcross trace subtraction, full residual feed and parent M residual ledger. This does not establish whole1024 benchmark performance.

Strict1 finished, zero failures/resource pass: FLOPs1,903,631,419,511; raw2.2737705052122692e-8 vsparent2.2281179568039988e-8 (+2.04893%); adjusted1.9683379716659492e-8 vsparent4.8714160493622575e-9 (+304.059%, 4.04059x). No promotion/expansion/submission or cost optimization. Joint trace correction recovers from R34 raw4.779e-7 to2.274e-8, but does not outperform parent. Sameinput targetfree comparison shows first transported birth deltaRMS5.11e-6 (R34 9.80e-5), final1.66e-5 (R34 6.65e-4), strongly attributing catastrophic earlier failures to trace inconsistency. This does not establish a precision improvement. CP4 D4 projection branch is closed pending a scientifically distinct coupled-state signal.

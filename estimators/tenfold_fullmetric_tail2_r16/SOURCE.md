# R16 penultimate full-metric K4 and complete one-step feed

Parent V37bH18 SHA256 `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`.
Candidate SHA256 `a1672cc0d2dedaa65eb668213612c1b47855ec46102bb0cb35721c6df1b7cfce`.
Frozen single primary mechanism: replace only the penultimate pre-ReLU K4
proxy's average metric by the actual current public weight metric, including
all D4/D31/D22 USE and its complete all-distinct K4->K3 feed, transported once
to the final layer. Earlier layers and parent K4 birth/projection remain.

At the activated layer, parent G=`diag(dG)+lambda*C_off` remains approximate.
It is symmetrized and paired with actual B=`W W.T`. The tensor is average-Sym4
G tensor B (six pairings/6). The same G/B supplies D4, D31 and D22 to the
nonlinear power-cumulant graph, and all three official CP feed groups:

https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py#L410-L454

The old parent average-metric feed is disabled at this unique birth. The
external feed stores the three full CP groups plus a correction CP(M,I,I)
with negative M, where M=diag(feedD3)+3*feedD21.T. This removes repeated
entries exactly, before any parent rank-r residual compression. Parent ordinary
M residual still supplies the nonlinear K3 diagonal/pair birth; no full-feed
repeated subtraction passes through the rank16 residual approximation. The
external all-distinct channel moves by raw W once (it is already post-ReLU)
and adds its D3 to the final mean path. It retains all neuron rows, so pruning
of parent sources is still approximate but does not silently prune this channel.

Generic depth>=3 runs activate the same gate for independent smallshape
contracts; depth<=2 does not activate because no K4 input exists at the
penultimate layer. Parent NO_CORR default remains true. FULL_LAST/NO_FEED/
NO_SRC_LAST nondefault probes are outside strict submitted configuration.

Actual estimator n8/n32 contracts compare captured G/B with independent dense
H4 for D4/D31/D22; expand actual stored four CP groups to verify zero repeated
birth; then compare actual final external D3 addition with independent raw-W
transport. Max float32 error `1.17e-7`, birth repeated leakage≤`6.51e-8`, final
transport≤`5.30e-8`. Evidence:
`results/phase2/tenfold_fullmetric_tail2_production_contract_r02.json`.

This candidate does not produce a complete dense K4 core or use all K211
cap terms for harmonic birth. Parent memoryless diagonal/lowrank proxy and
LAM/REF_R remain. It tests a local missing-metric mechanism with coherent
all-distinct feed, not complete official K3augment/trueK4 propagation.
All computations use allowed metered flopscope operations. No target/reference,
fitting, oracle, compiled kernel, concurrent prediction or meter intervention.
Strict1024 score/resource evaluation is recorded below under original limits.

Generic smallshape contract uses the parent's non-regeneration fallback core
`diag(g4row/METRIC_C)` because no Gaussian-trained suite tables activate there.
It checks actual candidate representation/transport algebra, but this fallback
need not reconstruct the parent's independent fallback D22 coefficients. Real
1024 uses the frozen regeneration G=`diag(dG)+lambda*C_off`. The current B is
computed from the reordered full W, even where old parent sources use pruned
blocks. G is explicitly symmetrized; parent's float32 covariance transport can
have tiny antisymmetric rounding residual, which is outside a symmetric K4
tensor and is removed. This rounding cleanup is recorded and not classified
as a separate algorithmic score improvement. K4 g4row division by METRIC_C
occurs only in the generic fallback, never in the 1024 regeneration path.
The generic non-regeneration graph does not necessarily read wk431 even
though this candidate computes/captures its full D31 object. Small contracts
verify tensor slice and source algebra, not all retained nonlinear diagram
outputs. The real1024 regeneration graph includes wk431 in b2 and uses the
computed full D31 there; real1024 execution was evaluated by the root strict run.

Root strict1 subsequently completed with zero failures: raw
`2.2281179568039988e-8 -> 2.2318463522e-8` (+0.16733%); adjusted
`4.8714160493622575e-9 -> 5.1632267972e-9` (+5.9903%). Candidate FLOPs
`508729276534`, predict wall `44.6090s`, residual `0.193475329s`.
This mechanism is rejected; no tail-layer sweep or score-irrelevant fusion
optimization candidate follows. Small algebra/component evidence remains.

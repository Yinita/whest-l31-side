# r25 sparse D4 history read-only integration review

Candidate owned by tenfold_k4; no candidate file is edited by this reviewer.
Reviewed initial diff against frozen V37bH18 c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2.
This is a design/read-only algebra and lifecycle audit, not a contract result,
resource acceptance or benchmark score. Actual final candidate hashes/contracts
are recorded by its owner and root.

The represented K4 is H=Σ_r c_r q_r⊗q_r⊗q_r⊗q_r. The independent K3 feed history
is F=Σ_r Sym(X_r,q_r,q_r), with Sym averaging all six permutations. A correct
nonlinear lifecycle is:

```
Qpost = w1 * Qpre
Xnew = 1.5 * c * w2 * Qpre²
Xpost = w1 * Xpre + Xnew
cnew = produced ordinary K4v − Σ c Qpost⁴
append(Qpost, I), append(Xpost, 0), append(c, cnew)
next linear: raw W applied to Q and X exactly once
```

Xnew is already the post-ReLU feed leg. Multiplying it again by w1 would be
incorrect. The newborn cnew must be appended after Xnew is formed, to avoid
feeding the current newly produced fourth cumulant in the same nonlinearity.
This implementation order is correct in the reviewed initial candidate.

D3/D21 include the prior represented feed once before pK evaluation. The parent
D3_w/D21_w includes these prior feed slices scaled with w1. Newfeed repeated
slices are added once to D3_new/D21_new, and parent S3c/S21 subtract both old
scaled feed and current newborn feed before ordinary M compression. Thus
independent feed history and parent M do not duplicate this represented feed.
Existing thin feedback uses total D21, as intended by the joint closure; this
nonlinear consequence is not an independently tunable coefficient.

The parent harmonic g4row/wk431/wk4m are overwritten rather than incremented.
The old regen feed branch is disabled, so its c1/c2/y/u contributions are zero;
existing parent ordinary K3 birth support remains separate. Remaining parent
harmonic g_prev/K4_sigma calculations are wasted but do not enter overwritten
USE/feed values. Their removal is a possible exact computation optimization
once actual parity is documented. Historical parent beta/mean riders remain
inherited and may have model mismatch; no new empirical refit is introduced.

The independent fullwidth Q/X feed is added after parent saturated-row K3
readout zeroing and does not adopt parent dead-row pruning or confinement. This
scope is explicit; the examined lifecycle is internally consistent. Suite
relabeling W uses current/previous permutations, so full Q/X postleg row order
follows the same current relabeling without truth-dependent choices.

Concrete exact cost reductions sent to owner:

1. Use existing K4 slices for newborn feed repeated slices:
   `new3=1.5*w2*w1²*D4`; `new21=zero_diag(w2_i*w1_i*w1_j*D31_ij + .5*w1_i²*w2_j*D22_ij)`.
   Keep Xnew persistent history, but avoid two extra dense GEMMs per layer.
2. At final mean-only layer, priorfeed D3 is a vector rowsum(X Q²); skip its D21
   GEMMs because no further nonlinearity/state birth consumes them.
3. Optional exact birth-block transport: most recent Q block is I and X block
   zero, so copy W and zero into newblock after multiplying older columns.
   This avoids redundant identity/zero GEMMs but is not needed for the first
   easily audited candidate.

For 16 layers, the dense cost model has Q/X transport k=1..15 (sum120), K4
D31/D22 and oldfeed D21 at nonlast k=1..14 (sum105). Original additional cost
is1800 n³≈1.933T at1024; with newfeed-slice reuse and final priorfeed trim it is
1320 n³≈1.417T. Added to parent≈480.8G gives≈1.898T before elementwise/concat
costs, within2**41≈2.199T in this rough model. Elementwise array traffic, server
peak memory, overhead and120s/.4s clocks require actual strict RPC validation;
the algebra/cost model cannot establish acceptance or tenfold improvement.


Independent actual-production ledger trace has now completed on frozen candidate
SHA `b6fe6506fe37061a9efe48326349100bae18dead466ff6d80657b8e920447438`.
Evidence: `results/phase2/tenfold_sparse_d4_r25_ledger_audit.json`; actual
_predict_core n8/n32 depth4, no reference or target access, all finite. It
checks q/X raw-W once, oldX gain and newX append once, newborn identity/zero,
static c, signed K4birth residual, priorfeed total D3/D21 inclusion once,
newfeed repeated slices against independent full CP3, total D3w/D21w gain,
and S3c/S21 fullfeed subtraction before Rres alias overwrite. Max absolute
ledger error is below2.76e−7; no integration defect found. These tiny trace
costs and correctness checks do not establish strict1024 resources or score.

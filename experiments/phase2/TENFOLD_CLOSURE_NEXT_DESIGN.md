# Tenfold closure: actual r14 candidate and next legal mechanism

2026-10-13. This file covers a production change, with score acceptance pending.
The fixed stage1 gate remains adjusted <5.779e-10. The score multiplier floor
means raw MSE must at least be <5.779e-9; an expensive candidate requires a
still lower raw MSE. Current best V37bH18 remains unchanged until strict evidence.

| Version | Name | Before | After | Evidence/status |
|---|---|---|---|---|
| r14 | Exact heavy-tree K3 history | Parent's rank8 D21 feedback legs form an approximate B1 product; other heavy orientations/K22/doubled-C trees omitted. Previously tested changes restore only one layer/birth. | Every layer restores all cost3 heavy/light official K111 paths. Eight directed heavy/light pairs collapse to a single H matrix. Full independent A/P/H/Z history retains all rows, with exact repeated-birth subtraction. Thin feedback removed to avoid overlap/cost4 product. | Candidate frozen 832bb894…; independent tensor, real cached small-state and FP32 Strassen contracts pass. StrictRPC1 zero failures: raw2.2287540702e-8 vs2.2281179568e-8 (+0.02855%); adjusted1.6548841062e-8 vs4.8714160494e-9 (+239.71%). Negative; stopleg-reuse extension. |
| rejected after r14 | Exact young-leg reuse | Independent heavy A/P transport duplicates parent's young unconfined A/P. | Reuse only those already-full parent young A/P, retaining independent full old legs when parent begins confinement. H/Z unchanged. | r14rawnegative, no leg-reuse expansion selected. No candidate prepared. |
| next K3 mechanism | K111×edge non-hypertree birth | Official factored closure drops full K111 multiplied by covariance edge; scalar D3 rescale cannot add its nearly orthogonal missing component. | Target-free full-state orbit birth, not young-column subset, preserving distinct-index exclusions before history transport. | Low-rank covariance energy/noise gates currently inadequate. Do not claim promising; exact representation/cost below. |

## r14 production equations and exactness boundary

Pinned [official factor_k3 lines367–393](https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py#L367-L393)
uses light leg C, cost2 legs D21/2,D21.T/2,G22/4,C²/2, and pairs only total cost<=3.
For symmetric, zero-diagonal C/G, D21[i,j]=K3[i,i,j], and Wick derivatives w1,w2,w3:

```
A = diag(w1) C
H = .5 diag(w1) D21.T diag(w3)
  + .5 diag(w2) D21 diag(w2)
  + .25 diag(w2) G22 diag(w3)
  + .5 diag(w2) (C*C) diag(w3)
T_heavy = 6 Sym(A,I,H)
M = -6 (A*H).T
T_distinct = T_heavy + Sym(M,I,I)
```

Sym averages six permutations. Each heavy-left path equals its paired heavy-right
path under factor permutation; column Wick scalings move onto H. Hence eight
paths collapse into one H leg rather than eight CP histories. The M term is
essential: it makes **every repeated birth entry zero**, including D21, while
leaving all-distinct entries unchanged. It must not also be subtracted through
the parent's approximate rank16 residual.

The new history has four full matrices A,P,H,Z=PM. At its first linear step use W;
at later steps use WD=Wdiag(previous w1), including index relabeling but without
new rank/age/row truncation. Its readout has exact full D3 and D21 under this
expected-derivative transport:

```
D3 = 6 rowsum(A*P*H) + rowsum(Z*P*P)
D21 = 2(P*H) A.T
    + [2(A*H)+(2/3)(Z*P)] P.T
    + 2(A*P) H.T
    + (P*P)/3 Z.T
```

Current mean/covariance/K3 enter parent nonlinear evaluation before birth residual
centering, so the restored history feeds future states. This differs from a
final scalar correction or a one-step layer14 gate. Parent's other M/feed,
covariance and K4 proxy approximations remain. This is **complete heavy-tree
restoration under the represented input slices**, not exact true K3 or full
augmented K3. Covariance triangle and K111×edge remain absent.

Production inserts independent transport after parent's source relabeling,
adds its D3/D21 immediately before Wick construction, and appends exact distinct
birth after parent's residual assembly. The extra readout deliberately includes
all rows. Parent's own dead-row/saturation pruning remains an inherited
approximation. Local sym(C/G) canonicalizes existing float32 antisymmetric
roundoff without mutating parent covariance objects.

## Cost and verified contracts

Dense arithmetic upper bound at n1024/depth16: four transported matrix legs plus
four D21 matrix contractions per source/transition, each ~2n³ FLOPs. Across
1+…+15=120 transport and1+…+14=105 nonfinal readout source ages, additional
upper bound ~8n³(120+105)=1.933e12, excluding vector/copy/birth work and a
nontrivial frozen-parent cost. Plain dense arithmetic can exceed2**41 overall.
The candidate uses inherited legal Strassen array operations at the parent's
level4/5 caps and<=12matrix families; only actual strict billing establishes
feasibility. This formula supersedes earlier rough105-for-both estimates.
Four full matrix legs in two ping-pong slots cost480MiB at15sources/float32,
plus240MiB readoutleft scratch and inherited buffers. Memory/resource gates
are not relaxed. No compiled kernels, concurrency or meter intervention.

[Independent contract](../../results/phase2/tenfold_exact_hypertree_history_contract_r14.json):
synthetic n4/7/12/32 exact official path loops vs compact form max7.11e-15;
strict repeated birth max7.11e-15; three full transport ages D3/D21 max2.49e-14.
Actual frozenV29 network0 pre-ReLU layer14 n8/32 cache, original commit/reference
identity preserved, birth tensor max1.63e-19; no reference/target arrays accessed.

[Integration contract](../../results/phase2/tenfold_exact_hypertree_history_integration_r14.json):
n32/n64 depth4 wholepredict finite; FP32 n64 level1 legal Strassen D3/D21
relative3.47e-7/3.90e-7. Initial n8/depth16 smoke became nonfinite and is retained
as a failed script snapshot; small-shape rider behavior does not imply suite
acceptance. First helper attempt used unsupported+=; replaced with legal out=.
First real cache contract exposed tiny C/G asymmetry; canonicalization made the
mathematical compaction valid. A Strassen test harness initially passed4CP legs
in the wrong batch dimension, corrected via legal reshape, candidate unchanged.

Frozen [r14 production](../../estimators/tenfold_exact_hypertree_history_r14/SOURCE.md)
SHA832bb894ca238fa2014d963ab1f81c3ca08bb6d27de7434bef9eec70d9e4d7ad.
Root strict1 completed:1,632,808,519,965FLOPs, wall99.4213s/residual.28660s, zero failures. Raw worsens0.02855%, adjusted worsens239.71%; full restored heavy history does not explain the tenfold gap. No new best.

## Why this mechanism follows the new diagnostic

The independent r02 diagnostic has90successful groups and0failure. Fixed
late local K2/K3/K4 corrections project final local MSE4.419e-9→1.514e-10,
SE1.552e-10; K2/K3/K4 alone project2.314/3.275/3.220e-9. These estimates use
oracle moments and have substantial uncertainty, so they are not production
scores. They identify real covariance/third/fourth state gaps instead of
another global mean coefficient. Late standardized ΔD3 has near-zero cosine
with parentD3, further weakening scalar rescaling as a mechanism. r14 restores
a concrete omitted K3 path family with fixed theoretical coefficients.

## Subsequent non-hypertree route: explicit gate before candidate

Official factored code explicitly drops covariance triangles and K111×edge,
which contribute at the same leading squared n^-3 order. For the target orbit
`((0,1,1),(1,1,1))`, each labelled covariance edge multiplies full distinctK111,
with Wick w2 on the two edge endpoints andw1 on the remaining endpoint. Verify
exact labelled coefficient using the pinned generator, then independent n4/8
scalar enumeration before production; old corrected H29 has no complete history.

If C=Σ_s λ_s v_s v_s.T and K111=Σ_r Sym(a_r,b_r,c_r), each edge mode transforms
one source into three symmetric CP types, requiring3R×S columns. All-source
R~3×14×1024 makes naive rank multiplication unaffordable. Old actual rank1
covariance energy~0.75% invalidates a casual dominant-mode approximation.
Unbiased colored tensor/Hutchinson sketches need an actual readout variance gate;
existing triangle gate optimistic250probes SE/signal .72/.94/.34/1.46 already
rejects that unchanged method. Do not call rank64 promising without measured
whole-readout signal/noise relative to required raw5.779e-9.

The remaining viable design question is whether a fixed, target-free
output-weighted contraction can transport a compressed **functional** of the
new birth, with an error bound for subsequent D3/D21. Keep distinct-index removal
exact and retain all source ages. A single-step scalar contraction cannot recover
later covariance/K3 feedback, and optimizing against reference truth is excluded.
r14 is now negative, so this missing non-hypertree functional is the next mechanism; a target-free cost/signal gate must precede any production claim.

## Completed non-hypertree component gates, r18/r19/r21

The pinned generator confirms the mixed K111×edge orbit has3 labellededge terms,
each coefficient1; symmetrizing lets its nextD3 scalar be `3*source_projection`
with H=diag(W_i*w2)Cdiag(W_i*w2),u=W_i*w1. Independent n4/8 full labelledtensor
contracts pass. [r18](../../results/phase2/tenfold_mixed_fullhistory_readout_gate_r18.json)
uses actual all14 represented sources, each1024columns, including B1feedback,
M and feed. Fixedoutputs0–3 nextD3signals are−8.192109e-5,+2.515619e-5,
+2.973953e-5,+3.586395e-5; each float64 legal cost131,330,727,992FLOPs.
No truth or reference is accessed and no MSE is implied. Exact dense functional
for1024outputs would be prohibitive; component magnitude alone does not prove
it points towards true missingD3.

[r19](../../results/phase2/tenfold_mixed_fullhistory_variance_gate_r19.json) forms
the actual fixedoutput0 matrixQ just to diagnose exact Rademacher variance,
Var(3z.T C Q z)=18||offdiag Sym(CQ)||F². Exhaustive tinyprobes verify this formula.
Signal matches r18 exactly, variance3.021626e-7, SE/signal perprobe6.7100;
4503probes are needed for prespecified relativeSE≤.1. FormingQ legally costs
365,886,029,796FLOPs for onlyone output. The existingQ probe's8,388,608FLOPs
is explicitly **not** an implementable cheapproductionfee; the initial
optimisticknownQpass is insufficient and is superseded by actualmatrixfreecost.

[r21](../../results/phase2/tenfold_mixed_matrixfree_probe_cost_r21.json) avoidsQ
with exact CPmatrixaction: each of three factorpair products becomes
X(Y.Tz)+Y(X.Tz), followed by exact diagonal removal. n4/8/32 matrixaction parity
passes; actual output0 samefixedz probe matches r19 within5.83e-19. Full14source,
all1024column float64 matrixfreefee is4,227,952,640FLOPs peroutput/probe, after
sharedprep3,295,981,568FLOPs. Evenoptimistically giving remainingbudget onlyto
output0 allows418probes and relativeSE.32820>.1, so this productionnoisegate
fails. With this exactsource-by-source method1024outputs would cost4.329e12
FLOPs perprobe, beforehistory/D21; this is a measured methodcost, not a lower
bound for every possible reorganization. Do not prepare a noisy candidate.
All four outputs were fixed beforemeasurement, no outputselection/coefffit.

The negative gate removes this Hutchinson source-functional implementation,
while retaining the exact mixedorbit equations and actual readout data. A new
structured representation would need a substantially cheaper deterministic
history operator or a provedvariance reduction, not another arbitraryrank.

## Full represented M age/function diagnostics r22/r23

[r22](../../results/phase2/tenfold_m_source_age_readout_r22.json) separates every
one of14 actual frozenV29 sources into Mdiag=P*s, Mssep=3A*e,
Mrank16=Z[:,:16]ell[:,:16].T, and the separate rank1 feedM. It preserves all1024
birthcolumns and allsourceages; fixed outputs0–3 use nextlayerweights/currentgain.
Two functionals are measured: ordinary full `Sym(M,P,P)[g,g,g]`, including
repeatedcurrentindices, and the omittedstrictdistinct K111×edge functional
through legal sym_source_projection. Independent tiny denseSym contracts verify
both. Allcache/sourcehashes checked; originalweights loaded with referencecolumn
excluded. No MCtarget/direction comparison or MSE is recorded.

| component | ordinary nextD3 outputs0/1/2/3 | omittedmixed nextD3 outputs0/1/2/3 |
|---|---|---|
| diagonal | −7.08125e-4 /1.53627e-3 /5.96202e-4 /−2.22331e-3 | −2.1594e-6 /−2.51895e-6 /−2.63594e-6 /1.47009e-6 |
| ssep |1.41674e-3 /8.12856e-3 /1.82912e-3 /−9.39094e-3 |−3.07728e-5 /−1.77572e-6 /1.77089e-5 /1.23616e-5 |
| retained rank16residual |−1.20367e-7 /7.49210e-4 /−1.09189e-5 /−1.20595e-3 |−3.80679e-7 /1.93559e-6 /−2.43116e-7 /−1.61754e-6 |
| feedM_rank1 |1.71775e-4 /8.76621e-4 /1.01312e-3 /−1.00821e-3 |1.20442e-6 /−5.22808e-7 /2.69285e-7 /2.26150e-6 |

ssep dominates the ordinary M contribution here. Retained rank16residual's
small total inoutput0/2 masks sourceagecancellation: output1 age9–14 contributes
6.64225e-4 whileage1–4 contributes−1.19155e-4; output3old contributes−8.53474e-4
whileyoung contributes2.09330e-4. Do not infer oldsourceirrelevance from the total,
or choose ages using these four outputs. Fixedage1–4/5–8/9–14 binvalues and
allindividualsourcevalues are retained in the report. This measures represented
retainedM; it does not include the discardedRres that root'sr20 restores.

r22 mixes scalarordinary and scalaromittedmixed diagnostics, with sharedH@A/H@P;
its legalfloat64 fee128,454,467,584FLOPs peroutput is dominated by the mixed
projection. [r23](../../results/phase2/tenfold_m_source_fast_ordinary_r23.json)
implements an actual fast **ordinary-only** matrixfreecomponent:

```
G = nextweights[fixedoutputs,:] * currentw1[None,:]
CP = G P; CA = G A; CZ = G Z
diag = rowsum((CP*s)*CP²)
ssep = 3 rowsum((CA*e)*CP²)
rank16 = rowsum(((CZ16) ell16.T)*CP²)
feedM = rowsum(((CZ1) ell1.T)*CP²)
```

All14sources/fourfixedoutputs cost478,705,248 legalFLOPs afterconversion,
538,910,752FLOPs includingdiagnosticsetup/conversion;wall.2447s. Every individual
source/component matchesr22's independentlymaterializedM fullordinaryfunction
within2e-12. NoH/M/n³ tensor is formed. Both scriptsruffcheckpass.

These ordinaryM terms are alreadypresent inV29 and adding them toparentmean
woulddoublecount. The relevant futurediagnostic is whether parent sourceage
projection or residualdiscard shifts these fullfunctional directions; only
matchedH18state/MC/reference identities could quantify truebiasalignment.
CachedV29state must not be paired silently withH18MCtrace. No arbitraryrank or
scale wasselected, no newproductioncandidate/score claim fromr22/r23.

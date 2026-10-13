# R54: sequential local closure with learned prior states

Status: launched2026-10-13 19:56CST, PID1699263, immutable full14 run.
The original pre-launch design below is updated to the actual noise-aware
safety contract. R51 remains immutable evidence; the old26/newgraph trajectory
join failed its fixed reuse check, so R53 is not run. R54 keeps the original26 features, fixed ridge0.01, floors and signed
diagonal closure. Its sole mechanism change is replacing oracle prior-state
resets with previous locally trained heads on their own trajectories.

## Fixed data and five branches

Use only the original24 training IDs0..368 step16, frozen H18 and corpus
revision/hash already pinned by R48. Fresh validation, Mini8, and final-output
labels do not enter training, head selection or the pilot gate. Local targets
use the original MC four post-ReLU moments at layers1..14, converted to
central mean/variance/kappa3/kappa4 as in R48. No MC truth is sent to workers.

| Branch | Fitting IDs | Diagnostic held IDs | Prefix coefficient provenance |
|---|---|---|---|
|fold0|24 IDs excluding0..80|0..80 step16|Only this branch's18 fit networks|
|fold1|24 IDs excluding96..176|96..176 step16|Only this branch's18 fit networks|
|fold2|24 IDs excluding192..272|192..272 step16|Only this branch's18 fit networks|
|fold3|24 IDs excluding288..368|288..368 step16|Only this branch's18 fit networks|
|full24|All24|None|All24 fit networks; never used in fold CV|

At each actual cloud layer there are96 training captures (4x18+24) and24 held captures
(4x6). Run the training wave with a maximum96 CPU1 workers, then the held
wave with a maximum24. Held targets are consumed only by evaluation after
their branch's current head is frozen. Each worker sees one generated
network, the target layer, frozen source and branch-local prefix coefficients.
It sees no corpus reference arrays, current target head, held labels or other
branches. The orchestrator privately reads training labels and fits heads.

This is outer-network-held evaluation of a sequential training algorithm.
The branch's earlier head fits use those same18 training networks; that
training-set dependence is recorded. A full24 head must never be substituted
into a fold prefix. No new feature, regularization, blending or lambda sweep.

## Sequential worker and fitting algorithm

1. Construct the original full width1024/depth16 estimator setup and tables.
   Keep all depth-dependent schedule semantics. Generate exact original
   torch2.5.1 weights from seed770000+ID and verify the baseline weight hash.
2. Run H18 layers0..current. Layer0 injection is completely skipped. At each
   earlier layer1..current-1, build the original pre22 features, append the
   four own post central states divided by sigma powers, and apply exactly
   that branch's frozen previous head. Use R48 mean>=0 and variance>=1e-10
   floors, inherited delta once, fixed mixed-central slices and original
   nonlinear feed; ordinary-M diagonals/C diagonal/G are rebuilt downstream.
3. At the current posttrigger, before any current correction, capture
   canonical26 features, own four central states and sigma powers. Stop by
   a dedicated capture-complete exception. Do not change estimator.depth
   or truncate/rebuild schedules; no final prediction or score is claimed.
   Always remove the tracer in a finally block. Prefix trajectory capture
   is offline instrumentation, not a graded estimator.
4. The orchestrator computes `(original_MC_post4-own_current_post4)/sigma^p`
   on fitting networks only. Fit one linear26x4 ridge0.01 head at that layer
   using normalization estimated from that branch's fitting rows only.
   Freeze head and prefix JSONs with hashes, training IDs, source/data/tool
   versions and preceding-head hashes before held evaluation or next layer.
5. Evaluate this head on the held6 capture for that branch, retaining
   per-network/per-channel errors, all failures, floors and finite checks.
   The next layer uses only the completed/frozen branch-local prefix.

The worker may compute features with diagnostic NumPy, but submitted code
must later reproduce the same features with permitted flopscope operators.
The target-layer diagnostic call does not certify adjusted FLOPs, strict
timing, final error or resource acceptance.

## Pilot and evidence contract

First run fixed layers1,2,3. Layer1 has an empty learned prefix and reuses
the immutable R38 baseline24 current features/state/scales after source/weight/
payload/shape verification, building26 from the same trace. Each branch fits
only its own18/24 rows; no cloud baseline duplication. Layers2/3 use actual
branch-local prefix capture. Layers1..3 check integrity/finite/floors, then
the frozen full14 schedule continues; this policy was fixed before launch
using TRAIN-only marginal-noise evidence. A source/hash/finite/permutation/no-truth boundary smoke precedes
the training wave. Root/K4 prepare an actual full-body stop-capture contract
for prefix injection and canonical semantics; snapshot depth must stay16.

The pilot reports pooled held-network local residual error by channel and
layer versus zero current correction on the same learned-prefix trajectory.
Also show held mean/variance physical error after the current head and the
same network/layer H18 diagnostic comparison. Previous-layer gains, MC noise,
own-trajectory exposure and head fitting must be distinguished. These local
metrics are not a final raw or competition adjusted score. The earlier
proposed pooled1..3 accuracy gate was removed before launch because those
mean/variance residuals are dominated by the fixed MC-reference noise. No
local channel selects a layer, regularization or blend. Report all channels
and layers; final full16layer lawful fresh8 prediction decides whether there
is reliable raw signal. K3/K4 gains alone cannot accept a candidate.

For each unique branch/layer, use a new output path and immutable manifest
with command/PID/start UTC, pinned worker/helper/template/source hashes,
data/label/weight hashes, branch train/held IDs, prefix model hashes, target
layer, resource/thread limits and package versions. Persist compact current
features26/state4/scales4 (34 scalar columns per neuron), no giant weights,
full histories or shard download. Fit summaries and models remain private.
After a head is frozen, audit/checksum and delete no raw evidence. If full
14-layer storage exceeds free disk, preserve completed pilots and address
storage explicitly before extending. Do not substitute a mixed trajectory
or drop networks after seeing errors.

Any failure stops that branch/layer expansion without retry or relaxed gate.
Training waves already in flight may finish; preserve their failed evidence.
Only a subsequent independently verified final closed-loop gain warrants
lawful production parity, strict/RPC/resource checks and stage assessment.

## Target-free actual-body capture contract

`on_policy_capture_helper_r54.py` exposes `OnPolicyCapture(width,target_layer,prefix_models)`, `record_pre(state)` and `record_post(state)` plus `CaptureComplete`. It rejects any current/futurehead at construction: prefix keys must be exactly1..target−1. Layer0injection entirely skipped. At the current posttrigger it captures26/ownstate/scales before injection, canonicalizes once with inv_perm and stops by the dedicated exception. Prefix feature/normalization/update arithmetic is permitted fnp; NumPy copies are only diagnostic serialization/assertions. Worker helper sees no labels or fitting API.

`on_policy_capture_contract_r54_r01.json` checks width16/32prune/no-prune at target1/3(8cases), originaldepth16setup/CORRtablelength16, fixed16weightlayers, only literalridersTrue plus separately recorded smallschedule shapes in diagnostic snapshots. Actual frozenR48heads1/2serve arithmeticfixtures, not R54on-policytraining evidence. Ownpost26capture matches current pre-injection poststate/scales exactly; prefixlayers applied1..target−1only, current/futurehead rejection, originalpk1/mixedpairs nonmutation and canonical nonidentityperm checks pass. Dedicated stop prevents followinglayers; finally removes the tracer. This is small actualbody instrumentation, not1024prefixtraining or strictresource/candidateaccuracy. No cloud launched by the contract.


## Launch checkpoint and immutable execution notes

R54 source froze SHA
`1321d9713e9f00538092e03c2019beb4bfc161bf97dab09af6632ca9afe408dd` after
startup TRAIN24 noise evidence showed mean residual/noise near1.03 at layer1
and23.55 at layer14, variance1.03 at layer1 and97.33 at layer14. The initial
1..3 accuracy gate was therefore removed before launch: pilot1..3 checks
integrity/finite/floors only; the fixed method runs all14 layers once. This
uses the original26/.01 mechanism without a sweep. The r02 helper contract
passed8 full-depth16 small-body cases plus signed-delta/floor fixtures.

The first launcher PID1698800 exited before creating an experiment folder
and left a zero-byte original log; its exit status was unavailable. The log
is preserved. After confirming no active PID/folder, the same frozen source
was relaunched with a new launcher_r02 log. Unique active PID1699263,
Linux startticks3047507, manifest startUTC`2026-10-13T11:56:13Z`. Layer1 reuse
and all five frozen heads completed. Two actual layer2 fold0 training smoke
captures run before the remaining training/held waves. Source/model/raw
paths are excluded in the actual Git-directory info/exclude.

All raw cloud captures are committed to Modal Volume
`whest-onpolicy-r54-raw`, path`/raw/r54_r01/layerN/branch/role/idxNNNN.npz`,
with SHA preserved in the local journal. Local raw copies are retained while
free disk exceeds150MiB; volume copies remain even if future local storage
is skipped. Layer2 measured payload size determines a storage forecast.

Execution annotation: source eval_current comments call layer1 delta a carry
from analytic layer0, which is inaccurate; the general delta is the current
layer's original13-feature rider. Original H18 line358 defaultsNO_CORR=True,
and workers remove V-number environment variables, so delta isNone in this
run's default path. Thus the layer1 total-mean fp32 proxy equals K1 fp32 copy
on this default path. General nonzero-delta fixtures are covered separately;
canonical features record float64 K1+delta, whereas the next H18 mean executes
float32 addition. Local head metrics are diagnostic, not proof of final
emitted-mean or candidate bit parity.


Pilot1..3 completed without failures. Layer2 actual120 captures had median
6.105s; layer3 had median8.739s. Every branch's head was frozen before its
held evaluation and the next layer. Physical held local head/H18 ratios
(mean,variance,kappa3,kappa4) were layer1(.985815,.993190,.769263,.638953),
layer2(.973149,.996822,.831884,.733326), layer3(.970902,.993835,.905635,.793959).
The improvement in early mean remains small; K3/K4 improvements remain local
metrics. Current-head floors on24 held networks were layer1 mean0/var0,
layer2 mean0/var2, layer3 mean2/var7. Prefix floor counts and before-floor
minima are preserved separately in capture journals. Layers4..14 continue
unchanged as authorized; no early gain is promoted as the tenfold result.

Layer2 raw local payloads measured23,532,126 bytes total across120 captures (about196KiB each); a13-layer forecast is
305,917,638 bytes. Free disk at that checkpoint was531,337,216 bytes, with a
157,286,400-byte reserve. All evidence also has committed Volume copies.
Based on observed early-layer throughput, the initial full-run estimate is
8–20 minutes, subject to later-layer cost and cloud queueing. Maximum96
configured workers does not establish observed96-way concurrency.

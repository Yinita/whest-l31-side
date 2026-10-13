# Public corpus joint closure pilot r38

2026-10-13. No tenfold improvement has been verified. H18 remains the frozen
anchor. This document records a distinct offline learning experiment rather than
another tuned analytic transport parameter.

## Sources inspected

Dataset: [keenanpepper/arc-whestbench-p2-d8b-corpus-14k](https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k),
frozen commit `cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8`.
README says 14,048 networks, 16×1024 float32 He weights. Regeneration uses
PyTorch CPU `Generator.manual_seed(770000+idx)`, then one
`torch.randn(16,1024,1024,dtype=float32)` and multiplication by `sqrt(2/1024)`.
It does not pin PyTorch version. The pilot pins `torch2.5.1+cpu` and records each
regenerated weight digest; a first-layer analytic MC-noise gate tests compatibility
before fitting. Passing that gate is not a bit-exact RNG/backend proof.

`audit_public_closure_corpus_r36.py` read actual ZIP directories and NPY headers
with HTTP Range. Total shard payload inspected was 307,737 bytes. No full shard
was downloaded locally. Evidence is `public_closure_corpus_r36_audit.json`.

| Actual shard/member | Shape | Dtype | Meaning and limit |
|---|---|---|---|
| shard00 `moments` | 128×4×16×1024 | float32 | MC post-ReLU raw powers m1..m4 |
| shard00 `final_mean` | 128×1024 | float64 | MC final mean, N=100,000,000 |
| shard00 `noise_rep_final` | 1024 | float64 | Repeated-seed MC final mean, not 128 repeats |
| teacher00 `pre_mean`,`pre_var`,`k4d` | 128×16×1024 | float32 | Teacher state; extraction boundary unverified |
| teacher00 `k22_u`,`k22_v` | 128×16×1024×8 | float32 | Teacher rank-8 approximation, not MC pair truth |
| teacher00 `k22_fro`,`k22_res` | 128×16 | float32 | Teacher diagnostic norms |
| teacher00 `final_mean` | 128×1024 | float64 | Teacher output, not MC truth |
| ext00 `moments`,`final_mean` | 1500×4×16×1024;1500×1024 | float32;float64 | Additional MC networks |

Original and teacher shard00 actual `idx` values are **0,16,32,…,2032**,
not a contiguous 0..127 sequence. `n_done=128`, teacher `rank=8`,
MC `w_seed0=770000`, `s_seed0=880000`, `n_samples=100000000`.
The README's reported teacher Mini8 raw1.8023e-8 is an author's result, not our
reproduction. This corpus supplies MC marginal labels, not true joint K22/K31.

No LICENSE file, dataset-card license field, or README license declaration was
present. Public readability does not establish a redistribution license.
The pilot uses selected data privately for research; neither labels nor fitted
artifacts are to be published until that status is resolved. URLs advertised as
`keenanpepper/whest` and its `ap_p2_t3_teacherbake.py` source are currently
unavailable through GitHub API (404). We have not audited their code.

## Official training and submission rules

Fresh read-only official sources are frozen in
`public_closure_corpus_r36_official_sources.json`:

- [Phase2 launch, official post1](https://discourse.aicrowd.com/t/18197/1):
  “Data files remain permitted, including weights, lookup tables, and precomputed
  artifacts.” Runtime computation must all pass through flopscope, with
  2**41 FLOPs, 120 seconds, 0.4 seconds residual and 8GB memory.
- [Townhall, official post1](https://discourse.aicrowd.com/t/18078/1):
  “Offline training on external Monte Carlo data … is within the rules and
  untracked.” Private scoring uses entirely fresh networks.
- [PyTorch FAQ, official post3](https://discourse.aicrowd.com/t/18040/3):
  train/precompute offline, ship pickle-free `.npz`, load with `fnp.load` in
  setup. It does not allow PyTorch/NumPy arithmetic in a submission.

These permit learned parameters as data; they do not change corpus licensing or
make an offline NumPy intervention an admissible runtime implementation.

## Frozen experiment

Source anchor SHA256
`c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`.
MC shard00 SHA256
`aab6930a8a82b9dfe792a0e71456873165f73d143faac06fc32e5ca52b2b0fa5`.
Script `modal_joint_closure_pilot_r38.py`, launcher PID1652715. New output root
`results/phase2/modal_joint_closure_pilot_r38/` and separate launcher log.
The app is bounded to 32 concurrent 1CPU workers, 4096MiB each, 300s timeout,
no retries:32 baseline fullshape rolls and8 corrected validation rolls.
It neither bakes a new MC reference nor downloads the ~6GB collection locally.

Training: shard00 first24 raw indices0..368 step16. Validation: raw indices
384,400,416,432,448,464,480,496. Split is by complete network, frozen before
label fitting. No Mini8/public benchmark labels enter fit. This seed split is
separate from official public seeds, but corpus authors' historical use and exact
cross-dataset weight collision absence have not yet been established, so it is
not called independent competition validation.

H18 feature construction sees only regenerated weights/current rollout states.
MC labels are downloaded to a separate worker, returned only after baseline
capture, and used locally for fitting/scoring. Rollout workers never receive
labels or teacher states. They receive an index and, for validation intervention,
frozen coefficients and normalization constants.

The fixed model is one ridge (`lambda=.01`, normalized features, intercept
unpenalized), jointly predicting four scaled errors for effective post mean
`K1v+parent_delta`, variance, K3 diagonal and K4 diagonal. Labels convert MC raw
powers to ordinary central cumulants. Captures are actual H18 rolled-forward
states. H18 inverse neuron permutations are applied to every saved feature,
state and scale row before comparing canonical MC labels; live intervention
features retain internal order.

Features are original13 plus eight predetermined Hermite/skew/kurtosis channels
and depth fraction, total22. Layers0..14 supervise the rollout. Final mean-only
layer15 has no state intervention. No hyperparameter sweep, reference-selected
layer weights or teacher-forced input is used.

Before fit, first-layer analytic mean (`||w_i||/sqrt(2pi)`) is checked against MC
mean labels, with analytic mean-noise variance `var(ReLU)/N`. Aggregate noise
ratio outside [.4,2.5] terminates the pilot without fitting or closed rollout.
This aggregate32 gate was frozen before launch; there was no separate1–2-network
pre-launch smoke. Preserve that ordering limitation rather than claiming one.

## Intervention semantics and exact next ablation

R38 is expressly a **formal signed diagonal cumulant closure projection**,
not an assertion of a positive joint distribution. Model errors add to post
mean/variance/K3/K4. Mean and variance floors are fixed before labels; clipped
counts are recorded. Off-diagonal ordinary K11/K21/K22 remain the original
ReLU closure. Current nonlinear gain/feed birth remains original. The downstream
ordinary-M diagonal residual S3c is rebuilt once from corrected K3 minus original
old/new feed diagonal; covariance diagonal and harmonic K4 core consume the
corrected states in the same call. The original power cumulants and e_b birth
expressions remain original-ReLU expressions. This is an approximation to be
measured, not exact moment transport or arbitrary raw-moment recentering.

The launch manifest's phrase “All newborn diagonal sources … rebuilt” is too
broad. Its precise meaning is the ordinary-M diagonal residual and downstream
K4 projection; nonlinear feed birth is unchanged. Preserve original manifest
and this clarification without mutating the running script.

An independent finite-distribution audit by the K4 agent verifies deterministic
mean translation leaves all central joint cumulants unchanged and requires
corresponding raw-power transformations if those powers are reused. Its
`tenfold_joint_state_distribution_contract_r38_r02.json` demonstrates errors
below1e-15 and the wrong raw-fixed recentering failure. An exact next ablation
uses the same fitted model's mean channel only as `Y'=Y+b`, holding all central
states/source birth expressions fixed; fixed nonnegative mean clipping simply
changes deterministic b. It must be measured in a new r39 output path, with
the same paired weights and labels. A formal four-state gain cannot be treated
as an admissible candidate before valid closure semantics and flopscope port.

For all results report per-network before/after raw, modeled single-step4-state
errors separately from closed final error, clipping/failures, hashes and timing.
The corpus README's ~9e-10 MC noise floor is not a proven uncertainty for our
pilot. Single-repeat labels at N1e8 and float32 high-order raw moments limit
very small residuals. Neither corpus error reduction nor oracle state injection
satisfies the Mini8 `<5.779e-10` / independent paired tenfold stage criteria.


## r38 resource failure and r02 checkpoint

Original r38 returned32/32 infrastructure RemoteErrors, zero captures and no
fit. Launcher emitted SIGKILL137/OOM for 4GiB workers; original completion and
journal are preserved. No algorithm score is assigned to these failures.

`modal_joint_closure_pilot_r38_r02.py` is a new frozen script, SHA256
`c63802b1c119e11f19dfd0b8d3015caf823aba0846ff37894a31618bc3473fdf`,
launcher PID1655576, new root
`results/phase2/modal_joint_closure_pilot_r38_r02/`. This offline diagnostic
uses16GiB RSS memory; it does not pass the8GB competition process contract.
Two baseline smoke networks0/16 precede the remaining30. Each first-layer
analytic/MC mean-noise ratio must be within[.2,4.0]; the32-network aggregate
[.4,2.5] gate remains before fit. Workers record maxRSS/RLIMIT_AS and assert each
pre/post layer capture fires exactly once. Same frozen coefficients then run
mean-only translation first and formal four-state projection second on the8
validation networks, writing separate reports/arrays. Labels are downloaded on
an isolated worker but used only in local gate/fit/score, never rollout workers.

Local cached official corpus scan found65 unique network IDs/seeds across
Mini8 and extension shards. None of their64-bit seed values lies in the
770000..784047 external-corpus seed range. This is seed-disjointness evidence,
not a verified whole-weight collision check or evidence about upstream author
selection. Actual teacher/source boundaries remain unverified.

Current r02 status at19:07CST: selected MC label payload7.7MiB received;
first two actual H18 smoke rollouts running, zero results yet. No trained model
or accuracy improvement has been produced.


r02 two-smoke results at19:09CST: both completed, elapsed80.6421/83.2411s,
FLOPs480,763,746,423 each, peakRSS5,400,524/5,386,752KiB,
RLIMIT_AS unlimited. First-layer analytic/MC ratios .970375/.986810 pass.
Saved effective state mean equals returned canonical rows0..14 exactly(maxerr0).
Original final MCraw2.4072134e-8/2.4785995e-8. Remaining30 have begun.

Important result: original post K2 minimum is−2.96196e-6/−2.27461e-6.
Therefore r02's branch named`mean_only` applies the fixed variance floor as
well as mean shift at negative entries. It is a **mean plus variance-floor
ablation**, not exact mean translation. Its original filename/manifest must
remain preserved, but report this mechanism precisely. R39 production/template
and a separate validation run must update K1 only, preserve K2/3/4 including
negative entries, and attribute that difference explicitly. No evidence is
silently relabeled or overwritten.


r02 baseline32 completed zero failures and aggregate analytic/MC ratio.9932356.
All32 MC final raw average2.22602952e-8; train24 2.21211547e-8;
validation8 2.26777168e-8. Frozen joint ridge SHA256
`48909c1ab79be77b30699dd1b66385a4a50d048718cc31f70a7fd1f604a795d0`.
Single-step held-network4-state MSE ratios(before→after aggregate) are:
mean .99338268, variance .98575651, K3 1.17065254, K4 1.25677824.
This predicts only0.66%/1.42% mean/variance gains while worsening higher-order
states17.1%/25.7%; it is no evidence of tenfold. Closed rollouts remain required.

Exact mean-only ablation now has a separate frozen
`modal_corpus_exact_mean_r39.py` and launcher PID1659360, output
`results/phase2/modal_corpus_exact_mean_r39/`. It reuses the same frozen model,
train/validation split, baseline and local-only MC labels, running only8 new
rollouts. Intervention changes K1 only (including its fixed mean floor),
never floors/copies original K2/K3/K4. Clipped mean just selects a deterministic
translation b per neuron; that preserves central cumulants but does not
establish nonnegative support of the translated originalReLU distribution.
Negative approximate parent variances are recorded as diagnostics, preserved.
R39 and r02 run separate immutable scripts and unique result paths.


## Completed fixed shared-linear experiment

All8 paired networks complete with zero failures. Before MCraw2.26777168e-8.
Mean plus variancefloor after2.39329533e-8(+5.5351%); exact mean-only after
2.39290862e-8(+5.5181%); formal four-state after2.70830947e-8(+19.4261%).
These fixed variants are rejected. Do not spend a submission or Mini8 strict
screen on a negative corpus result. Selected small final evidence is
`public_closure_corpus_r38_completed_summary.json`; large NPZ/rawlabels and
fitted coefficient/model artifacts remain private/local and excluded from Git.

A distinct fixed r42 changes shared coefficients to per-layer coefficients,
with known analytic layer0 untouched. Train24 only,4fold networkCV(18fit/6held),
same22features/sameλ.01/no sweep, gives per-layer/shared error ratios
μ.9987566,variance.9745250,K3.7225127,K4.5148254; wins22/24,24/24,24/24,24/24.
This is a training-CV signal, not final-layer score. Frozen model SHA
`06187606cb8ceb65e42f76e18f1c510a3a3045af8cd61dff8c424864a00a211f`.
New fresh validation8 rawindices512..624 step16 were frozen before rollout.
`modal_corpus_per_layer_r42.py`, PID1663266, runs only8 corrected networks in
`results/phase2/modal_corpus_per_layer_r42/`; it receives no labels at all.
Paired H18 baseline/MC labels are shared with the independent r41 quadratic
experiment via `results/phase2/modal_joint_closure_quadratic_rollout_r41/`,
avoiding duplicate baseline compute. Separate local scoring will verify actual
weight hashes/label indices before scoring. No first-layer correction or floor
is performed in r42. First-layer state mean remains exact analytic.


R42 CV attribution removes an analytic layer0 control difference without
changing/refitting the frozen deployment model. Same4 train-network folds,
shared and per-layer corrected layer0 both fixedzero, yield aggregate ratios
μ.99878535,var.97456187,K3.74020653,K4.58446251. Restricting score to layers
1..14 yields.99872568/.97249665/.71471739/.50907323. Thus K3/K4 training-CV
signal is still present after isolating exact-layer0 correction. This
train-only attribution uses no held8 labels and performs no layer selection;
`analytic_layer0_attribution.json` preserves each network/layer metric.


R42 fresh8 complete0fail: originalraw2.22377851e-8→2.45966812e-8,
+10.6076%, all8 worse. `paired_score.json` checks weight and output digests,
shared MC IDs and matched canonical ordering. Unchanged first-layer outputs
have cross-container float32 roundoff on some networks rather than being all
bitwise identical; their maxabs is recorded, not silently made bitwise.
Training-CV improvement in K3/K4 did not transfer to a closed rollout, so this
fixed per-layer intervention is rejected.

Next r43 constructs **linearized local mean innovations** rather than total
post-state error. On train24 only, e_l=MCpostmean−actualH18postmean and
z_l=e_l−Phi_l*(W_l.T@e_(l−1)). Regenerated full weights must hash-match the
same baseline before transport. This is a first-order frozen-mean Jacobian
and omits covariance/higher-source responses; it is not exact nonlinear
innovation. Oracle MC error vectors are offline training-target construction
inputs, never candidate inputs or accuracy results. App needs only24 small
matrix-vector transports, not24 new H18 rolls. R43 PID1667283, output
`results/phase2/modal_mean_innovation_targets_r43/`; alternative agent owns
separate train-only r44 CV. Layer0 target noise is preserved in evidence,
but any learned layer0 correction is fixedzero. Final16th prefatures are
unavailable in original15state cache and require a distinct future capture;
this prototype covers layers0..14 only.


R43 train24 transports allcomplete0fail. At layer14, observed totalMCmean error
MSE1.97842855e-8 versus linearized local innovation4.07296822e-9; inherited
error explains much of the discrepancy, not a candidate gain. R44 fixed local
innovation model's train-CV effect was onlyabout1.4%, below its.90 gate.
R45 replays its held-network predicted innovations through exact regenerated
W and frozen Phi, all24complete0fail. Layer14 local-replay/before ratio
.99420693, versus total-error-injection/before1.12483026. This first-order
train-CV result rejects further full mean-only rollout; it is not full16layer
finalMSE or a lawful resource run. Next scientific data requirements are in
`JOINT_INNOVATION_DATA_REQUIREMENTS_R46.md`: the existing MC marginal corpus
cannot identify transported off-diagonal covariance or joint cumulant error.


R46 separate oracle pilot is now active(two train-only corpus networks0/16).
At post ordinary cumulant boundary, true MC postmean/variance/K3/K4 are reset
on layers1..14 using canonical truth[perm]; layer0 remains actual analytic,
and final layer15 has no truth reset. Existing mixed central pair slices stay
original and downstream ordinary diagonal sources are rebuilt. Oracle labels
are explicitly sent to the oracle research worker; no candidate/model inference
result is asserted. App PID1671520, `modal_joint_oracle_reset_r46.py`, output
`results/phase2/modal_joint_oracle_reset_r46/`. It records covariance minimum
and maximum eigenvalues/negative eigenvalue counts at every reset boundary.
Desired afterstate is saved float64 MC truth, while the live state is copied
float32; afterstate is not a claim of exact backend identity. Independent K4
orientation audit passes non-involutive permutations, both weight axes, raw
central conversion and inheriteddelta once. If even this oracle reset fails
to give strong final gain, larger training on this post-state interface is not
justified; if it succeeds it only identifies a possible interface upper bound,
not a deployable algorithm or measured tenfold stage.


R46 two-oracle gate completed0fail: network0 raw2.40721341e-8→4.62310067e-9
(ratio.19205197), network16 2.47859949e-8→4.35852950e-9(.17584646).
This5.2–5.7× oracle gain justifies studying local-closure targets but does not
reach10× even with oracle inputs. No covariance eigenvalues below−1e-9 on
these two runs; minima~−1e-12 are float noise. Reset returned means1..14 equal
stored float32 MCmeans exactly, layer0 unchanged. Truepost variance can be
exactzero for dead neurons; oracle reset uses raw truth without artificial
floor. Subsequent H18 prevar retains its original numerical floor.

R47 remaining22 training-corpus networks32..368 step16 are capturing the same
oracle-conditioned local output. PID1673549, `modal_joint_oracle_reset_r47.py`,
new output`results/phase2/modal_joint_oracle_reset_r47/`. Dense eigenvalue
checks are omitted after the two-network gate; live actual aftercopy float32
all4 states are also captured canonical to distinguish desired float64truth.
These are oracle training targets explicitly permitted onlyoffline, no
candidate score. R48 independent train-network CV will fit truepost minus
own closure output given previous truepost states, not baseline total error.


R47 remaining22 completed allsuccessful; final24 combined oracle before
2.21211547e-8→4.45453077e-9, ratio.20136972(about4.97×). This is still
oracle-only diagnostic, not a tenfold/deployable result. Full small combined
identity and per-network score evidence is`joint_oracle_train24_r47_summary.json`.
Private desired/actual state arrays are retained, no retry or duplicate run.
R48 independent train-only local-closure CV is now complete; the next26-feature
learner adds current own pre-reset poststate4/σpowers to frozen22features.
Alternative agent owns its fresh8 natural closed rollout on indices640..752,
with original baseline and separate local MC reference; no oracle labels
are available to the learned prediction worker. Its results are pending.

R49fresh8 complete0fail: frozen teacherforced-local26 natural closed rollout
raw2.32441291884e-8→2.33023356538e-8(+0.2504%),4/8wins, paired network
bootstrap95ratio[.97347164,1.03421333]. No reliable gain, no Mini8/strict RPC
or stage promotion. Source integration audit passes only explicit metadata,
label-boundary and existing helper semantics, not wholepredict acceptance;
`tenfold_local_closure_integration_source_audit_r49_r01.json` records scope.
Next separate mechanisms are fixed6signed graph context on TRAIN oracle
trajectory(R50/R51) and one final-only26total-error readout on unchangedH18
TRAIN24(R52). The latter bypasses middle-source intervention and tests actual
final_mean labels; previous L03final13 negative is documented rather than
forgotten. Fixedgate/spec and originalR52numerical smoke failure are recorded
in `TENFOLD_FINAL_READOUT_R52.md`. Neither route has a demonstrated scoregain.


## r49 fresh8真实闭环否决，转向联合图context / final-only检验

2026-10-13 19:40 CST：固定train24 local26模型在首次完整新8网idx640,656,672,688,704,720,736,752上，raw MSE **2.324412918841486e-8→2.330233565377604e-8（+0.250414%）**；4/8逐网改善、零失败。modelSHA f50a3ac0990ed659dd39dd66ff18a998a733305a09a8bc4dbcda5fe32bc5f559，parent/input hashes逐网保留。bootstrap95 ratio [.9734716407,1.0342133276]，不能称可靠收益。MC N1e8离线raw、16GiB与NumPy trace只作诊断，非adjusted/RPC/资源验收；不扩Mini8、不上传、不扫λ/scale/layers。最佳仍V37bH18、stage1阈值5.779e-10不变。

R50六个固定target-free有向协方差/混合source图context正确性及实际小网络只读capture已过；1024宽6feature计费4.343G/层，14层约60.8G，只有组件成本。r51将同train24 oracle轨迹仅capture六列，固定26→32网络CV检验剩余信息。独立r52将仅最终输出的固定26feature affine readout在独立corpus train24网络CV检验；与旧L03(V29/publicMini8/13列)负结果区别仅为H18/新数据/固定26，不调参、不复用已读fresh标签。只有可靠raw闭环信号再跑严格RPC与冻结Mini8；不把oracle约5倍或CV当十倍成功。

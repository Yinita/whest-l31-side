# Public corpus closure: R38–R41 audit

The frozen H18 source remains
`c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`.
The public MC corpus is frozen at
`cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8`, shard00 SHA
`aab6930a8a82b9dfe792a0e71456873165f73d143faac06fc32e5ca52b2b0fa5`.
MC labels are post-ReLU raw moments, not the approximate teacher cumulants.

## State and order audit

Features, post states and sigma powers are captured in H18's internal neuron
order. Saved training rows must be indexed by each layer's `inv_perm` to match
canonical MC labels; live feature inference remains in internal order. R38
corrected this alignment before launch. A target-free actual H18 trace at
width32/depth4 checked pruning off/on and mean/four-state operators. The frozen
body was changed only by a literal `riders=True` in a diagnostic snapshot;
tiny schedules were set in memory. Maximum ledger error was 2.4413e-8.

K1 and pk1 do not alias. The formal four-state projection changes the four
central marginal carriers while freezing mixed central K21/K22 and original
ReLU raw expressions. Downstream C diagonal, variance carry, ordinary-M
diagonal residual, harmonic G and K4 sigma are rebuilt. Delta-K3 enters M once
and survives the next source transport. The original nonlinear feed birth
remains unchanged. This defines a signed closure approximation; it does not
establish a positive joint distribution. K4 still uses a harmonic proxy and
does not retain the full corrected fourth-order tensor.

Evidence: `results/phase2/tenfold_joint_closure_ledger_audit_r38.json` and
`experiments/phase2/tenfold_joint_closure_ledger_audit_r38.py`. Failed harness
snapshots and logs are retained; the successful audit uses the same depth4
scope as earlier source-ledger audits, not the full competition shape.

## Resource and MC noise evidence

The initial R38 4GiB workers all exited with SIGKILL137 before any baseline
completed. The R38 r02 diagnostic completed 32/32 H18 traces at 16GiB and
measured 5.137–5.230GiB peak RSS per process, with exactly15 pre/post captures
per network. Trace arrays use about3.52MiB. The failed launch is an environment
failure, not an algorithm score. NumPy inference, Torch runtime and tracing
are offline diagnostics, not production memory/time acceptance.

On the 24 training networks, exact first-layer Gaussian-ReLU cumulants give
measured MC MSE divided by delta-method N=1e8 variance:

| Channel | Mean noise ratio | Range |
|---|---:|---:|
| mean | 0.990287 | 0.857742–1.076949 |
| variance | 0.991502 | 0.891317–1.149443 |
| kappa3 | 1.000124 | 0.930504–1.131234 |
| kappa4 | 1.003677 | 0.937622–1.071470 |

These are descriptive, with shared MC draws across neurons. They support the
RNG/order/raw-moment interpretation and show that even higher-order targets
contain finite MC noise. First-layer kappa4 MC MSE is about1.1e-6; first-order
fp32 storage sensitivity RMS is about6.1e-7 (squared about3.7e-13), so storage
rounding does not explain that noise. Train24 contains no negative target
variance. No neuron-independent confidence interval is inferred.

## Completed linear closed-loop negatives

The model uses fixed22 features, lambda0.01, train indices0..368 step16 and
held indices384..496 step16. Both modes use the same frozen model SHA
`48909c1ab79be77b30699dd1b66385a4a50d048718cc31f70a7fd1f604a795d0`.

| Version / actual mechanism | Raw MSE before | Raw MSE after | After / before | Wins | Paired network bootstrap95 ratio |
|---|---:|---:|---:|---:|---:|
| R38 r02 mean correction plus variance floor | 2.26777168e-8 | 2.393295e-8 | 1.055351 | 1/8 | 0.983607–1.131504 |
| R39 exact mean correction, no variance floor | 2.26777168e-8 | 2.39290862e-8 | 1.055181 | 1/8 | 0.983729–1.129168 |
| R38 r02 formal four-state projection | 2.26777168e-8 | 2.70830947e-8 | 1.194260 | 0/8 | 1.112896–1.270246 |

All three completed with zero failures. Sources, weight hashes, tool versions
and payload hashes were checked. Bootstrap resamples networks and keeps each
network's paired common reference. It excludes MC-reference sampling
uncertainty and does not establish competition generalization. The misleading
`mean_only` artifact name in R38 r02 actually includes variance flooring; R39
isolates the true mean translation. No linear model promotion is justified.

Evidence: `joint_closure_pilot_r38_r02_statistical_audit_completed.json` and
`corpus_exact_mean_r39_network_audit.json` under `results/phase2/`.

## R41 fixed quadratic mechanism

R41 changes one main model mechanism: fixed degree2 features, with 253columns
(22 original standardized features and231 upper-triangle products of the21
nonconstant features). Raw and expanded columns are normalized using fold
training networks only. Lambda0.01 remains fixed. Four blocked network folds
train18 and hold6 within the original train24; the eight R38 validation labels
are never used to fit, tune or choose this model.

The mechanism gate was declared before execution: any channel with pooled
normalized MSE quadratic/linear<=0.90 and at least16/24 held-network wins.

| Channel | Normalized quadratic / linear | Raw quadratic / linear | Network wins |
|---|---:|---:|---:|
| mean | 0.988156 | 0.987424 | 24/24 |
| variance | 0.976686 | 0.959962 | 24/24 |
| kappa3 | 0.871288 | 0.739042 | 23/24 |
| kappa4 | 0.766829 | 0.505307 | 24/24 |

Kappa3 and kappa4 passed this development gate. Their raw quadratic/H18 ratios
are0.8755 and0.6809. There is no tenfold or downstream-score claim. The all24
frozen model SHA is
`0610032b01bb3538a5ca6bea09079c92601222e7155775d1ea4ff0b292231e3b`, stored
at `results/phase2/joint_closure_quadratic_cv_r41/frozen_train24_quadratic.json`.

The next closed rollout uses fresh indices512..624 step16, paired identical
H18 weights/reference, all four channels jointly, and first-layer correction
exactlyzero because Gaussian-ReLU is analytic there. The rollout source is
`experiments/phase2/modal_joint_closure_quadratic_rollout_r41.py`; outputs use
`results/phase2/modal_joint_closure_quadratic_rollout_r41/`. PID1663482 launched
at2026-10-13T11:15:57Z with CPU1/memory16GiB, at most8 workers, no retries.
Its closed rollout completed at2026-10-13T11:19:30Z with zero failures. Fresh8
raw MSE was2.2237785107468484e-8 before and2.4114898364038576e-8 after,
ratio1.084411 (+8.4411%). Only1/8 networks improved. First-layer correction
coefficients were zero;4/8 returned first-layer rows were bit-identical, and
the remaining first-layer differences require a separate numerical audit.
The training-state CV improvement
therefore did not translate into a better recursive final prediction. This
rejects the frozen R41 joint quadratic rollout. Evidence and network-bootstrap
uncertainty are in `results/phase2/joint_closure_quadratic_rollout_r41_network_audit.json`.
The fresh paired baseline remains reusable by separately frozen mechanisms so
that baseline cost and evidence are not duplicated.

The strict first-stage adjusted target remains below5.779e-10, with independent
tenfold confirmation, legal operators, RPC and resource checks still required.

## R43–R45 causal mean innovation

Fitting total post-layer error and injecting it at every layer can double-count
inherited error. R43 computed, on only the original24 training networks,
`innovation_l = total_error_l − Phi_l * W_l.T @ total_error_(l−1)` using the
canonical gate/order and regenerated weights whose hashes matched each H18
baseline. All24 transports completed; the exact decomposition held. The
smaller innovation label is a change of target, not an achieved score. Same
MC draws across layers induce label-noise covariance, so subtracting an iid
noise floor from this residual would be unjustified.

R44 kept the same22 features, per-layer linear lambda0.01 and sigma-scaled
residual. Four blocked train24 network folds each fit18/hold6, exclude analytic
layer0 from fitting, and leave final-layer local correction zero. The pooled
normalized innovation MSE ratio was0.988860806 (raw0.986199688);22/24 networks
improved. Paired network bootstrap95 was0.985569–0.992352. The predeclared
mechanism gate required ratio<0.90, so it failed. A frozen all24 model is
retained for audit, SHA
`4cd60179babfd045b3b4947343f6b96e5879a4b42e3fe06d63fc71f9f850bb8a`.
The style-cleaned source only renames a comprehension variable; the executed
snapshot/hash is retained in `source_style_provenance.json`.

R45 replayed held-network predictions through the actual regenerated weight
Jacobian, without fitting or changing a trajectory. At layer14, original
error MSE1.9784285518494912e-8 became1.966967378410733e-8 for local innovations
(ratio0.994206931), whereas repeatedly injecting predicted total error gave
2.2253962994861504e-8 (ratio1.124830259). This attributes an actual duplication
problem but offers only about0.58% development replay improvement. It is not
the final16th-layer score or nonlinear closed-loop candidate. The fixed linear
mean-only mechanism stops without a full candidate launch.

Evidence folders: `modal_mean_innovation_targets_r43`,
`causal_mean_innovation_cv_r44` and `modal_mean_innovation_replay_r45` under
`results/phase2/`. Earlier V29 Mini8 local-innovation LOO was a separate,
similarly small diagnostic (ratio0.984852), not an independent improvement to
H18 or the tenfold stage.

## R46 partial diagonal oracle probe

On two predeclared training networks0/16, the offline probe resets only the
post-ReLU marginal mean/K2/K3/K4 at layers1–14 to the MC labels. Layer0 stays
analytic, mixed central slices and original nonlinear source births remain
the inherited approximation, and the final layer is not reset. This is a
nondeployable diagnostic of this partial diagonal mechanism; it is not a
rigorous upper bound on the full recurrence or a trained candidate.

| Network | Original final raw MSE | Oracle final raw MSE | After / before |
|---|---:|---:|---:|
|0|2.4072134117e-8|4.6231006666e-9|0.192052|
|16|2.4785994874e-8|4.3585294953e-9|0.175846|
|Mean|2.4429064495e-8|4.4908150809e-9|0.183831|

Both completed without failure. Captured means match the reset fp32 MC means;
covariance eigenvalues never fall below−1e-9 across the captured15 layers.
Tiny negatives down to−2.36e-12 are recorded. Positive covariance does not
prove that all higher-order joint moments are realizable.

The repeated-reference vector is empirically nearest to network0 among the32
available labels (MSE1.7989558940e-9 versus the next network1.06494). The README
does not state its network ID, so this pairing is explicitly inferred. Using
that reference gives network0 original2.5308377645e-8 versus oracle
6.4212465088e-9, ratio0.253720. No noise floor is subtracted. Same MC draws
across layers/reference can inflate oracle agreement, and two training
networks cannot certify a tenfold gain. Evidence:
`results/phase2/joint_oracle_reset_r46_independent_audit.json`.

The useful next question is whether train-only local closure errors, captured
under corrected previous states, can be predicted without oracle access.
R47 captures the remaining22 training networks with the same reset mechanism;
R48 will compare fixed per-layer22-feature ridge against26 features that add
the model's own four pre-reset post states, using only network splits inside
train24. This remains a research checkpoint, not a stage transition.

R47 completed all22 remaining traces without failures. R48 then completed the
prespecified train24 network CV; no validation or final-output labels selected
features or regularization. The local target is MC post-state minus the model's
own pre-reset post-state, given corrected previous layers. The appended26-state
features are those own pre-reset states divided by sigma powers, not truth.

| Channel | Feature22 / baseline normalized MSE | Feature26 / baseline normalized MSE | Feature26 / baseline physical raw MSE | Feature26 wins |
|---|---:|---:|---:|---:|
|mean|0.974451|0.972440|0.974000|22/24|
|variance|0.959771|0.958857|0.959485|24/24|
|kappa3|0.751931|0.748595|0.838127|24/24|
|kappa4|0.577425|0.557808|0.616321|24/24|

Kappa3/Kappa4 passed the predeclared development gate (normalized ratio<0.80
and at least16/24 network wins). The frozen26 model SHA is
`f50a3ac0990ed659dd39dd66ff18a998a733305a09a8bc4dbcda5fe32bc5f559` at
`results/phase2/local_joint_closure_cv_r48/frozen_train24_local_closure_26.json`.
Feature26 adds only modest incremental improvement over22, and teacher-forced
CV still has exposure bias and correlated MC noise. Neither error table is a
competition adjusted score.

R49 launches the frozen learned26 model on fresh networks640..752 step16,
paired H18 baseline8 followed by candidate8, with CPU1/16GiB, at most8 workers,
no retries and timeout300s. PID1678259 launched in this checkpoint; source
`experiments/phase2/modal_local_joint_closure_rollout_r49.py`, outputs
`results/phase2/modal_local_joint_closure_rollout_r49/`. Runtime rollout sees
weights and frozen coefficients only, not MC labels. First-layer injection is
completely skipped; final-layer local correction remains zero. All four
channels change through the declared signed diagonal closure.

R49 finished at `2026-10-13T11:36:30Z`: all paired8 networks completed with
zero failures. Frozen source SHA was
`73eaf6c20a79bbe909423f99863a1be7f85393f2f2c845db931f100d3582e3b5`.

| Network | H18 final raw MSE | Learned26 final raw MSE | After / before |
|---|---:|---:|---:|
|640|2.4316773131e-8|2.3103409139e-8|0.950102|
|656|2.2137768927e-8|2.2916874768e-8|1.035194|
|672|2.1667704519e-8|2.1164174102e-8|0.976761|
|688|2.1456167913e-8|2.2011980580e-8|1.025905|
|704|2.3812213915e-8|2.4267318773e-8|1.019112|
|720|2.5376245675e-8|2.7393893727e-8|1.079509|
|736|2.1742393038e-8|2.1254884728e-8|0.977578|
|752|2.5443766388e-8|2.4306149413e-8|0.955289|
|Mean|2.3244129188e-8|2.3302335654e-8|1.002504|

The aggregate worsened0.2504%;4/8 networks improved. The paired-network
bootstrap95 interval for the after/before ratio was0.973471641–1.034213328
(20,000 resamples, seed261041), so it provides no resolved gain. MC sampling
uncertainty is retained in the raw paired scores and is not covered by that
network-bootstrap interval. Source/model/data/weight/version/payload hashes
passed the generic closed-pair audit. Candidate observed peak RSS was
5.159–5.214GiB and wall time64.97–105.73s in the16GiB diagnostic containers;
these are not a strict resource pass. First-layer injections are skipped;
5/8 first-layer outputs were bit-identical, the other3 show backend numerical
path differences before correction. No adjusted score is claimed for the
unmetered diagnostic.

Conclusion: teacher-forced K3/K4 CV improvement did not become closed-loop
final-mean improvement. Stop the frozen26 mechanism without lambda/scale
scanning or promotion. Evidence:
`results/phase2/local_joint_closure_rollout_r49_network_audit.json` and the
immutable `modal_local_joint_closure_rollout_r49/` raw manifests/journal.


## R50/R51 fixed six preactivation graph-context features

The new single change appends six target-free graph contractions to the R48
26 features: rho2 row mean, rho3 row mean, normalized diagonal triangle,
rho times left/right K21 row means, and rho2 times K22 row mean. The helper
uses permitted flopscope arithmetic; C_off, D21, wk4m and sigma are observed
before the original nonlinear rider correction and never mutated. Graph
features are canonicalized with the same inverse permutation as the original
22 features and own post states. Layer0 is exactly zero, layer15 is skipped.
Small actual-width16/32, prune on/off observer checks found unchanged outputs;
this is a component check, not a full candidate or resource pass.

R51 source `modal_local_joint_pre6_capture_r51.py` froze SHA
`778645219a8b9e41e9e262891181ffc4d5f13b853076411906c51b1a47f01c40` and launched
PID1684090 at `2026-10-13T11:40:11Z`. No corpus shard or new baseline is
created. The worker reconstructs the same24 teacher-forced trajectories with
original MC post-state resets, first runs0/16 smoke, then the remaining22.
Only graph15x1024x6 is stored; old R46/R47 state/features/scales are intended
for reuse. Both smoke networks passed finite/source-weight/hash/nonmutation
integrity checks. However, network0 showed absolute differences from R46:
features26 max0.0041660234, state max0.000732421875, prediction max3.5763e-7;
network16 was exact. Other returned networks also show differences. These
are retained as numerical-path diagnostics, not accepted trajectory parity.

The planned R53 fixed26 versus32 CV keeps R48 ridge0.01 and network splits,
with a stricter graph signal gate: at least one channel32/26 normalized
ratio<0.90, at least16/24 wins, and paired-network-bootstrap95 upper<1.
Mean/variance results must be shown explicitly. Fit is not launched until
the new graph and reused old26 trajectory pairing passes its stated check;
current absolute state/features differences do not pass. No gate is silently
relaxed, and no graph correction is promoted from these capture checks.


R51 completed24/24 with zero worker failures. Every compact payload passed
hash, finite/shape and analytic-first-layer-zero checks; stored graph data
is15,710,116 bytes total. The stated legacy-pair absolute gate passed14/24
and failed10/24. Maximum differences across24 were features26=0.01220452064,
state=0.00244140625 and prediction=3.75509262085e-6. Capture does not mutate
input arrays, but backend/container arithmetic plus central high-order
cancellation can still change captured own states. Final-mean differences
alone cannot certify identical high-order trajectories. R53 has therefore
not run, and no old26/newgraph mixed-trajectory model is fitted. Evidence:
`results/phase2/local_joint_pre6_capture_r51_network_audit.json`. A scientifically
valid continuation needs own26/state/scales captured on the same graph
trajectory, or a separately justified invariant-preserving comparison;
increasing an absolute tolerance without that evidence is not a repair.

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

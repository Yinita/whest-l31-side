# R52: frozen final-only 26-feature readout gate

2026-10-13. H18 remains the frozen anchor, tenfold stage1 is unmet. This is one
fixed diagnostic and does not reopen a general ridge/degree/feature sweep.

## Why this is distinct and what repeats

L03/v30.098 already fitted a final-only13 affine model on V29 with Mini8
network leave-one-out. Its full charged CV raw2.26975299e-8→2.28283026e-8
(+0.5762%) rejected it. L05/6/7 threshold/skew/kurtosis additions were also
negative. Therefore final readout itself is not new. R52 changes the training
evidence to independent random-network corpus TRAIN24, the anchor to H18,
and uses one previously fixed26 feature definition with the stored float64
`final_mean` target. It tests a specific missing observation; it is not evidence
that more scalar features or more data guarantee tenfold gain.

R49 local26natural closed fresh8 raw2.32441291884e-8→2.33023356538e-8
(+0.2504%),4/8wins,0fail; paired network-bootstrap95ratio
[.97347164,1.03421333]. Earlier repeated total-state corrections were worse.
Final-only total error can be added once without propagating a corrected
K3/K4 birth state or repeatedly adding inherited history. It cannot repair
unknown full joint state, and may still have negligible predictable signal.

## Fixed specification before fitting

- Train IDs0,16,…368 only; frozen corpus revision
  `cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8`, existing private TRAIN labels.
- H18 SHA `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`.
  Same torch2.5.1CPU full randn16×1024×1024, seed770000+idx; every weight hash
  must equal the previously captured baseline. Original H18 predicts unmodified.
- Capture final15 original pre22 and ownpost4/σpowers. PK1rawp1..p4 are
  converted to ordinary marginal cumulants in float64. Effective ownpostmean
  is the actual parent's float32 `pk1+delta`, subsequently promoted; generic
  middle-layer float32 centering is not silently substituted for this definition.
- Final H18 `trim` leaves C_off/D21 unavailable. D21norm is exactly zero;
  no32graph features or restored final mixed slices, which would change cost.
- One linear26 ridgeλ.01, intercept unpenalized, training standardization.
  Target `(float64final_mean−actualH18final)/presigma`. Final prediction
  `(parent+sigma*normalized_delta).astype(float32)`; no mean floor or clipping.
- Four fixed blocked folds6whole networks;18train/6held. Gate rawratio<.90,
  at least16/24networkwins, paired network-bootstrap95upper<1. Bootstrap
  seed520013/10000samples. No hyperparameter or feature/layer selection.
- Only if this gate passes: freeze one TRAIN24 model, fresh unseen network
  final26target-free capture and paired scoring, then lawful fnp whole parity
  and strict RPC. No benchmark labels fit, no adjustedscore/stage claim from CV.

## Actual capture status

`modal_final_readout_capture_r52.py` started PID1685132 at19:41:29. Two smoke
workers completed0runtimefail with weight hashes matching, analytic firstMC
noise ratios.97037505/.98681048, captured feature mean equals returned final
exactly. The preregistered numerical comparison gate `maxpredictiondiff≤1e-6`
failed: both maxabs4.29153442e-6 versus old unchanged H18 capture, finalRMS
about9e-7. Failure is preserved in originalcompletion/smoke_gate and
`smoke_numerical_audit.json`; the22remaining did not start in this version.

Differences begin at analytic layer0(~2e-7), smoothly grow through depth, with
all values finite and no injection. This is consistent with cross-container
float32 kernel drift; its causal origin is not proved. Old/new raw differs by
~.04–.05%, far below proposed10%gate. Any continuation must state a revised
numerical diagnostic bound explicitly, retain original failure, reuse the
two smoke captures, and compute paired fit/CV against each capture's own
parent prediction. It must not call this exact whole parity or strict acceptance.

R52r02continuation explicitly sets numericalcapture bound1e-5 only, reuses
original2verifiedsmoke, and captures remaining22 with no algorithm/feature/fit
change. Original failed1e-6gate remains untouched. PID1689444,new output
`results/phase2/final_readout_capture_r52_r02/`; source and local trainer are
lint/compile verified. The trainer uses each capture's own parent returned
prediction and features, never joins an older baseline prediction to new
features. Observed~.05%drift makes tiny improvements at that scale
uninterpretable; the fixedCV gate remains a much larger10%raw improvement.

### Full24 R52r02 numerical comparison also fails

24/24capture succeeds0runtimefailure, all weight/source/tool versions and
metered480,763,746,423FLOPs match oldcaptures. All samecapturefeaturemean
equals its returnedfinal exactly. Oldcomparison1e-5gate nevertheless fails
on indices32/48/160/176/208/256/304; maxdrift3.12328339e-5, maxfinalRMS
7.70539008e-6. These nonidentical runs start with layer0differences~2e-7 and
exceed1e-5 at layers10..14. FirstanalyticMCnoise ratios all.858..1.077.
No third numerical tolerance change is made. Both originalfailedgates remain.

Old/new aggregate raw2.21211547022e-8→2.21192411443e-8(ratio.99991350)
is numerical drift, not algorithm gain. Largest per-network rawdrift.6374%,
largest old-new residualMSE5.93730363e-11. DriftRMS divided by baselineerrorRMS
is max5.0274% and pooled2.1679%. A CPU/BLAS/QR numerical cause is plausible
but unproved without host metadata and samehostreplicates. Full evidence is
`final_readout_capture_r52_r02/full24_numerical_failure_attribution.json` and
`r55_samecapture_preflight.json`. This newcapture cannot establish oldbaseline
parity or candidate migration, even though each capture is internally paired.

### R55 samecapture-only fixedCV: negative signal gate

After the explicit data-boundary audit, a separately named R55 trains only
the newcapture26features with each samecaptureparent prediction. It retains
the original fixed26/.01/4blockednetworkfolds/gate; no oldprediction join,
no heldout labels, no tolerance override or candidate claim. Script
`train_final_readout_cv_r55.py`; output`results/phase2/final_readout_cv_r55/`.

OOF raw2.21192411443e-8→2.20310455167e-8, ratio.9960127191(−.3987%),
15/24wins, paired network-bootstrap95[.9920517108,1.0003000591],0failure.
All three fixed requirements fail: ratio<.90,>=16wins,upper<1. The small
estimate does not support a reliable improvement and numerical/reference
uncertainty remains.1420negative outputunits are recorded; no floor/clip
was added to the preregistered affine readout. No fullTRAIN24model is fitted
or retained, no freshheld rollout or strictRPC launched. This rejects this
specific fixed scalar finalreadout mechanism; it does not exclude models
with genuinely new joint/directional or dynamic information. Best remainsH18.

The worker receives only idx/expectedweightSHA and never reads MC labels or
models. Original local labels are used only outside workers for identification,
analytic noise and later fitting/scoring. Offline memory16GiB/NumPy tracer
does not establish8GiB/.4sresidual competition compliance.

## Joint context inference, separate from R52

On24TRAIN partialdiag MCoracle, raw2.21211547e-8→4.45453077e-9 (~4.97×).
This is one intervention, not a general ceiling: mixedcentralK11/K21/K22,
nonlinear source feeds, harmonic K4 proxy and final15own closure remain
approximate. True marginal MC labels cannot identify full covariance or
third/fourth joint transport. The original [Wu et al. paper,
§4/S.4](https://arxiv.org/html/2605.05179v2) tracks joint cumulant contractions
and distinguishes repeated-index power cumulants; marginal features alone
omit this structure. This motivates the separate fixed directional graph
context R50/R51, whose component cost has been measured, rather than λ scans.
It does not prove those graph descriptors predict residuals or yield a candidate.

Raw diagonals corrected while mixed central entries remain fixed form a
formal approximate closure. PSD of the sampled covariance boundaries is
helpful but does not establish realizable full joint moments. Keep this
limitation through every learned rollout and production source ledger.

# R56 normalized on-policy candidate preparation

Preparation only. No full sequential TRAIN14 model is available at this
checkpoint, so no estimator directory/support/model/candidate is generated.

`build_onpolicy_closure_template_r56.py --audit-output <new-json>` without a
model writes an audit template and exits. Building requires a frozen private
model path, independently supplied expectedSHA256, and a new outputdirectory.
The model must declare26features, TRAIN0..368step16, layers1..14, first0/final15
off, ridge.01, σoutputpowers1..4, meanfloor0/varfloor1e-10, corpusrevision/H18SHA,
and sequentialonpolicy provenance for each head's previousprefix and capture.
Incomplete R54pilot prefixes1..3 are rejected. A model passing this schema is
not automatically accepted as effective or a valid fulljoint closure.

Generated body derives from frozen R48H18 body SHAe177526a… and changes only
support loading/modelidentity/normalized-head evaluation. It stores private
float64 means/scales/theta and evaluates `((features−mean)/scale)@theta`,
preserving R54operation order rather than folding affine coefficients. Existing
fnp marginal injection, deterministicmean rider once, centralmixed slices and
original nonlinearfeed semantics are retained. Helpers/features/updates use
fnp only; JSON/hash/fileI/O occur in setup. No NumPy/torch/fitting/labels in
generated prediction. Coefficients/metadata are ignored locally pending data
redistribution permission. No Git commit or upload is performed by the builder.

`onpolicy_support_contract_r56.py` checks fixture normalizedsupport on two
existing actualsaved26feature trajectories, without reading labels/reference/
prediction or an actual learnedmodel. fnp versus NumPy normalizeddelta
maxerror<1e-12; six schema negativecontrols reject missinglayer14, includedheld
network, wrongfloor, wrongfeatureorder, missingprovenance, nonpositivescale.
It is a component/supportschema result, not trainedmodel/wholeprediction or
strict resource acceptance. Reports: `onpolicy_closure_template_r56_audit.json`
and `onpolicy_support_contract_r56_r01.json`.

Before using any future generatedcandidate: freeze actual fullsequentialmodel
and evidence, show targetfree wholeprediction parity with samecycleproduction
H18, collect freshnetworkrawsignal, then strictRPC Mini8/independent adjusted
scores/resourcechecks. Do not port a negativepilot merely because the builder
works. Stage1target remains<5.779e-10 and tenfold is unachieved.

R54earlynoise interpretation is separately audited in
`onpolicy_early_noise_gate_audit_r54_integration.json`. Mean layer1 observed
residual is~97.26% nominalMCnoise; pooledphysical1..3 mean~85.45%noise,
variance~65.45%. Actualnormalizedgate has different σweights. Earlygate failure
can stop a fixedpilot for economy but cannot identify a whole14layersequential
method ceiling; late mean residual/noise is much larger. No threshold or model
change follows from this read-only interpretation.

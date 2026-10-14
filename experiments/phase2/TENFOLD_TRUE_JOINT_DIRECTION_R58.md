# R58 true joint directional pilot

This changes the research target. Public post-four marginal labels cannot
identify cross-neuron covariance or joint third/fourth cumulants. R58 measures
their contractions along the actual next weight vectors, using samples of
the complete unchanged network rather than an independent Gaussian surrogate.
It fits no model and gives no candidate or score improvement.

The two identities are fixed public development networks already used by the
project: Mini global0 `logan-fitzgerald`, original NPZ SHA7fb2e3f5… and shard02
global16/local0 `adam-morrison`, original NPZ SHAabc7991f…. These are not corpus
idx0/16, new training networks or an independent holdout. The original official
reference arrow files and revision aa99830f… stay unchanged and hash checked.

At zero-based pre layers8/14/15, choose the64 canonical original output IDs
0,16,…1008 before seeing any MC outcome. Each sample supplies preceding post
activations X, the float64 scalar projections Y=X@nextW[:,columns], and the
actual float32 forward preactivations. The same samples produce1024 marginal
cumulants of X and64 directional cumulants of Y. For p2/3/4 the joint target is

`Jp = kappa_p(Y) - sum_i nextW_ij**p * kappa_p(X_i)`.

The mean difference is a linearity/numerics control. No n² covariance or n³/n⁴
tensor is created. Directional J2 includes every cross covariance contraction,
while J3/J4 include all omitted mixed third/fourth contractions. They do not
reconstruct the full joint distribution. Subtract the same diagonal expression
from actual H18 prestate to distinguish the marginal contribution from joint
error. Preserve the same-sample dependence when propagating noise.

Float32 canonical weights, Gaussian inputs and network forward match the old
CPU MC diagnostic. Selected projection is float64 to separate reduction
roundoff. Float64 centered Pébay batch/parallel merge avoids raw-power
cancellation; unbiased k-statistics are calculated after full stream merge.
Every raw shard keeps count/mean/M2/M3/M4 and first-layer calibration statistics.
Two independent seed streams each use64 shards×65,536=4,194,304 samples per
network, batch256. Total256 jobs,16,777,216 network samples. OneCPU/2GiB per
compute worker, maximum96,300s timeout, zero automatic retries. Compute workers
do not commit a Volume; raw compact output first returns durably to localdisk,
then an independent serial archival worker commits batches8. Archive latency
is separate from MC compute latency. Keep failed evidence and no missing-shard
success declaration.

Existing covariance_causal_r02 actually returned1015success/9timeout out of1024,
66,519,040 effective samples:33.605CPU hours,2025.25s wall at64 workers. Per65,536
full16-layer pre/post shard mean119.19s,median118.57s,p95147.45s. At that measured
throughput R58 MC256 predicts8.476CPU hours; three96-capacity waves are roughly
6–8minutes plus coldstart/baseline/archive. New compact statistics and64
float64 projection GEMMs have not been measured, so this is an estimate.
Official standard Functions rates imply approximately$0.535 for those MC
CPU+2GiB seconds; actual account billing, baseline16GiB costs, failures,
preemption, region multipliers and archival overhead are unknown and extra.

Pilot1 runs both actual H18 capture/control pairs and four smokeMC shards
(network×stream shard0) first. Exact source/weights/columns/shapes/finite hashes,
split-batch Pébay relative error<1e-11 and projected f32/f64 difference MSE<1e-10
must pass. Traced versus untouched H18 sameworker prediction RMS divided by
official error RMS must be<.01, and raw control drift<1%. First-layer exact
Gaussian mean/variance calibrated MC mean-noise ratio must lie.2..4 pernetwork.
This is a numeric/data gate, not an accuracy selection. Only then run252 shards.

Pilot2 examines fixed pre14 jointK2 readout effect. Both network two-stream
cross signal energies must be positive and average-stream noise proxy/signal
<.25. Eight whole-direction MC chunk pairs give a diagnostic bootstrap lower
bound>0 for jointK2 signal. Retain all64 direction values, all three layers,
all channels, positive physical errors and failures. K3/K4-only signal cannot
pass the covariance gate. Cross-noise cancellation estimates signal energy;
negative estimates or intervals reflect uncertainty, never negative MSE or
an improved score. Two networks/eightMC chunks do not establish generalization.

Future causal inputs are separately named32, rather than existing ownpost26:
pre22 with original pk1slot replaced by Gaussian ReLU mean available before
nonlinearity, four previous post-diagonal nextW^p contractions divided by
current sigma^p, and six current pregraph features. No current ownpost is
available before prewick correction. Layer15 trims Coff so graph is zero and
marked unavailable; use its truth only for direction/numerical diagnosis.
Actually deploying a learned directional correction requires network-separated
training with independent fresh truecross labels, full legaloperator parity,
closed-loop benefit, strictRPC/resource checks and original tenfold stage
criteria. This pilot cannot itself satisfy them.

Original sources: [Pébay SAND2008-6212](https://www.sandia.gov/research/publications/details/formulas-for-robust-one-pass-parallel-computation-of-covariances-and-arbitr-2008-09-01/)
for centered parallel moments, and [Modal official pricing](https://modal.com/pricing)
for standardFunctions rates. Source/input freeze is
`results/phase2/joint_direction_preparation_r58_r02.json`; toy mathematical
contract is `results/phase2/joint_direction_contract_r58_r02.json`.

2026-10-14 repository checkpoint: original R58 module import failure is retained, no MC samples returned. r02 immutable prepared driver inlines configuration and decouples archive; its result directory is absent at this checkpoint. Prepared helper/contract syntax verified; no new real-network result, model or gain claim. Source hashes remain unchanged during cleanup.

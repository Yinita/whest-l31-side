# tenfold radial sphere r12

Frozen parent V37bH18: `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`.
Candidate SHA256: `cf0d3c001b0f542a3f6ffc23ab0c1169290ba023d951a72ea35a36d5a22699a2`.
LICENSE/NOTICE preserved. Sole primary mechanism: isolate the exact independent
input radius using positive homogeneity, approximate one spherical residual
distribution, then multiply every output mean by `E[chi_n]/sqrt(n)`.

Gaussian first-ReLU scalar raw moments and pair power cumulants are converted
through raw-moment degree factors a1/a2/a3/a4. Pair inputs are not directly
divided: raw cross moments are restored first, divided by their total degree's
radial factor, then centered against the new scalar raw moments. Sphere mean,
covariance, K4 marginal, and K22 replace the firstpost state; existing memoryless
harmonic regeneration uses the converted K4/K22.

The complete represented parent firstpost K3 tensor is divided by a3 via its
linear coefficients w2, s, e, and residual R columns. Covariance carrier A/P is
preserved. Two extra full channels are added:

```
B = C_Gaussian_firstpost, v = mu_Gaussian_firstpost
k = 1/a3-1/a1
c = 1/a3-3/a1+2/a1^3
T_extra_ijk = k*(B_ij*v_k+B_ik*v_j+B_jk*v_i)+c*v_i*v_j*v_k
```

First channel transport is raw W (it is already post-ReLU); later transports
use the same expected-gain linear map WD as the parent K3 chain. Full B is
transported by WD B WD.T and v by WD v. Their complete D3/D21 are added to the
parent readout before its Wick/nonlinear feedback and birth residual subtraction.
This extra channel retains every row, including rows omitted by the parent's
shape pruning. Existing parent source pruning and confinement remain approximate.

The tensor transformation preserves the complete **represented** Gaussian
tensor, including its inherited rank-residual approximation. It does not replace
it with the true Gaussian/sphere tensor. In particular the parent's marginal/pair
represented-K3 mismatch remains an approximation; no oracle or exact-repeated
replacement is added. Gaussian-trained LAM/REF_R remain unchanged and may be
miscalibrated under spherical residual law. This is not full K4 propagation.

Actual n8/n32 parent/candidate firstbirth full tensors and actual second-pre D3,
D21 agree with independent dense transformed tensors within float32 error
(max birth 1.02e-7, max slice 3.60e-7). Sphere mean/C/K4/K22 conversions verified
independently. Evidence: `results/phase2/tenfold_radial_production_contract_r12.json`.
Instrumented smallshape contracts are not resource acceptance or benchmark scores.
The initially incorrect first transport (WD instead of raw W) failed the contract;
failure logs retained and corrected before freezing this candidate.

All algorithmic array operations use allowed metered flopscope operators. No
reference/targets, compiled kernels, concurrency, or meter interventions.
Generic depth1 remains the original Gaussian firstlayer output; depth>=2 enables
the transformed residual chain. Full real1024 score/time/residual/memory remains
subject to strict evaluation with original 2**41/120s/.4s/5s limits.

Strict supervisor clears V17..V29 environment overrides, and the source default
`NO_CORR=True` disables the empirical mean rider in both parent and candidate.
Nonzero CORR_BETA entries do not imply their application. The nondefault
`V17_NO_CORR=0` probe would overwrite its firstlayer delta during sphere conversion
and is outside evaluated scope; `NO_SRC_LAST` probe is likewise not supported.

Real network0 strict evaluation completed with zero failures: parent raw
`2.2281179568039988e-8`, candidate `2.2248848097206064e-8` (only −0.145%), adjusted
`4.8714160493622575e-9 -> 5.5192238827473855e-9` (+13.30%). Increased cost cancels
the tiny raw difference. This candidate is rejected, no next radial scale sweep,
no promotion or online upload. Exact score/resource evidence is maintained by
the root evaluator; contracts remain distinct from scored production checks.

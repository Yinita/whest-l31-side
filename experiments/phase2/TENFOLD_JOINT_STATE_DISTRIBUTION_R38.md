# Joint state correction must describe one distribution

The finite-distribution contract `tenfold_joint_state_distribution_contract_r38_r01.json` verifies deterministic shift, affine scale/shift, independent coordinate noise, CP3 row scaling, and next-linear mean/covariance to <=8.88e-16 using an arbitrary 23-point nonnegative joint distribution. It reads no benchmark labels. An explicit wrong implementation that keeps raw powers fixed but recenters around a new mean changes covariance erroneously by0.19869.

The corpus four-channel ridge pilot may measure held-network predictability of postmean/variance/κ3/κ4 residuals. Arbitrarily copying all four diagonal values into a centered state is a formal closure intervention; it has no automatic common-distribution interpretation. Marginal variance positivity alone is insufficient, and unchanged mixed pairs plus changed means must be interpreted consistently.

## Smallest closed correction

A deterministic post-nonlinear shift Y'=Y+b changes only K1. All central joint cumulants of degree>=2, including K21/K22 and CP3 sources, stay unchanged. Apply after ordinary cumulants were formed; the original raw/power moments and Wick table continue to describe the original ReLU birth, so `e_b=w12−2*oldpk1*w1` must keep the old pk1. Do not substitute the shifted mean into that expression without transforming the associated raw power moment as well. The next linear mean is W(mu+b); covariance/source histories are unchanged. This is an exact distribution operation and a useful first closed rollout. Negative b can put mass below zero; it is a distribution on R but no longer necessarily a nonnegative ReLU output. Support preservation requires a bound on b or a separate nonnegative affine/noise construction.

## Affine correction

For Y'_i=a_i Y_i+b_i, K1'=aK1+b and every central κ(i1..id) multiplies by ∏a(indices). Full C changes as diag(a) C diag(a), not only its diagonal. Every CP3 source factor row gets a once; CP4 factors, cumulant feedback, covariance carrier and all source age representations must scale consistently after complete source assembly. Mixed K21 scales a_i² a_j; K22 scales a_i² a_j². A lawful fitted correction can estimate a,b and derive its 3/4-order prediction, but independent four-channel predictions generally do not coincide with a two-parameter affine operation. Positive a and nonnegative b preserve nonnegative support.

## Independent noise correction

Y'=Y+Z with independent coordinate noise changes K1 and diagonal higher cumulants by the cumulants of Z; original mixed cumulants stay unchanged. This can be represented by new diagonal source births. Noise variance must be nonnegative. For centered noise v=κ2,s=κ3,t=κ4+3v², the moment matrix `[[1,0,v],[0,v,s],[v,s,t]]` must be positive semidefinite, requiring `κ4+2v²>=κ3²/v` for v>0. Unconstrained ridge changes can violate this; negative variance correction cannot be independent-noise addition. Marginal PSD is still not a proof of arbitrary joint cumulant realizability. The finite product-noise contract verifies correct κ addition and positive moment matrices.

## Production ledger boundary

At the H18 posttrigger before `mu=K1v...`, K21/K22 have already been centered using original raw pk moments. Deterministic translation at this point is consistent with retaining these central pairs. Changing K2/3/4 diagonals at the same point may rebuild Cdiag, S3c and harmonic G, but it does not prove mixed support belongs to a valid joint state. Any closed four-channel pilot should label this as a formal diagonal closure intervention unless a lawful operator and every central-carrier transform are explicitly implemented. Pilot scores are offline corpus evidence, not Mini8, independent competition, RPC admissibility or tenfold-stage acceptance.

R02 extends the finite contract with the exact power-cumulant centering formulas for every offdiagonal pair, before and after true shifted samples. K21=`pK21−2μ_i pK11`; K22=`pK22−2μ_i pK12−2μ_j pK21+4μ_iμ_j pK11−2pK11²`. All pair formula errors<=9.02e-17. Central K21/K22 are shift-invariant, while pK21/pK22 change; this is direct evidence for keeping the original birth raw-power ledger when applying a deterministic shift after centering.

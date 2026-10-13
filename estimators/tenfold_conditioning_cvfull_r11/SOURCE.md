# tenfold.conditioning.cvfull.r11 (prepared)

Parent r10 CVdiag. Candidate SHA256 `bbf7bafbd9fd93d431290b6da6b17776de79dcf42aeee6d57f90ecb2873e949d`. LICENSE/NOTICE unchanged.

Single major change: fixed GH8+zero cheap Gaussian response passes fullcov firstchaos rather than diagonalcov after each ReLU. Expensive t0 conditional parent, weight-only direction, nodes, parentriders and sharedBudgetContext unchanged. Covariance/weightsfloat32, marginalvectorsfloat64. FirstCov conditional Gram−qqT shared acrossnodes; firstmarginalvar exactlyr10 float64 sumW²−q64². Nodes streamedoneatatime; ninecovstates notretained.

Cpost firstchaos formula: gate_i*gate_j*Cpre; diagonal setto exactGaussianmarginalReLUvariance (higherchaosdiagonal countedonce). This is not exactbivariateGaussianReLU covariance. Covariancepre W.T C W symmetrized for numericalroundoff. Float32recurrence is a numericalimplementationdifference fromr10 diagonalvectorfloat64, explicitlyincludedinmechanism.

Bounded n8/32/64 depth16 independentmixedfloatrecurrence contracts passed maxerror1.81e-9; script experiments/phase2/tenfold_conditioning_cvfull_contract.py report results/phase2/tenfold_conditioning_cvfull_contract_r11_r02.json. n64 extrapolates1024 cheap FLOPs `585536757760` (26.63%budget), plus parent~20.7% gives~47.3% total; fixedGH8 preserved. This is estimate, not measured1024 resource. Singlethread wall mayexceed120s despitebudgetfit; statusprepared untilstrictactualpilot.

No reference accessed/no newscore/no heavy benchmark. Toy firstchaos improved3/3 knownn3 cases butdoesnotprovegeneralization.

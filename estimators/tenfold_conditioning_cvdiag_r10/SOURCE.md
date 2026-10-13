# tenfold.conditioning.cvdiag.r10

Parent frozen r09 SHA `1e67ccba8497913208c827074786cf00f89bcaea5880be36f3a08801cb4e9d3b`; original V37bH18 MIT LICENSE/NOTICE unchanged.

Candidate SHA256: `cc42a1ea8644f12839b32a9e6fb3b6f547620fbf70f5633c0b64c6b5dae6705f`.

Scientific mechanism: one complete parent conditional run at t=0, inputCov I-u u.T, plus fixed GH8 cheap diagonal-Gaussian mean response average(t)-response(0). Direction weights-only and all parentriders/gates/source/pruning frozen. The conditional parent and all9 cheap nodes share original budget/time context; no setup/reset/nestedcontext/fullparentpilot. Only1024×16 wrapper applies; genericshape originalparent.

Cheap skeleton firstvariance is diag(w0.T w0)-q² with q=w0.T u; firstmean=tq. Laterpremean=w.T mean, prevariance=(w²).T variance. Gaussian ReLU exact marginalmean/variance formula; higher correlations/conditional cumulants omitted. First variance/weight squares legally computed once and reused. Returned correction converted float32 to parentoutputdtype.

This is an approximate deterministic integration-difference correction, not an unbiased knownzero control variable. A toy gate already found one diag-response degradation. GH8 finite integration bias, conditional father empiricalrider bias and omitted postcorrelations can dominate. Do not use unconditioned originalparent plus this correction, which would doublecount rankone input effect.

Bounded reproducible script experiments/phase2/tenfold_conditioning_cvdiag_contract.py passes n4/8 independent skeleton, zero-u originalparent reduction, parentpoolalias and finite outputs. Firstlayer GH8 vs analytic unconditioned mean recorded: n4 truncation5.78e-4 for arbitrarily strongdirection, n8 5.76e-10; no claim of exact quadrature. Report results/phase2/tenfold_conditioning_cvdiag_contract_r10.json.

No scores/resourcepass claimed until strict runner. Previously frozen r09 remains unchanged.

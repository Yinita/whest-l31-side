# r24 marginal Edgeworth gain for represented K3 history

Frozen parent V37bH18 SHA256: `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`.
Frozen candidate SHA256: `7ff47e2a24fae92837b0229a8040318c1e2ab979e5582809f27e5eca6ed86a94`.

The single primary change replaces the Gaussian derivative Phi in older
represented K3 history transport with the fixed first-order marginal Edgeworth
expected ReLU derivative:

```
gain = Phi + D3 * (alpha² - 1) phi / (6 sigma³)
           + g4 * (3 alpha - alpha³) phi / (24 sigma⁴)
```

For X=mu+sigma Z and f=ReLU, the cumulant exponential gives
E[f′(X)] = E_G[f′(X)] + kappa3 E_G[f⁽⁴⁾(X)]/6
+ kappa4 E_G[f⁽⁵⁾(X)]/24 to first order. Hermite integration of the formal
Edgeworth density over X>0 independently verifies both signs and coefficients.
D3 and g4 are inherited online marginal estimates. No fitted coefficient,
test/reference access, clipping or gain-dependent mean correction is added.

The gain reaches dense young A/P row prescaling, shared Q/QP basis transport,
thin residual/feed Z, and thin feedback Zf through the same w1_prev/WD path.
D3_w = D3 gain³ and D21_w = diag(gain²) D21 diag(gain) remove the same
inherited tensor from newborn residuals. Newborn birth diagrams retain their
Gaussian Wick weights; new source A/P/Z/Zf make their first transport with raw
W. The post-ReLU covariance C carrier makes its sandwich with raw W. Existing
Gaussian Wick tables, pK evaluations, parent pruning, confinement and K4
regeneration remain inherited.

This is an experimental extension of the parent approximation. Products of
three corrected gains retain higher-grade terms and are not advertised as
missing official budget3 terms or a complete nonlinear tensor closure.
Public cached V29 network0 layer14 corrected gains span
−2.6356797291e−5 .. 1.0000296226, with 82 negative and 93 above-one gains;
RMS change .00130914623, max absolute change .00553651169. These signed formal
Edgeworth values are reported rather than clipped.

The initial contract report verified the same candidate logic with SHA
`b37861ff6d0bd781d5fab4c31836bf29c63133fc2677e4f9a18bde97e465e23a`;
the frozen candidate differs only in blank lines before the new helper.
Its executed diagnostic source is preserved as
`experiments/phase2/tenfold_nongaussian_k3_gain_contract_r24_executed.py`
(SHA `211e0b7c2954faca658ca0a9c354cb03d308320b42e84094aa85ec3d09f7fdd6`).
The report retains this original identity and is not overwritten.

Formal indicator/Hermite quadrature errors are <1.8e−15; independent dense
CP transport/centering n4/8/16 errors are <1.4e−15. Whole-predict n32/n64
depth4 smokes are finite. Actual candidate site integration is recorded
separately. Actual frozen-candidate line-trace n32depth6 runs with confinement off/on
verify dense young gain prescale, WD, QP join, Z/Zf residual/feed transport,
newborn raw-W A/P/Z/Zf, covariance raw-W sandwich, independent full represented
CP D3/D21 slices and consistent birth subtraction. Max CP slice absolute error
3.30e−7; no gain-site integration defect was found. Initial validation harness
failures (comment-line event and empty source slice) are preserved separately;
neither changed production code.

Strict RPC1 network0 is negative: raw MSE2.2659300213945244e−8 versus
2.228117956804e−8 (+1.69704%); adjusted4.9540952601586264e−9 versus
4.87141604936e−9 (+1.69723%). Cost480,781,426,807FLOPs, wall44.84788s,
residual.199365s, zero failures. No gain/age/rank scaling sweep or broader
confirmation is selected. The tenfold goal remains unachieved.

Primary sources: cumulant differential expansion and ARC's pinned K3 transport
implementation at commit `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`:
https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py
and the inherited public V37bH18 implementation. ARC's official cap3 accounting
does not establish this gain product as an omitted official cap3 mechanism.

# tenfold.precorelambda.alllayers.r07

Candidate SHA256: `b20fd6a1044c462914acabb1c01b0ccd98c337e28583bc66585f9fbd9b1ce005`.

Derived from local r03 (`tenfold_corelambda_li14_r03`, SHA256 `3c071df2f3d6fd0b22f0be0d21b43f611d17eae4cd6d8a240719697603a010fb`) and the MIT-licensed public V29/V37bH18 lineage. The inherited MIT license accompanies this candidate. Diagram formulas are checked against the official `mlp_cumulant_propagation` source frozen at commit `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`, especially `factor_k3.py` lines 330–355, 395–453, 459–533, and `kprop_harmonic.py` lines 276–313.

Hypothesis: the original LAM table targets the next **preactivation** covariance, whereas r01–r05 fitted postactivation covariance. This version instead fits the full represented post core after transport by the actual next weight matrix. It applies the same prescribed operation continuously at layers 1–14, with no reference access, layer selection, source truncation, or output-column sampling. All source birth columns are retained and old sources use the r03 shared-basis contraction.

For next weights `W` and post covariance `C`, form `D=off(W C W.T)` and `H=W.T D W`. Compute the numerator through the adjoint contraction of the complete represented post core with `H`, including the post-core diagonal, and divide by `sum(D*D)`. Input-by-output weight storage and the current input permutation are handled explicitly. A future output permutation leaves this Frobenius fit unchanged. The scalar is cast to float32, matching parent precision. The V25 beta fitted to the old LAM table is bypassed only when this new scalar exists; the first-layer parent path remains.

Limits: this still regenerates `diag(g)+lambda*Coff` on the next layer rather than preserving dense transported core; inherited source/rank and average-metric approximations remain. Small algebraic contracts do not establish actual 1024-width accuracy, official resource compliance, adjusted-score improvement, or the tenfold goal. Root must run the frozen candidate under unchanged strict RPC constraints. Post covariance uses `pk2v-pk1v**2` and symmetric `pk11`, consistent with the inherited default NO_CORR state.

Independent transport evidence: `results/phase2/tenfold_prels_transport_contract_r01.json` (candidate hash bound), n8/n32 direct dense versus adjoint/source-scalar absolute errors `1.3552527156068805e-20` / `1.1117307432712692e-21`; actual weight orientation and input/future output permutations checked. No reference labels or adjusted score were used.

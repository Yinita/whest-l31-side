# R59 complete implementation parity

Completed 2026-10-14 15:40 CST: R57fullfresh8 raw+2.259982%,4/8wins,0fail; rawgate failed, fixed26 linear on-policy mechanism rejected, no strict expansion/upload. R59full16 paritypassed maxabs0. Earlier checkpoints below are historical.

Current checkpoint 2026-10-14 15:34 CST: actual R61 completion/full14 verified; R62 actual wrapper and R56 component/source audit passed. R57fresh8 and R59wholeparity launched with frozen identities. No new raw/adjusted or strict resource acceptance yet. Earlier preparation-only descriptions below remain historical.

## Earlier preparation checkpoint

R54 stopped with a retained 300-second FunctionTimeoutError at layer8,
full24 idx320, after 815 successful captures. Its complete full14 model does
not exist. R56 wrapper/build, R57 fresh8 and R59 actual-model parity therefore
remain unstarted. R59 cannot replace missing heads, invent completion, or
accept the partial through7 model. A separately evidenced continuation must
first freeze all heads1..14 and bind their actual capture provenance.

`onpolicy_whole_observer_r59.py` is a separate target-free diagnostic observer;
it does not modify the frozen R54 helper. It injects exactly the frozen
normalized26 heads1..14 into original H18's ordinary post-centering state,
using the same allowed fnp order `((features-mean)/scale)@theta` and signed
central marginal update. It skips first0 and final15, records all16 prestate
boundaries, preserves original pk1/K21/K22, and returns a complete16-layer
prediction. It never stops at an intermediate layer or reads labels.

`modal_onpolicy_whole_parity_r59.py` compares this original-H18 observer to
the real frozen R56 candidate on fixed TRAIN idx0, seed770000, setupseed0,
one full16×1024×1024 torch2.5.1 CPU weight tensor in one worker. The worker
clears V-number environment overrides, asserts default NO_CORR, checks source,
helper, model and support hashes, and uses separate setup/predict budgets for
instrumented and actual predictions. The former is diagnostic instrumentation,
not a submission. Per-layer RMS/max error, full output hashes, FLOPs/backend/
overhead/residual timing, versions and cumulative RSS are preserved. Fixed
whole-prediction tolerance is max absolute error `<1e-6`; it is not widened
after observing results. No raw score or reference enters this check.

The diagnostic uses 1 CPU, 1 thread, 16GiB, zero retries and a total600s
timeout for two predictions. These are not official strict resource limits.
The phase order is observer then actual, so cache effects and cumulative RSS
must not be called resource acceptance. R57 raw evidence, strict8GiB RPC,
Mini8 adjusted score and independent tenfold gains remain separate gates.

`onpolicy_whole_observer_contract_r59.py` passed a synthetic n7 boundary/update
fixture and three invalid-layer-set negative controls. This only verifies
component behavior; it is not actual frozen-model or whole-prediction evidence.
All four new source files pass lint and compilation. The launcher privately
loads already authorized Modal credentials; no model means no cloud launch.

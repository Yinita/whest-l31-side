# tenfold conditioning 4-node candidate GH4

Frozen V37bH18 scientific mechanism: deterministic rank-one Gaussian input conditioning. Weight-only expected final-mean Jacobian uses a cheap diagonal Gaussian skeleton. Forward weight storage is input-by-output (`w.T @ mean`); backward Jacobian uses `w @ (gate*gradient)`. Direction constructed using legal billed operations; no reference or extra full parent pilot.

At node t, first preactivation mean is `t*(w0.T@u)` and covariance `w0.T@w0-(w0.T@u)(w0.T@u).T`. 4 serial complete parent calls are combined by fixed standardnormal GH4; all use the original FLOP/time context, estimator setup runs once, no resets or nested BudgetContext. Persistent pool is reused but all per-node scientific state is local and the returned stacked prediction/new weighted sum owns its memory.

SHA256: `4cdabc281dea802c1aa89c829ceb7d2e94b0bc3b065415f86acf60adbb3d6b01`. LICENSE/NOTICE preserved from V37bH18.

Shape guard: only width1024/depth16 uses conditional wrapper. Generic shapes run original parent algorithm. Direction float64 normalization then float32 cast gives unit norm to parent arithmetic tolerance; input covariance minimum eigenvalue may differ from zero by about float32 roundoff. Conditional firstlinear covariance remains symmetric; variance is floored through the original parent guard.

Unconditional empirical mean rider, LAM/REF_R and metric regeneration, source confinement and shape-based pruning remain unchanged. Conditional K3 model and finite Gaussian quadrature remain approximations; GH4 moment exactness does not imply exact deep ReLU expectation. No parameter fitting/tuning to scores; no runtime oracle.

GH4 can exceed 120-second budget with four parent calls. GH2 isolates fixed quadrature order4→2 to assess a feasible pilot; GH2 integration bias can exceed GH4. A resource failure must remain recorded. No score/resource claim before strict runner finishes.

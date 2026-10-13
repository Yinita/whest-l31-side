# r20 read-only correctness review

Reviewed frozen candidate SHA `c50fd5d81ca67fdfd6b5163b82c5970b7f371c185cb6cd396ba5ee5722b57de0`; no source modifications and no heavy run.

No actionable implementation bug found in reviewed birth extraction, source lifetime,
transport ordering or D3/D21 contraction. The remainder is
`Mrem=3*(Rres-Q@Bm).T`, carried as `Sym(Mrem,P,P)`. Its diagonal must be
preserved: Rres has zero diagonal while Q@Bm generally does not. The added
remainder cancels the parent's approximate M diagonal at birth; zeroing it
would defeat restoration.

Rres aliases the S21 pooled buffer, but the extracted matrix is copied immediately
to an independent birth stack after Q/Bm are computed. Following parent writes
read Bm and Q; they do not overwrite the stored remainder. Subsequent layer
S21/temporary reuse therefore does not invalidate birth history.

The helper packs `P²/3` against M and `2MP/3` against P, producing
D21=(P²)M.T/3+2(MP)P.T/3; D3=rowsum(MP²). This is exactly the repeated slice
of the six-permutation-average CP(M,P,P), including the coefficient3 already
folded into M. First/newborn transport uses rawW; older slots useWdiag(w1).
The same relabeled fullW is used, without separately applying gain a second time.

The stated approximation boundary remains material: the retained parent's
Q@Bm history uses parent pruned/confined P, while the new discarded-part
history uses independent fullP. Their sum therefore is not the fullyunconfined
Rres source at all later ages. It accurately restores birth discarded content
under its new full transport and does not duplicate the retained birth.
This limitation does not invalidate the requested single-factor experiment.

Line pointers (one-based):

- `def residual_remainder_slices`: 835
- `Rres = fnp.subtract`: 1827
- `Bm = fnp.matmul`: 1835
- `fnp.subtract(Rres, NN("rem_approx")`: 1841
- `if rem_stack is not None:`: 1423, 1532
- `rd3, rd21 = residual_remainder_slices`: 1533

Only the actual strict1024 result may establish score/resource acceptance.

Root returned strict1 result:0failure, raw2.1447732479e-8 versus parent2.2281179568e-8 (−3.7406%), adjusted1.0426297154e-8 (+114.03%),1,069,002,046,455FLOPs,wall74.02816s,residual.264957639s. This is a measurable raw improvement but cost rejectspromotion and tenfold target remains unmet. No rank scan or additional heavy run was launched.

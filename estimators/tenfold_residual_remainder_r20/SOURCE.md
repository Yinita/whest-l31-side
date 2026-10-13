# Residual remainder r20

Parent V37bH18 SHA256 `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`.
Candidate SHA256 `c50fd5d81ca67fdfd6b5163b82c5970b7f371c185cb6cd396ba5ee5722b57de0`.

The sole mechanism restores the discarded birth residual as
`Sym(3*(Rres-Q@Bm).T, I, I)` in an independent two-leg full-width history.
Its diagonal remains present to cancel the approximate parent's spurious
`Q@Bm` diagonal. New births use raw W; older histories use W times the previous
expected derivative, with the same index relabeling. Parent covariance, fourth
cumulant proxy, feedback, pruning, and confinement remain inherited.

This restores the discarded birth component, not an exact full M replacement:
the parent P history is subsequently confined, while the extra P history is full.
There is no reference dependence, fitted scale, compiled kernel, concurrency, or
meter modification. Original MIT license and notice are retained.

Independent n4/8/32 tensor contracts through three ages agree within 1.55e-15;
n32 depth4 full prediction is finite. The strict development RPC network0 is
zero-failure: raw 2.2281179568e-8 -> 2.1447732479e-8 (-3.7406%), but adjusted
4.8714160494e-9 -> 1.0426297154e-8 (+114.03%), at 1,069,002,046,455 FLOPs.
The version is rejected for promotion. Its small raw signal does not justify
a rank or scale sweep and does not reach the tenfold goal.

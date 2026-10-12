# O12 represented CP edge exact contraction

For each CP rank and each of three edge orientations set x=W*a*u,
y=W*b*v, z=W*c*v, where multiplication scales input columns. Covariance C
has its diagonal zeroed once. The strict-distinct contraction for each output is

`sum(x)*sum(y*(z@C.T)) - sum(x*y*(z@C.T)) - sum(x*z*(y@C))`.

The first term includes all i,j,k with j!=k. The two subtractions exclude i=j
and i=k. Their intersection has i=j=k and is already zero in C, so no extra
intersection term is needed. Permutation-average symmetric CP gives the same
output-diagonal contraction because W_oi W_oj W_ok is symmetric. Derivative
vectors u,v are explicit inputs: target H29 uses u=w1,v=w2; earlier real-state
r06 used u=w2,v=w1 and does not validate that target.

Every factor, cz and cy has shape (output_slab,n); C and W have shape(n,n).
The implementation never allocates a candidate n³ tensor. Fullbatch has at most
n² elements per individual intermediate, with multiple simultaneous arrays:
this is not a claim of only one n² array resident. Dense oracle is offline only.
There are two dense matrix products per orientation, three orientations/rank:
6R GEMMs with leading billed cost 12R n³ under 2*m*k*n counting. Elementwise,
reductions, mask and input transfer add O(R n²+n²+Rn). Source construction and
all-source R growth must be charged separately. Arbitrary full K111 need not
have small R; no claim of universally O(n³) exact K111 propagation follows.

Measured fullbatch n32/R4 cost1737744, n64/R8 cost26464032; independent dense
oracle relative error1.62e-6/1.87e-6. Corrected r18 shape metadata is n²; r17
had a stale conservative n³ label despite only 2D production arrays. Preserved.
Production n1024/R16 measured206817513984 FLOPs,1.73510s,residual0.015146s
(r16); synthetic represented-state component only. These results complete the
O12 cost/shape diagnostic, not H29 source coverage, candidate accuracy or full
network resource acceptance. No algorithm before/after score is applicable.

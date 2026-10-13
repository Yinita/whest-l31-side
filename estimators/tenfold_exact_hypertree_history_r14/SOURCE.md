# r14 exact hypertree history

Parent V37bH18: `c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2`.
Candidate: `832bb894ca238fa2014d963ab1f81c3ca08bb6d27de7434bef9eec70d9e4d7ad`.

The primary change restores **all cost3 heavy+light K111 hypertree paths at every
birth**, replacing the parent's rank8 feedback product. These are the official
left/right leg types K21/2, K12/2, K22/4 and doubled covariance/2 paired with a
light covariance edge. It does not add the non-hypertree triangle or K111×edge.

For zero-diagonal symmetric covariance C and K22 G, D21[i,j]=K3[i,i,j]:

```
A = diag(w1) C
H = .5 diag(w1) D21.T diag(w3)
  + .5 diag(w2) D21 diag(w2)
  + .25 diag(w2) G diag(w3)
  + .5 diag(w2) (C*C) diag(w3)
M = -6 (A*H).T
T_new = 6 Sym(A,I,H) + Sym(M,I,I)
```

Sym averages six permutations, and * is elementwise. The paired heavy-left and
heavy-right official paths are equal after CP factor permutation, which yields
this single-H representation. Local C/G symmetrization canonicalizes inherited
float32 antisymmetric roundoff; parent covariance objects remain unchanged.
The M term removes all repeated birth indices exactly, so added birth D3/D21
are zero. It must not also be subtracted from the parent's M residual.

Four full-row matrices A/P/H/Z=P M are retained per birth, with no new age,
rank or row truncation. At the first transport use raw W; subsequent transports
use W diag(previous expected ReLU derivative), including the parent's index
relabeling. The parent keeps its own approximate A/P confinement/pruning;
the independent extra history is exact for the restored birth tensor under
this expected-derivative transport. Original thin D21 feedback is fixed off
because its retained pair paths overlap, and its product contains cost4 terms.

Readout uses D3=6 sum(A*P*H)+sum(Z*P*P), and four matrix contractions for D21:

```
[2(P*H)] A.T
+ [2(A*H)+(2/3)(Z*P)] P.T
+ [2(A*P)] H.T
+ [(P*P)/3] Z.T
```

Births and readouts are packed into <=12-matrix legal Strassen families using
the inherited array-only helper. No compiled kernel, concurrency, meter change
or oracle enters the candidate. Buffers preserve negative evidence and contracts.

Contracts: official independent path loops and explicit dense tensors n4/7/12/32
agree through three ages; real public-network0 cached n8/32 birth agrees. The
separate FP32 n64 level1 Strassen contraction relative errors are below4e-7.
Small n32/n64 depth4 predict smoke is finite. An initial n8 depth16 smoke became
nonfinite; its executed source and failure are retained and do not qualify as
resource acceptance. Strict1024 score/resource/RPC screening is pending.

Unmodified inherited limitations: covariance and K4 proxies, rank16 M residual,
Gaussian reference tables and closure truncation. No empirical coefficient or
truth-selected rank is introduced. No tenfold success is claimed.

StrictRPC1 actual screening is negative: raw2.2287540702e-8 (+0.02855%), adjusted1.6548841062e-8 (+239.71%),1,632,808,519,965FLOPs, wall99.4213s/residual.28660s, zero failures. All-history heavy-tree restoration does not explain the tenfold gap; no age/rank/leg-cost extension is selected.

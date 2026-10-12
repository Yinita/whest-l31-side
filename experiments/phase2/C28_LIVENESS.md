# C28 r02 final-only scratch liveness

At _dslices entry need_d21=False occurs only final trim. LA/LP are assigned
only in the guarded need_d21 block. All later LA/LP uses occur after the early
`return D3,None`. Therefore lap[:,0] and lap[:,1] are dead for the complete
final _dslices call and can receive Xt and Yt. lap storage is independent of
AP0/AP1 source legs and ap/pp/mp/t/u feed scratch. A shallow local buffer map
keeps persistent buffer identities unchanged for the next network predict.

AP and PP remain live through feed R; MP/T/U have their existing uses. No
source/view alias is overwritten. Arithmetic and charged calls stay unchanged.
This uses allocated dead scratch, so it is not a claim of peak-memory reduction:
xt/yt pool entries remain allocated for previous layers and later networks.
Acceptance still needs small allocation-poison contract and real local/RPC
zero failures with identical FLOPs. This document alone does not complete C28.

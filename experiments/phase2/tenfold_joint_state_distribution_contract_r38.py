"""Finite-distribution evidence for admissible post-state transformations.

No benchmark/reference access; arbitrary deterministic finite distribution.
Proves central/raw consistency and covariance/source transport ledger.
"""
import argparse,hashlib,itertools,json,math
from pathlib import Path
import numpy as np


def partition(n):
 if n==0:return [()]
 out=[]
 for p in partition(n-1):
  out.append(p+((n-1,),))
  for j in range(len(p)):out.append(p[:j]+(p[j]+(n-1,),)+p[j+1:])
 return out


def cumulant(y,w,idx):
 result=0.
 for p in partition(len(idx)):
  term=(-1)**(len(p)-1)*math.factorial(len(p)-1)
  for block in p:term*=w@np.prod(y[:,[idx[t] for t in block]],axis=1)
  result+=term
 return result


def states(y,w):
 n=y.shape[1];mu=w@y
 c=np.empty((n,n));k3=np.empty((n,)*3);k4=np.empty((n,)*4)
 for k,t in ((2,c),(3,k3),(4,k4)):
  for idx in itertools.product(range(n),repeat=k):t[idx]=cumulant(y,w,idx)
 return mu,c,k3,k4


def raw(y,w,powers):return w@np.prod(y**powers,axis=1)


def run():
 rng=np.random.default_rng(261014145);n=3;y=np.maximum(rng.normal(size=(23,n)),0);w=np.arange(1,24,dtype=float);w/=w.sum();base=states(y,w)
 b=np.array([.23,-.17,.09]);a=np.array([.8,1.2,1.1]);shift=states(y+b,w);affine=states(y*a+b,w)
 errors={'shift_mean':float(np.max(abs(shift[0]-base[0]-b)))}
 for degree in (2,3,4):
  errors[f'shiftK{degree}']=float(np.max(abs(shift[degree-1]-base[degree-1])))
  factor=np.prod(np.stack(np.meshgrid(*[a]*degree,indexing='ij')),axis=0)
  errors[f'affineK{degree}']=float(np.max(abs(affine[degree-1]-base[degree-1]*factor)))
 # Centering mixed raw moment with shifted means but unchanged pKs is wrong.
 raw11=np.array([[raw(y,w,np.eye(n,dtype=int)[i]+np.eye(n,dtype=int)[j]) for j in range(n)] for i in range(n)])
 wrongc=raw11-shift[0][:,None]*shift[0][None,:]
 wrongrawmix=float(np.max(abs(wrongc-base[1])))
 # Power-cumulants transform nontrivially under shift but ordinary mixed
 # central cumulants remain invariant; keepbothledgers and verify formula.
 def power_pair(t,mi,mj,i,j):
  u=t[:,i]**mi;v=t[:,j]**mj
  return float(w@(u*v)-(w@u)*(w@v))
 for i,j in itertools.product(range(n),repeat=2):
  if i==j:continue
  for name,t,m in [('base',y,base[0]),('shift',y+b,shift[0])]:
   pk11=power_pair(t,1,1,i,j);pk21=power_pair(t,2,1,i,j);pk12=power_pair(t,1,2,i,j);pk22=power_pair(t,2,2,i,j)
   k21=pk21-2*m[i]*pk11
   k22=pk22-2*m[i]*pk12-2*m[j]*pk21+4*m[i]*m[j]*pk11-2*pk11**2
   ref=base if name=='base' else shift
   errors[f'{name}_rawcenter21_{i}_{j}']=abs(k21-ref[2][i,i,j])
   errors[f'{name}_rawcenter22_{i}_{j}']=abs(k22-ref[3][i,i,j,j])
 # Genuine independent vector noise via full Cartesian finite product.
 z=np.array([[-.3,.2,-.1],[.1,-.4,.2],[.6,.1,.5],[-.1,.1,-.3]])
 wz=np.array([.2,.3,.1,.4]);z=z-wz@z
 # independent coordinates, individual 4point noise; cross-noise vanish
 zz=np.array(list(itertools.product(*[z[:,i] for i in range(n)])))
 wwz=np.array([math.prod(wz[i] for i in ids) for ids in itertools.product(range(4),repeat=n)])
 mixed=(y[:,None,:]+zz[None,:,:]).reshape(-1,n);wm=(w[:,None]*wwz).reshape(-1)
 noise=states(zz,wwz);added=states(mixed,wm)
 for degree in (1,2,3,4):errors[f'independentNoiseK{degree}']=float(np.max(abs(added[degree-1]-base[degree-1]-noise[degree-1])))
 # Marginal moment matrix PSD (Hamburger order2 necessary/sufficient).
 m2=np.diag(noise[1]);m3=np.einsum('iii->i',noise[2]);k4=np.einsum('iiii->i',noise[3]);m4=k4+3*m2*m2
 determinant=m2*m4-m3*m3-m2**3
 psdmin=[]
 for v,s,t in zip(m2,m3,m4):psdmin.append(float(np.linalg.eigvalsh([[1,0,v],[0,v,s],[v,s,t]])[0]))
 # Source CP3 row-affine central-cumulant scaling, carrierC and nextlinear.
 q=rng.normal(size=(n,5));p=rng.normal(size=(n,5));r=rng.normal(size=(n,5));source=np.einsum('ir,jr,kr->ijk',q,p,r)
 scaled=np.einsum('ir,jr,kr->ijk',a[:,None]*q,a[:,None]*p,a[:,None]*r)
 expected=source*a[:,None,None]*a[None,:,None]*a[None,None,:]
 errors['source3_affine']=float(np.max(abs(scaled-expected)))
 mat=rng.normal(size=(n,n));nextmean=mat@(a*base[0]+b);nextcov=(mat*a[None,:])@base[1]@(mat*a[None,:]).T
 errors['nextlinear_mean']=float(np.max(abs(nextmean-w@((y*a+b)@mat.T))))
 centered=(y*a+b)@mat.T-nextmean
 errors['nextlinear_cov']=float(np.max(abs(nextcov-(centered*w[:,None]).T@centered)))
 assert max(errors.values())<3e-12,errors
 return {'maxerrors':errors,'wrong_fixedraw_changedmean_cov_error':wrongrawmix,'noisevariance':m2.tolist(),'noisecumulant3':m3.tolist(),'noisecumulant4':k4.tolist(),'noiseorder2moment_determinants':determinant.tolist(),'noiseorder2moment_min_eigenvalues':psdmin,
  'state_semantics':{'shift':'K1+=b; allcentralK>=2 unchanged incl mixed; rawpowers must transform only if recomputing moments','affine':'K1=aK1+b; everyjointcentralκi..*=product(a_indices), fullC/sourcelegs scaling required','independentnoise':'allcentralκ addnoise; independentcoordinate noise changesonlydiagonalκ>=2; noiseκ2nonnegative+momentPSD essential'},
  'birth_ledger':{'postshift_after_center':'centralK21/K22 stayoriginal; keep original raw pk1/w12 for originalReLU birth calculation, sourcecentraldistribution unchanged','affine':'postprocessedraw/powerreconstruction or scale finalfullyassembledC/K3/K4history, notchange4diagonly','independentnoise':'adddiagonalC/K3/K4birth channels; ordinaryM residual must own newK3diag once; trueK4history or regenproxy approximation isexplicit'},
  'limits':['finite distribution algebra only, no benchmark truth','necessary marginalmomentPSD doesnotprove arbitrarymixedcumulants jointlyvalid','independentcoordinate noise restricts δvariance>=0; arbitraryfourridgechannels notequivalent','shift canmakenonnegativeReLUoutputnegative; requireb>=0 or perneuronessinfbound foractualnonnegativesupport']}


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();r={'version':'tenfold-joint-state-distribution-contract-r38','complete':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**run(),'no_benchmark_truth':True}
 a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({k:r[k] for k in ('maxerrors','wrong_fixedraw_changedmean_cov_error','noiseorder2moment_min_eigenvalues')}))
if __name__=='__main__':main()

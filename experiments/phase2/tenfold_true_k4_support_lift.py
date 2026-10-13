"""Exact 4/31/22 support lift; legal matrix-free transport/contractions.

Stores disjoint symmetric tensor support. Only index metadata uses NumPy.
No reference access. Dense n4 tensors are prohibited in this component.
Full transported D22 is O(m n3); all D31/trace contractions are O(n3).
"""
import flopscope.numpy as fnp
import numpy as np


def zero_diag(a):
    return a-fnp.diag(fnp.diag(a))


class SupportLift:
    def __init__(self,d4,d31,d22):
        self.d4=d4;self.d31=zero_diag(d31)
        self.d22=zero_diag((d22+d22.T)/2);self.n=len(d4)

    @classmethod
    def birth(cls,producer):
        """Full retained birth slices minus actual leading-gain history."""
        from experiments.phase2.tenfold_true_k4_producer import pair_entry
        n=producer.n;i,j=np.indices((n,n));gain=producer.wicks[(1,1)]
        i4=np.arange(n);sets=[(i4,)*4,(i,i,i,j),(i,i,j,j)];values=[]
        for idx in sets:
            value=producer.entries(idx)
            for a,b in producer.pairs:
                factor=gain[idx[0]]*gain[idx[1]]*gain[idx[2]]*gain[idx[3]]
                value=value-pair_entry(a,b,idx)*factor
            values.append(value)
        return cls(*values)

    def entries(self,idx):
        """Partition masks are exclusive: 4, 31, 22; other support zero."""
        a,b,c,d=idx;shape=np.broadcast_shapes(*(np.shape(x) for x in idx))
        a,b,c,d=np.broadcast_arrays(a,b,c,d)
        out=fnp.zeros(shape,dtype=fnp.float64)
        out=out+self.d4[a]*((a==b)&(a==c)&(a==d))
        for r,s,t,u in ((a,b,c,d),(a,b,d,c),(a,c,d,b),(b,c,d,a)):
            out=out+self.d31[r,u]*((r==s)&(r==t)&(r!=u))
        for r,s,t,u in ((a,b,c,d),(a,c,b,d),(a,d,b,c)):
            out=out+self.d22[r,t]*((r==s)&(t==u)&(r!=t))
        return out

    def apply(self,b):
        """H:B source-space, exact O(n2), including all collisions."""
        diagonal=self.d4*fnp.diag(b)+2*fnp.sum(self.d31*b,axis=1)+self.d22@fnp.diag(b)
        off=self.d31*fnp.diag(b)[:,None]+self.d31.T*fnp.diag(b)[None,:]+2*self.d22*b
        return off+fnp.diag(diagonal)

    def transported_apply(self,w,b):
        return w@self.apply(w.T@b@w)@w.T

    def transported_d4_d31(self,w):
        """All H'(aaab), H'(aaaa); three m-by-n GEMMs plus readout."""
        w2=w*w;w3=w2*w
        grad=w3*self.d4+3*w2*(w@self.d31.T)+w3@self.d31+3*w*(w2@self.d22)
        d31=grad@w.T
        return fnp.diag(d31),d31

    def transported_d22(self,w):
        """Exact bounded stream: one source n2 + m-by-n multiply per output.

        Cannot claim O(n3) for arbitrary dense source D22: all rows O(m2n2).
        """
        return fnp.stack([fnp.sum((w@self.apply(w[a,:,None]*w[a,None,:]))*w,axis=1) for a in range(len(w))])

    def transported_p_full(self,w,z):
        """P_jk=sum_i z_i H'(i j j k), includes repeated i."""
        v=w.T@z;w2=w*w
        l=w2*(self.d4*v)+2*(w*v)*(w@self.d31.T)+w2*(self.d31@v)
        l=l+(w2*v)@self.d31+v*(w2@self.d22)+2*w*((w*v)@self.d22)
        return l@w.T

    def use_and_caps(self,w,z):
        d4,d31=self.transported_d4_d31(w);d22=self.transported_d22(w)
        f=self.transported_apply(w,fnp.diag(z))-z[:,None]*d31-z[None,:]*d31.T
        p=self.transported_p_full(w,z)-z[:,None]*d31-z[None,:]*d22
        return d4,zero_diag(d31),zero_diag(d22),zero_diag(f),zero_diag(p)

    def all_feed(self,w,mean,w1,w2):
        """Exact transported support-lift feed marginal/current-centering.

        Full feed histories need a separate state; these are current USE only.
        """
        d4,d31=self.transported_d4_d31(w);d22=self.transported_d22(w)
        zf=mean*w2;zp=mean*w1
        f=zero_diag(self.transported_apply(w,fnp.diag(zf))-zf[:,None]*d31-zf[None,:]*d31.T)
        p=zero_diag(self.transported_p_full(w,zp)-zp[:,None]*d31-zp[None,:]*d22)
        feed3=1.5*w2*w1*w1*d4
        feed21=zero_diag(w2[:,None]*w1[:,None]*w1[None,:]*d31+.5*w1[:,None]**2*w2[None,:]*d22)
        center=-(w1[:,None]*w1[None,:]*f+w2[:,None]*w1[None,:]*p+w1[:,None]*w2[None,:]*p.T)
        return feed3,feed21,center

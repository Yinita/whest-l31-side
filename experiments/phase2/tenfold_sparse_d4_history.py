"""Exact CP4 diagonal-support histories, one shared leg per source.

USE and all-three feed current contractions are exact for this representation.
Birth retains only D4 residual; D31/D22/211/1111 birth support is omitted.
Not a full estimator and no test reference access. No dense higher-order arrays.
"""
import flopscope.numpy as fnp


def zero_diag(a):return a-fnp.diag(fnp.diag(a))


class DiagonalHistory:
    def __init__(self,legs,coefs):self.legs=legs;self.coefs=coefs

    @classmethod
    def born(cls,d4):return cls(fnp.eye(len(d4),dtype=d4.dtype),d4)

    def move(self,w,gain=None):
        operator=w if gain is None else w*gain[None,:]
        return DiagonalHistory(operator@self.legs,self.coefs)

    def slices(self):
        q=self.legs;q2=q*q;q3=q2*q
        d31=(q3*self.coefs)@q.T
        d22=(q2*self.coefs)@q2.T
        return fnp.diag(d31),d31,d22

    def apply(self,b):
        q=self.legs
        weighted=self.coefs*fnp.sum(q*(b@q),axis=0)
        return (q*weighted)@q.T

    def caps(self,z):
        q=self.legs;q2=q*q;_,d31,d22=self.slices()
        f=(q*(self.coefs*(q2.T@z)))@q.T
        p=(q2*(self.coefs*(q.T@z)))@q.T
        return zero_diag(f-z[:,None]*d31-z[None,:]*d31.T),zero_diag(p-z[:,None]*d31-z[None,:]*d22)

    def use_feed(self,mean,w1,w2):
        d4,d31,d22=self.slices();f,_=self.caps(mean*w2);_,p=self.caps(mean*w1)
        feed3=1.5*w2*w1*w1*d4
        feed21=zero_diag(w2[:,None]*w1[:,None]*w1[None,:]*d31+.5*w1[:,None]**2*w2[None,:]*d22)
        center=-(w1[:,None]*w1[None,:]*f+w2[:,None]*w1[None,:]*p+w1[:,None]*w2[None,:]*p.T)
        return d4,zero_diag(d31),zero_diag(d22),feed3,feed21,center

    def add_exact_d4_birth(self,target_d4):
        # Retained model's full produced ordinary D4 minus actual inherited D4.
        # Never adds target_d4 on top of a full old univariate diagonal.
        inherited=self.slices()[0]
        n=len(target_d4)
        return DiagonalHistory(fnp.concatenate([self.legs,fnp.eye(n,dtype=target_d4.dtype)],axis=1),
                               fnp.concatenate([self.coefs,target_d4-inherited]))

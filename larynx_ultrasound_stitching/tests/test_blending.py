import numpy as np
from src.blending import linear_blend
def test_constant_overlap_is_finite_and_bounded():
 a=np.zeros((20,30),np.float32);b=np.ones_like(a);m1=np.zeros_like(a,bool);m2=m1.copy();m1[:,:20]=1;m2[:,10:]=1;f,w=linear_blend(a,b,m1,m2);assert np.isfinite(f).all();assert f.min()>=0 and f.max()<=1;assert np.allclose(w[m1&~m2],1)

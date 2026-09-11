import numpy as np
from src.saliency import compute_saliency,saliency_variance
def test_saliency_and_eq10_variance():
 x=np.zeros((64,64),np.float32);x[30:34,30:34]=1;s=compute_saliency(x);assert s.shape==x.shape and 0<=s.min()<=s.max()<=1;assert np.isclose(saliency_variance(np.array([[0,0],[2,0]],float)),1)

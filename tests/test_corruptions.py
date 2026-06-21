import numpy as np
from csi_er.corruptions import add_gaussian_noise

def test_gaussian_noise_shape_seed_copy_and_severity():
    x=np.ones((4,1,10,8),dtype=np.float32); x[:,0]+=np.linspace(0,1,10)[:,None]
    y1=add_gaussian_noise(x,1,seed=1); y1b=add_gaussian_noise(x,1,seed=1); y5=add_gaussian_noise(x,5,seed=1)
    assert y1.shape==x.shape
    assert np.allclose(y1,y1b)
    assert not np.shares_memory(x,y1)
    assert np.std(y5-x)>np.std(y1-x)
    assert np.allclose(x, np.ones((4,1,10,8),dtype=np.float32)+np.linspace(0,1,10,dtype=np.float32)[None,None,:,None])

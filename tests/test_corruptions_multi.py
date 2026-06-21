import numpy as np
from csi_er.corruptions import apply_corruption, CORRUPTIONS

def test_multi_corruptions_seed_shape_copy_and_strength():
    x=np.random.default_rng(0).normal(size=(4,1,20,10)).astype('float32')
    for c in CORRUPTIONS:
        y1=apply_corruption(x,c,1,seed=7); y1b=apply_corruption(x,c,1,seed=7); y5=apply_corruption(x,c,5,seed=7)
        assert y1.shape==x.shape
        assert np.allclose(y1,y1b)
        assert not np.shares_memory(x,y1)
        assert np.mean(np.abs(y5-x)) >= np.mean(np.abs(y1-x)) - 1e-6
        assert np.isfinite(y5).all()

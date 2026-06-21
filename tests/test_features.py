import numpy as np
from csi_er.features import time_stats_features, fft_features, PCAFeatureExtractor

def test_features_finite_and_pca_no_leakage():
    rng=np.random.default_rng(0); xtr=rng.normal(size=(12,1,20,10)).astype('float32'); xte=rng.normal(size=(5,1,20,10)).astype('float32')
    ts=time_stats_features(xtr); ff=fft_features(xtr)
    assert np.isfinite(ts).all(); assert np.isfinite(ff).all(); assert ff.shape[1]==9
    p=PCAFeatureExtractor(n_components=4).fit(xtr); ztr=p.transform(xtr); zte=p.transform(xte)
    assert ztr.shape==(12,4); assert zte.shape==(5,4)
    assert not hasattr(p.pca, 'test_seen_')

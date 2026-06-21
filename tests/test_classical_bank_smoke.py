from csi_robustbench.features import FeatureTransformer
from csi_robustbench.reference_bank import build_reference_configs, load_yaml
import numpy as np


def test_feature_transformers_are_finite():
    x = np.random.default_rng(0).normal(size=(12, 1, 32, 16)).astype("float32")
    cfg = load_yaml("configs/reference_bank.yaml")
    rows = build_reference_configs(cfg)
    seen = {}
    for row in rows:
        if row.feature_family not in seen:
            seen[row.feature_family] = row
    assert set(seen) == {"time_stats", "fft_stats", "dwt_energy", "stft_bands", "autocorrelation", "pca_raw"}
    for row in seen.values():
        z = FeatureTransformer(row.feature_family, row.feature_params, seed=0).fit_transform(x)
        assert z.shape[0] == len(x)
        assert z.shape[1] > 0
        assert np.isfinite(z).all()

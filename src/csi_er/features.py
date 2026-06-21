from __future__ import annotations
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA

class PCAFeatureExtractor:
    def __init__(self, n_components=64, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None
    def fit(self, X):
        flat = np.asarray(X).reshape(len(X), -1)
        n = max(1, min(self.n_components, flat.shape[0], flat.shape[1]))
        self.pca = PCA(n_components=n, random_state=self.random_state)
        self.pca.fit(flat)
        return self
    def transform(self, X):
        if self.pca is None:
            raise RuntimeError('PCAFeatureExtractor must be fit first')
        return self.pca.transform(np.asarray(X).reshape(len(X), -1)).astype(np.float32)
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def _finite(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def time_stats_features(X):
    A = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    q75 = np.percentile(A, 75, axis=1)
    q25 = np.percentile(A, 25, axis=1)
    feats = [
        A.mean(1), A.std(1), A.min(1), A.max(1), np.median(A, axis=1), q75 - q25,
        np.sqrt(np.mean(A * A, axis=1)), np.mean(A * A, axis=1), skew(A, axis=1), kurtosis(A, axis=1),
    ]
    B = np.asarray(X, dtype=np.float32)
    if B.ndim == 4:
        B = B[:, 0]
    per_time = B.mean(axis=2)
    per_sub = B.mean(axis=1)
    for M in [per_time, per_sub]:
        feats += [M.mean(1), M.std(1), M.min(1), M.max(1)]
    return _finite(np.vstack(feats).T)

def fft_features(X, top_k=5):
    B = np.asarray(X, dtype=np.float32)
    if B.ndim == 4:
        B = B[:, 0]
    spec = np.abs(np.fft.rfft(B, axis=1))
    energy = spec ** 2 + 1e-8
    nfreq = spec.shape[1]
    thirds = np.array_split(np.arange(nfreq), 3)
    total = energy.sum(axis=(1, 2))
    feats = []
    for idx in thirds:
        feats.append(energy[:, idx, :].sum(axis=(1, 2)) / total)
    freqs = np.arange(nfreq, dtype=np.float32)[None, :, None]
    centroid = (freqs * energy).sum(axis=(1, 2)) / total
    bandwidth = np.sqrt((((freqs - centroid[:, None, None]) ** 2) * energy).sum(axis=(1, 2)) / total)
    feats += [centroid, bandwidth]
    flat = spec.reshape(len(B), -1)
    top = np.sort(flat, axis=1)[:, -top_k:]
    feats += [top.mean(1), top.std(1), top.max(1)]
    dom = spec.mean(axis=2).argmax(axis=1).astype(np.float32)
    feats += [dom]
    return _finite(np.vstack(feats).T)

def fusion_features(X, pca_features=None):
    parts = [time_stats_features(X), fft_features(X)]
    if pca_features is not None:
        parts.append(pca_features)
    return _finite(np.concatenate(parts, axis=1))

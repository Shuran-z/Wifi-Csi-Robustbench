from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _as_csi(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 4 and a.shape[1] == 1:
        return a[:, 0]
    if a.ndim == 3:
        return a
    if a.ndim == 2:
        return a[:, None, :]
    raise ValueError(f"Expected CSI array with 3 or 4 dimensions, got {a.shape}")


def _stats(block: np.ndarray, names: list[str]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    axes = tuple(range(1, block.ndim))
    for name in names:
        if name == "mean":
            out.append(block.mean(axis=axes))
        elif name == "std":
            out.append(block.std(axis=axes))
        elif name == "min":
            out.append(block.min(axis=axes))
        elif name == "max":
            out.append(block.max(axis=axes))
        elif name == "median":
            out.append(np.median(block, axis=axes))
        elif name == "iqr":
            q75 = np.percentile(block, 75, axis=axes)
            q25 = np.percentile(block, 25, axis=axes)
            out.append(q75 - q25)
        elif name == "energy":
            out.append(np.mean(block * block, axis=axes))
        else:
            raise ValueError(f"Unknown statistic: {name}")
    return out


def _finite_features(parts: list[np.ndarray]) -> np.ndarray:
    feat = np.stack(parts, axis=1).astype(np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def time_stats_features(x: np.ndarray, *, segment_bins: int = 1, statistics: list[str] | None = None, **_: Any) -> np.ndarray:
    a = _as_csi(x)
    statistics = statistics or ["mean", "std", "min", "max", "median", "iqr"]
    parts: list[np.ndarray] = []
    for idx in np.array_split(np.arange(a.shape[1]), int(segment_bins)):
        if len(idx) == 0:
            continue
        parts.extend(_stats(a[:, idx, :], statistics))
    return _finite_features(parts)


def fft_stats_features(x: np.ndarray, *, bands: int = 4, statistics: list[str] | None = None, **_: Any) -> np.ndarray:
    a = _as_csi(x)
    statistics = statistics or ["mean", "std", "energy"]
    spectrum = np.abs(np.fft.rfft(a, axis=1)).astype(np.float32)
    parts: list[np.ndarray] = []
    for idx in np.array_split(np.arange(spectrum.shape[1]), int(bands)):
        if len(idx) == 0:
            continue
        parts.extend(_stats(spectrum[:, idx, :], statistics))
    return _finite_features(parts)


def dwt_energy_features(x: np.ndarray, *, levels: int = 2, wavelet: str = "haar", **_: Any) -> np.ndarray:
    del wavelet
    a = _as_csi(x).mean(axis=2)
    cur = a
    parts: list[np.ndarray] = []
    for _level in range(int(levels)):
        even = cur[:, 0::2]
        odd = cur[:, 1::2]
        m = min(even.shape[1], odd.shape[1])
        if m == 0:
            break
        avg = (even[:, :m] + odd[:, :m]) * 0.5
        diff = (even[:, :m] - odd[:, :m]) * 0.5
        parts.append(np.mean(diff * diff, axis=1))
        cur = avg
    parts.append(np.mean(cur * cur, axis=1))
    return _finite_features(parts)


def stft_bands_features(x: np.ndarray, *, window: int = 32, bands: int = 4, **_: Any) -> np.ndarray:
    a = _as_csi(x).mean(axis=2)
    win = max(8, min(int(window), a.shape[1]))
    step = max(1, win // 2)
    windows = []
    for start in range(0, max(1, a.shape[1] - win + 1), step):
        chunk = a[:, start : start + win]
        if chunk.shape[1] == win:
            windows.append(np.abs(np.fft.rfft(chunk, axis=1)).astype(np.float32))
    if not windows:
        windows = [np.abs(np.fft.rfft(a, axis=1)).astype(np.float32)]
    spec = np.stack(windows, axis=1)
    parts: list[np.ndarray] = []
    for idx in np.array_split(np.arange(spec.shape[2]), int(bands)):
        if len(idx) == 0:
            continue
        block = spec[:, :, idx]
        parts.append(block.mean(axis=(1, 2)))
        parts.append(block.std(axis=(1, 2)))
        parts.append(np.mean(block * block, axis=(1, 2)))
    return _finite_features(parts)


def autocorrelation_features(x: np.ndarray, *, lags: int = 8, normalize: bool = True, **_: Any) -> np.ndarray:
    a = _as_csi(x).mean(axis=2)
    a = a - a.mean(axis=1, keepdims=True)
    denom = np.sum(a * a, axis=1) + 1e-8
    parts: list[np.ndarray] = []
    for lag in range(1, int(lags) + 1):
        if lag >= a.shape[1]:
            break
        val = np.sum(a[:, :-lag] * a[:, lag:], axis=1)
        if normalize:
            val = val / denom
        parts.append(val)
    return _finite_features(parts)


@dataclass
class FeatureTransformer:
    feature_family: str
    params: dict[str, Any]
    seed: int = 42

    def fit(self, x: np.ndarray) -> "FeatureTransformer":
        if self.feature_family == "pca_raw":
            flat = _as_csi(x).reshape(len(x), -1)
            n_components = int(self.params.get("n_components", 16))
            n_components = max(1, min(n_components, flat.shape[0] - 1, flat.shape[1]))
            self.scaler_ = StandardScaler()
            z = self.scaler_.fit_transform(flat)
            self.pca_ = PCA(n_components=n_components, random_state=self.seed)
            self.pca_.fit(z)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.feature_family == "time_stats":
            return time_stats_features(x, **self.params)
        if self.feature_family == "fft_stats":
            return fft_stats_features(x, **self.params)
        if self.feature_family == "dwt_energy":
            return dwt_energy_features(x, **self.params)
        if self.feature_family == "stft_bands":
            return stft_bands_features(x, **self.params)
        if self.feature_family == "autocorrelation":
            return autocorrelation_features(x, **self.params)
        if self.feature_family == "pca_raw":
            flat = _as_csi(x).reshape(len(x), -1)
            return self.pca_.transform(self.scaler_.transform(flat)).astype(np.float32)
        raise ValueError(f"Unknown feature family: {self.feature_family}")

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
